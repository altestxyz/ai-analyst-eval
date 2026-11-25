import csv
import json

import pytest

from aae import headline, models, rescore, table, verify


def run(rid, model, ctx, q, correct, db="d", **kw):
    base = {
        "run_id": rid,
        "question_id": q,
        "model": model,
        "context": ctx,
        "db": db,
        "correct": correct,
        "repeat": 0,
        "asked": False,
        "had_error": False,
        "had_suspicious": False,
        "format_fail": False,
        "input_tokens": 1000,
        "cache_read_tokens": 0,
        "output_tokens": 100,
        "final_sql": "SELECT secret",
        "result_preview": [[1]],
        "timestamp": "t",
    }
    base.update(kw)
    return base


def test_table_never_carries_sql_or_results():
    row = table.row_for(run("1|L1|M1|0", "M1", "L1", "1", True), {})
    assert "final_sql" not in row and "result_preview" not in row and "attempts" not in row
    assert set(row) == set(table.COLUMNS)


def test_rescored_verdicts_replace_logged_ones_only_when_the_sql_was_rerun():
    rec = run("1|L1|M1|0", "M1", "L1", "1", False)
    rescored = {"1|L1|M1|0": {"run_id": "1|L1|M1|0", "status": "ok", "strict": False, "tol": True}}
    row = table.row_for(rec, rescored)
    assert (row["strict"], row["lenient"], row["rescore_status"]) == ("0", "1", "ok")
    row = table.row_for(rec, {"1|L1|M1|0": {"run_id": "1|L1|M1|0", "status": "no_db"}})
    assert (row["strict"], row["lenient"]) == ("0", "0")


def test_ambiguous_questions_are_not_scored():
    row = table.row_for(run("taxi-amb-01|L1|M1|0", "M1", "L1", "taxi-amb-01", None, db="nyc_taxi"), {})
    assert row["part"] == "ambiguous" and row["strict"] == ""


def test_tolerant_scorer_accepts_extra_columns_but_never_fewer():
    gold = [("a",), ("b",)]
    assert rescore.tolerant_same([(1, "a"), (2, "b")], gold, False)
    assert not rescore.tolerant_same([(1, "a"), (2, "c")], gold, False)
    assert not rescore.tolerant_same([("a", "x")], [("a", "x"), ("b", "y")], False)
    assert not rescore.tolerant_same([("a",)], [("a", "x")], False)


def test_summary_counts_agreement():
    results = [
        {"group": "g", "status": "ok", "strict": True, "tol": True, "logged_correct": True, "model": "M1"},
        {"group": "g", "status": "ok", "strict": False, "tol": True, "logged_correct": True, "model": "M1"},
        {"group": "g", "status": "no_db", "model": "M1", "logged_correct": False},
    ]
    s = rescore.summarize(results)
    assert s["rescored_runs"] == 2 and s["strict_agreement_pct"] == 50.0 and s["lenient_only_right"] == 1


def test_route_prefers_the_proxy_only_when_both_credentials_exist(monkeypatch):
    monkeypatch.delenv("AAE_ROUTE", raising=False)
    monkeypatch.delenv("MODEL_PROXY_URL", raising=False)
    monkeypatch.delenv("MODEL_PROXY_API_KEY", raising=False)
    assert models.route() == "direct"
    monkeypatch.setenv("MODEL_PROXY_URL", "http://proxy")
    assert models.route() == "direct"
    monkeypatch.setenv("MODEL_PROXY_API_KEY", "k")
    assert models.route() == "kproxy"
    monkeypatch.setenv("AAE_ROUTE", "direct")
    assert models.route() == "direct"


def test_hosted_models_resolve_to_the_chosen_route():
    cfg = {
        "provider": "hosted",
        "label": "X",
        "price_in": 1,
        "routes": {"kproxy": {"id": "vendor/x"}, "direct": {"id": "x", "base_url": "https://h/v1/", "api_key_env": "X_KEY"}},
    }
    a, b = models.resolve(cfg, "kproxy"), models.resolve(cfg, "direct")
    assert (a["provider"], a["id"]) == ("kproxy", "vendor/x")
    assert (b["provider"], b["id"], b["api_key_env"]) == ("api", "x", "X_KEY")
    assert models.resolve({"provider": "ollama", "id": "q"}, "direct")["provider"] == "ollama"


def test_the_shipped_model_file_resolves_on_both_routes():
    for chosen in ("kproxy", "direct"):
        loaded = models.load_models("models.yaml", chosen)
        assert loaded["F1"]["provider"] == ("kproxy" if chosen == "kproxy" else "api")
        assert loaded["M1"]["provider"] == "ollama"


def make_table(tmp_path):
    rows = []
    for q in range(10):
        for ctx, right in (("L1", q < 2), ("L2", q < 5), ("L3", q < 4)):
            rec = run(f"{q}|{ctx}|M1|0", "M1", ctx, str(q), right, db="bird_db")
            rows.append(table.row_for(rec, {}))
            rec = run(f"{q}|{ctx}|F1|0", "F1", ctx, str(q), right or q == 9, db="bird_db")
            rows.append(table.row_for(rec, {}))
    path = tmp_path / "t.csv"
    table.write(rows, path)
    return path


def test_headline_gain_matches_a_hand_count(tmp_path):
    df = headline.load_table(make_table(tmp_path))
    m = headline.compute_metrics(df, models.load_models("models.yaml", "direct"))
    assert m["acc.M1.L1.strict"] == pytest.approx(20.0)
    assert m["gain.M1.L2-L1.strict"] == pytest.approx(30.0)
    assert m["gain.M1.L3-L2.strict"] == pytest.approx(-10.0)
    assert m["runs.total"] == 60


def test_verify_passes_and_fails_by_rounding_tolerance(tmp_path):
    path = make_table(tmp_path)
    claims = tmp_path / "claims.json"
    claims.write_text(
        json.dumps(
            {
                "claims": [
                    {"metric": "gain.M1.L2-L1.strict", "expected": 30, "decimals": 0},
                    {"metric": "gain.M1.L3-L2.strict", "expected": -12, "decimals": 0},
                    {"metric": "no.such.metric", "expected": 1},
                ]
            }
        )
    )
    out = tmp_path / "out.json"
    rc = verify.main(["--table", str(path), "--claims", str(claims), "--rescore-summary", str(tmp_path / "none.json"), "--json", str(out)])
    status = [r["status"] for r in json.load(open(out))]
    assert rc == 1 and status == ["PASS", "FAIL", "MISSING"]


def test_verify_rounds_half_up_to_the_claimed_decimals():
    metrics = {"x": 12.5, "y": 39.47}
    claims = [
        {"metric": "x", "expected": 13, "decimals": 0},
        {"metric": "x", "expected": 12, "decimals": 0},
        {"metric": "y", "expected": 40, "decimals": 0},
        {"metric": "y", "expected": 39, "decimals": 0},
    ]
    assert [r[2] for r in verify.check(claims, metrics)] == ["PASS", "FAIL", "FAIL", "PASS"]


def test_the_shipped_claims_only_name_metrics_the_code_can_compute():
    names = {c["metric"] for c in json.load(open("plan/claims.json"))["claims"]}
    assert all(
        n.split(".")[0]
        in {
            "runs",
            "rescore",
            "list_cost_usd",
            "acc",
            "gain",
            "acc_challenging",
            "gain_taxi",
            "acc_taxi",
            "acc_shared",
            "gap_shared",
            "silent_wrong_pct",
            "lenient_minus_strict",
            "nudged",
            "cents_per_correct",
            "prompt_tokens_median",
            "extra_correct",
            "usd_per_extra_correct",
            "latency",
            "latency_median",
            "latency_p95",
            "all_wrong_shared",
            "wrong_by_shape",
            "frontier_misses",
            "gain_ci",
            "asked_pct",
            "asked_n",
            "taxi_frontier",
        }
        for n in names
    )


def test_table_csv_round_trips(tmp_path):
    path = make_table(tmp_path)
    with open(path) as f:
        assert len(list(csv.DictReader(f))) == 60
