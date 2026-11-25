import datetime
import json

import pandas as pd
import pytest

from aae import diagnose, publish, runner, shard
from aae.score import shape_of


def qs(n_bird=6, n_taxi=2, n_amb=2):
    out = [{"question_id": f"b{i}", "db": "d1"} for i in range(n_bird)]
    out += [{"question_id": f"t{i}", "db": "nyc_taxi"} for i in range(n_taxi)]
    out += [{"question_id": f"a{i}", "db": "nyc_taxi", "ambiguous": True} for i in range(n_amb)]
    return out


MODELS = ["F1", "F2"]


def all_jobs(q, models=MODELS):
    return set(shard.job_ids(q, models, 1))


def test_order_is_stable_and_bird_first():
    a = [q["question_id"] for q in shard.ordered_questions(qs(), 7)]
    b = [q["question_id"] for q in shard.ordered_questions(list(reversed(qs())), 7)]
    assert a[:6] == sorted(a[:6], key=a.index) and set(a[:6]) == {f"b{i}" for i in range(6)}
    assert a[6:8] and all(x.startswith("t") for x in a[6:8]) and all(x.startswith("a") for x in a[8:])
    assert len(a) == len(b) == 10


def test_slices_are_balanced_and_cover_everything_once():
    ids = [f"q{i}" for i in range(150)]
    s = shard.make_slices(ids, 40)
    assert [len(x) for x in s] == [38, 38, 37, 37]
    assert [i for x in s for i in x] == ids
    assert shard.make_slices([], 40) == [] and shard.make_slices(["a"], 40) == [["a"]]


def test_plan_skips_what_the_baseline_has_and_is_deterministic():
    questions = qs()
    done = set()
    for q in questions[:3]:
        done |= all_jobs(q)
    done.add(shard.run_id("b3", "L1", "F1", 0))
    p1 = shard.make_plan(questions, done, MODELS, 1, 3, 7)
    p2 = shard.make_plan(list(questions), done, MODELS, 1, 3, 7)
    flat = [i for s in p1["slices"] for i in s]
    assert len(flat) == 7 and not (set(flat) & {q["question_id"] for q in questions[:3]})
    assert p1["slices"] == p2["slices"] and p1["hash"] == p2["hash"]
    p3 = shard.make_plan(list(reversed(questions)), done, MODELS, 1, 3, 7)
    assert sorted(i for s in p3["slices"] for i in s) == sorted(flat)


def test_baseline_cutoff(tmp_path):
    p = tmp_path / "runs.jsonl"
    rows = [
        {"run_id": "a", "timestamp": "2026-11-18T10:00:00+00:00"},
        {"run_id": "b", "timestamp": "2026-11-20T10:00:00+00:00"},
        {"run_id": "c", "timestamp": "2026-11-18T10:00:00+00:00", "tag": "cache_warm"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n{half a line")
    assert shard.baseline_ids(p, "2026-11-20T00:00:00+00:00") == {"a"}
    assert shard.done_ids(p) == {"a", "b"}


def plan_and_by_id(done=()):
    questions = qs()
    return shard.make_plan(questions, set(done), MODELS, 1, 4, 7, addons=["cache", "effort"]), {q["question_id"]: q for q in questions}


def test_pick_by_date_and_by_hand():
    plan, by = plan_and_by_id()
    d0 = datetime.date(2026, 11, 20)
    c = shard.pick(plan, set(), by, "auto", "2026-11-20", d0)
    assert c["kind"] == "main" and c["index"] == 0 and c["qids"] == plan["slices"][0]
    c = shard.pick(plan, set(), by, "auto", "2026-11-20", d0 + datetime.timedelta(days=len(plan["slices"])))
    assert c["kind"] == "cache"
    c = shard.pick(plan, set(), by, "auto", "2026-11-20", d0 + datetime.timedelta(days=len(plan["slices"]) + 1))
    assert c["kind"] == "effort"
    assert shard.pick(plan, set(), by, "1", "2026-11-20", d0)["index"] == 1
    assert shard.pick(plan, set(), by, 99, "2026-11-20", d0)["kind"] == "none"
    assert shard.pick(plan, set(), by, "auto", "2026-11-20", d0 - datetime.timedelta(days=1))["kind"] == "none"


def test_pick_heals_a_gap_only_when_shards_are_visible():
    plan, by = plan_and_by_id()
    d = datetime.date(2026, 11, 20)
    done0 = set()
    for qid in plan["slices"][0]:
        done0 |= all_jobs(by[qid])
    c = shard.pick(plan, done0, by, "auto", "2026-11-20", d, have_shards=True)
    assert c["index"] == 1
    assert shard.pick(plan, done0, by, "auto", "2026-11-20", d, have_shards=False)["kind"] == "none"
    late = d + datetime.timedelta(days=30)
    assert shard.pick(plan, done0, by, "auto", "2026-11-20", late, have_shards=True)["index"] == 1
    everything = set()
    for q in by.values():
        everything |= all_jobs(q)
    c = shard.pick(plan, everything, by, "auto", "2026-11-20", late, have_shards=True, addon_done={"cache": True, "effort": True})
    assert c["kind"] == "none"


def test_check_plans_warns_on_different_hashes(tmp_path, capsys):
    (tmp_path / "shard_00_a.json").write_text(json.dumps({"plan_hash": "x"}))
    assert shard.check_plans(str(tmp_path))
    (tmp_path / "shard_01_b.json").write_text(json.dumps({"plan_hash": "y"}))
    assert not shard.check_plans(str(tmp_path))


@pytest.mark.parametrize(
    "pred,gold,want",
    [
        ([(1,), (2,)], [(1,), (2,)], "match"),
        ([(2,), (1,)], [(1,), (2,)], "order_only"),
        ([], [(1,)], "empty"),
        ([(1, 2)], [(1,)], "wrong_columns"),
        ([(1,), (2,), (3,)], [(1,), (2,)], "too_many_rows"),
        ([(1,)], [(1,), (2,)], "too_few_rows"),
        ([(1,), (9,)], [(1,), (2,)], "partial_overlap"),
        ([(8,), (9,)], [(1,), (2,)], "disjoint_values"),
        ([(0.5,)], [(0.5000001,)], "match"),
    ],
)
def test_shape_of(pred, gold, want):
    assert shape_of(pred, gold) == want


@pytest.fixture
def one_run(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    for lvl in ("L1", "L2", "L3"):
        (tmp_path / "d" / f"{lvl}.txt").write_text("schema")
    monkeypatch.setattr(runner, "CTX_DIR", tmp_path)
    monkeypatch.setitem(
        runner.MODELS,
        "P",
        {"provider": "kproxy", "id": "x/p", "price_in": 1.0, "price_out": 2.0, "max_output_tokens": 1024, "reasoning": "medium"},
    )

    def go(reply_rows, gold_rows):
        monkeypatch.setattr(
            runner,
            "chat",
            lambda cfg, m, repeat=0, **kw: {
                "text": "```sql\nSELECT 1\n```",
                "input_tokens": 100,
                "output_tokens": 50,
                "reasoning_tokens": 30,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "latency_s": 1.0,
            },
        )
        monkeypatch.setattr(runner, "run_sql", lambda con, s: {"columns": ["c"], "rows": reply_rows, "truncated": False})
        q = {"question_id": "q1", "db": "d", "question": "?", "difficulty": "simple"}
        return runner.run_one(q, "L1", "P", 0, {"q1": {"rows": gold_rows, "ordered": False}}, {"d": None})

    return go


def test_run_record_has_reasoning_shape_and_settings(one_run):
    r = one_run([(9,)], [(5,)])
    assert r["correct"] is False and r["shape"] == "disjoint_values"
    assert r["reasoning_tokens"] == 30 and r["max_output_tokens"] == 1024 and r["reasoning_setting"] == "medium"
    assert one_run([(5,)], [(5,)])["shape"] == "match"


def test_regressions_need_two_models_by_default():
    rows = []
    for m, l3 in (("F1", 0), ("F2", 0), ("F3", 1)):
        for ctx, ok in (("L2", 1), ("L3", l3)):
            rows.append({"question_id": "q1", "model": m, "context": ctx, "correct": float(ok), "tag": "", "run_id": f"q1|{ctx}|{m}|0"})
    for m in ("F1", "F2", "F3"):
        for ctx, ok in (("L2", 1), ("L3", 0 if m == "F1" else 1)):
            rows.append({"question_id": "q2", "model": m, "context": ctx, "correct": float(ok), "tag": "", "run_id": f"q2|{ctx}|{m}|0"})
    r = diagnose.regressions(pd.DataFrame(rows))
    assert [x["question_id"] for x in r] == ["q1"] and r[0]["models"] == ["F1", "F2"]


def test_publish_only_takes_new_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "runs/runs.jsonl").write_text('{"run_id": "a"}\n{"run_id": "b"}\n{"run_id": "c"}\n')
    (tmp_path / "runs/sample.jsonl").write_text('{"run_id": "z"}\n')
    restored = tmp_path / "restored.json"
    restored.write_text(json.dumps({"runs.jsonl": ["a", "b"]}))
    monkeypatch.setattr(publish, "RESTORED", restored)
    monkeypatch.setattr(publish, "OUT", tmp_path / "shard_out")
    (tmp_path / "artifacts/this_shard.json").write_text(json.dumps({"index": 2, "stamp": "20261122T0100", "plan_hash": "h"}))
    t, n = publish.build()
    assert n == 1 and t == "shard_02_20261122T0100"
    assert (tmp_path / "shard_out/runs/runs.jsonl").read_text() == '{"run_id": "c"}\n'
    assert not (tmp_path / "shard_out/runs/sample.jsonl").exists()
    assert json.load(open(tmp_path / f"shard_out/artifacts/{t}.json"))["plan_hash"] == "h"


def test_integer_question_ids_work_end_to_end():
    questions = [{"question_id": i, "db": "d1"} for i in range(10)] + [{"question_id": "taxi-01", "db": "nyc_taxi"}]
    plan = shard.make_plan(questions, set(), MODELS, 1, 4, 7)
    by = {q["question_id"]: q for q in questions}
    flat = [i for s in plan["slices"] for i in s]
    assert sorted(map(str, flat)) == sorted(str(q["question_id"]) for q in questions)
    assert ",".join(str(b) for b in plan["slices"][0])
    c = shard.pick(plan, set(), by, 0, "2026-11-20")
    assert c["qids"] == plan["slices"][0]
    done = {j for q in questions[:4] for j in shard.job_ids(q, MODELS, 1)}
    assert shard.slice_missing(plan, 0, done, by)["F1"] >= 0
