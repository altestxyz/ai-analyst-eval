import pytest

from aae import runner
from aae.guard import GuardError

Q = {"question_id": "q1", "db": "d", "question": "How many?", "difficulty": "simple"}
GOLD = {"q1": {"rows": [(5,)], "ordered": False}}


def sql(s):
    return f"```sql\n{s}\n```"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    for lvl in ("L1", "L2", "L3"):
        (tmp_path / "d" / f"{lvl}.txt").write_text("schema")
    monkeypatch.setattr(runner, "CTX_DIR", tmp_path)
    monkeypatch.setitem(runner.MODELS, "T", {"provider": "fake", "id": "fake", "price_in": 1.0, "price_out": 2.0})

    def install(replies, results):
        replies, results = list(replies), list(results)

        def fake_chat(cfg, messages, repeat=0, **kw):
            return {
                "text": replies.pop(0),
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "latency_s": 0.5,
            }

        def fake_run_sql(con, s):
            r = results.pop(0)
            if isinstance(r, Exception):
                raise r
            return {"columns": ["c"], "rows": r, "truncated": False}

        monkeypatch.setattr(runner, "chat", fake_chat)
        monkeypatch.setattr(runner, "run_sql", fake_run_sql)

    return install


def go(opt=None, q=Q):
    return runner.run_one(q, "L1", "T", 0, GOLD, {"d": None}, opt)


def test_clean_run(setup):
    setup([sql("SELECT 5")], [[(5,)]])
    r = go()
    assert r["correct"] is True and r["n_turns"] == 1 and not r["had_error"]
    assert r["cost_usd"] == pytest.approx((100 * 1 + 10 * 2) / 1e6)


def test_error_then_fix(setup):
    setup([sql("SELEC 5"), sql("SELECT 5")], [GuardError("SQL error"), [(5,)]])
    r = go()
    assert r["correct"] is True and r["had_error"] and r["n_turns"] == 2


def test_zero_gets_nudged_once(setup):
    setup([sql("SELECT 0"), sql("SELECT 5")], [[(0,)], [(5,)]])
    r = go()
    assert r["had_suspicious"] and r["correct"] is True and r["n_turns"] == 2


def test_nudge_off_accepts_zero(setup):
    setup([sql("SELECT 0")], [[(0,)]])
    r = go({"nudge": False})
    assert r["correct"] is False and not r["had_suspicious"] and r["n_turns"] == 1


def test_repeating_the_same_query_confirms_it(setup):
    setup([sql("SELECT 0"), sql("SELECT 0")], [[(0,)], [(0,)]])
    q = {**Q, "question_id": "q2"}
    r = runner.run_one(q, "L1", "T", 0, {"q2": {"rows": [(0,)], "ordered": False}}, {"d": None})
    assert r["correct"] is True and r["n_turns"] == 2


def test_ask_on_clear_question_is_a_miss(setup):
    setup(["ASK: which one?"], [])
    r = go()
    assert r["asked"] and r["correct"] is False


def test_ask_on_ambiguous_question_is_not_scored(setup):
    setup(["ASK: trips or revenue?"], [])
    r = go(q={**Q, "ambiguous": True})
    assert r["asked"] and r["correct"] is None


def test_prose_is_a_format_failure(setup):
    setup(["I think the answer is 5."], [])
    r = go()
    assert r["format_fail"] and r["correct"] is False


def test_error_after_a_result_keeps_the_earlier_result(setup):
    setup([sql("SELECT 0"), sql("SELEC bad"), "no idea"], [[(0,)], GuardError("SQL error")])
    r = go()
    assert r["had_error"] and r["had_suspicious"] and r["n_turns"] == 3
    assert r["result_preview"] == [[0]] and r["correct"] is False


def test_verify_ok(setup):
    setup([sql("SELECT 5"), "OK"], [[(5,)]])
    r = go({"verify": True})
    assert r["correct"] is True and r["n_turns"] == 2 and not r["verify_changed"]


def test_verify_revision_fixes_a_wrong_answer(setup):
    setup([sql("SELECT 7"), sql("SELECT 5")], [[(7,)], [(5,)]])
    r = go({"verify": True})
    assert r["correct"] is True and r["verify_changed"] and r["n_turns"] == 2


def test_cost_with_cache():
    cfg = {"price_in": 3.0, "price_out": 15.0, "price_cache_read": 0.3, "price_cache_write": 3.75}
    billed, listed = runner.cost_of(cfg, 100, 1000, 0, 10)
    assert billed == pytest.approx((100 * 3 + 1000 * 0.3 + 10 * 15) / 1e6)
    assert listed == pytest.approx((1100 * 3 + 10 * 15) / 1e6)


def test_append_record_is_one_line_per_record(tmp_path):
    p = tmp_path / "log.jsonl"
    runner.append_record(p, {"run_id": "a", "x": 1})
    runner.append_record(p, {"run_id": "b", "x": 2})
    assert runner.done_ids(p) == {"a", "b"}
