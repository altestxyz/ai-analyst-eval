import pandas as pd
import pytest

from aae import cost, llm


def test_costs_are_recomputed_from_tokens_not_the_stored_zero(monkeypatch):
    monkeypatch.setattr(
        cost,
        "MODELS",
        {"X": {"price_in": 2.0, "price_out": 10.0, "price_cache_read": 0.5}, "L": {"provider": "ollama", "price_in": 0, "price_out": 0}},
    )
    row = {"cache_write_tokens": 0, "cost_usd": 0.0, "cost_list_usd": 0.0}
    df = pd.DataFrame(
        [
            {**row, "model": "X", "input_tokens": 1000, "cache_read_tokens": 1000, "output_tokens": 100},
            {**row, "model": "L", "input_tokens": 5000, "cache_read_tokens": 0, "output_tokens": 500},
        ]
    )
    out = cost.add_costs(df)
    assert out.cost_usd[0] == pytest.approx((1000 * 2 + 1000 * 0.5 + 100 * 10) / 1e6)
    assert out.cost_list_usd[0] == pytest.approx((2000 * 2 + 100 * 10) / 1e6)
    assert out.cost_usd[1] == 0 and df.cost_usd[0] == 0


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _resp(text, finish, prompt, out):
    return _Obj(
        choices=[_Obj(message=_Obj(content=text), finish_reason=finish)],
        usage=_Obj(prompt_tokens=prompt, completion_tokens=out, prompt_tokens_details=None),
    )


class _FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = _Obj(completions=_Obj(create=self.create))

    def create(self, **kw):
        self.calls.append(dict(kw))
        return self.replies.pop(0)


MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
SQL = "```sql\nSELECT 1\n```"


def test_reply_cut_off_at_a_low_limit_is_retried_once_at_4096(monkeypatch):
    fake = _FakeClient([_resp("", "length", 100, 1024), _resp(SQL, "stop", 100, 60)])
    monkeypatch.setattr(llm, "_client", fake)
    monkeypatch.setattr(llm, "_quirks", {})
    out = llm._kproxy_openai({"id": "x/y", "max_output_tokens": 1024}, MSGS)
    assert [c["max_tokens"] for c in fake.calls] == [1024, 4096]
    assert out["limit_retry"] and "SELECT 1" in out["text"]
    assert out["output_tokens"] == 1084 and out["input_tokens"] == 200


def test_normal_reply_is_not_retried(monkeypatch):
    fake = _FakeClient([_resp(SQL, "stop", 100, 60)])
    monkeypatch.setattr(llm, "_client", fake)
    monkeypatch.setattr(llm, "_quirks", {})
    out = llm._kproxy_openai({"id": "x/y", "max_output_tokens": 1024}, MSGS)
    assert len(fake.calls) == 1 and not out["limit_retry"]


def test_no_retry_when_already_at_the_usual_limit(monkeypatch):
    fake = _FakeClient([_resp("", "length", 100, 4096)])
    monkeypatch.setattr(llm, "_client", fake)
    monkeypatch.setattr(llm, "_quirks", {})
    out = llm._kproxy_openai({"id": "x/y"}, MSGS)
    assert len(fake.calls) == 1 and not out["limit_retry"]
