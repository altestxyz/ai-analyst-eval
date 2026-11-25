import pytest

from aae import llm
from aae.llm import classify_error


def test_per_minute_limit_is_a_rate_limit_not_a_stop():
    assert classify_error("429 quota exceeded for requests per minute") == "rate"


def test_daily_quota_stops_the_run():
    assert classify_error("429 RESOURCE_EXHAUSTED: daily quota exhausted") == "quota"
    assert classify_error("You exceeded your quota for today. limit: 0") == "quota"


def test_expired_credential():
    assert classify_error("401 Unauthorized: token expired") == "auth"


def test_plain_429_is_retried():
    assert classify_error("429 Too Many Requests") == "rate"


def test_other_errors():
    assert classify_error("connection reset by peer") == "other"


def test_missing_proxy_credentials_stop_at_once_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("MODEL_PROXY_URL", raising=False)
    monkeypatch.delenv("MODEL_PROXY_API_KEY", raising=False)
    monkeypatch.setenv("AAE_PROXY_MODE", "openai")
    monkeypatch.setattr(llm, "_client", None)
    cfg = {"provider": "kproxy", "id": "x/y"}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with pytest.raises(llm.ProxyAuthExpired, match="MODEL_PROXY_URL"):
        llm.chat(cfg, msgs, use_disk_cache=False)
