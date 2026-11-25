import sys
import types

import pytest

from aae import llm


class FakeCompletions:
    def __init__(self, log):
        self.log = log

    def create(self, **kw):
        self.log.append(kw)
        usage = types.SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=9,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=20),
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=0),
        )
        choice = types.SimpleNamespace(finish_reason="stop", message=types.SimpleNamespace(content="SELECT 1"))
        return types.SimpleNamespace(usage=usage, choices=[choice])


@pytest.fixture
def fake_openai(monkeypatch):
    made, calls = [], []

    class FakeClient:
        def __init__(self, base_url, api_key, timeout):
            made.append((base_url, api_key))
            self.chat = types.SimpleNamespace(completions=FakeCompletions(calls))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))
    monkeypatch.setattr(llm, "_api_clients", {})
    monkeypatch.setattr(llm, "_quirks", {})
    return made, calls


CFG = {"provider": "api", "id": "vendor-model", "base_url": "https://h.example/v1/", "api_key_env": "VENDOR_KEY", "max_output_tokens": 1024}
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]


def test_a_missing_vendor_key_stops_the_run_with_the_variable_name(monkeypatch, fake_openai):
    monkeypatch.delenv("VENDOR_KEY", raising=False)
    with pytest.raises(llm.ProxyAuthExpired, match="VENDOR_KEY"):
        llm._api(CFG, MSGS)


def test_the_vendor_endpoint_is_called_with_the_configured_model_and_limits(monkeypatch, fake_openai):
    made, calls = fake_openai
    monkeypatch.setenv("VENDOR_KEY", "test-key")
    out = llm._api(CFG, MSGS)
    assert made == [("https://h.example/v1/", "test-key")]
    assert calls[0]["model"] == "vendor-model" and calls[0]["max_tokens"] == 1024 and calls[0]["temperature"] == 0
    assert out["text"] == "SELECT 1" and out["input_tokens"] == 100 and out["cache_read_tokens"] == 20


def test_one_client_is_reused_per_endpoint_and_key(monkeypatch, fake_openai):
    made, _ = fake_openai
    monkeypatch.setenv("VENDOR_KEY", "test-key")
    llm._api(CFG, MSGS)
    llm._api(CFG, MSGS)
    assert len(made) == 1


def test_the_provider_table_routes_both_hosted_paths():
    assert llm.PROVIDERS["api"] is llm._api
    assert llm.PROVIDERS["kproxy"] is llm._kproxy
    assert set(llm.PROVIDERS) == {"ollama", "kproxy", "api"}
