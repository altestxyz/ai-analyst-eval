import hashlib
import json
import os
import time
from pathlib import Path

import httpx

CACHE = Path("runs/cache")
CACHE.mkdir(parents=True, exist_ok=True)
_last_call = {}
_client = None
_api_clients = {}
_quirks = {}
RETRY_OUTPUT_TOKENS = 4096


class QuotaExhausted(Exception):
    pass


class ProxyAuthExpired(QuotaExhausted):
    pass


def classify_error(msg):
    low = str(msg).lower()
    per_minute = any(k in low for k in ("per minute", "per_minute", "perminute", "rpm"))
    if not per_minute and any(k in low for k in ("quota", "per day", "per_day", "daily", "limit: 0")):
        return "quota"
    if any(k in low for k in ("401", "unauthorized", "invalid api key", "invalid token", "expired")):
        return "auth"
    if any(k in low for k in ("429", "rate limit", "too many requests", "resource_exhausted")):
        return "rate"
    return "other"


def _pace(cfg):
    rpm = cfg.get("rpm")
    if not rpm:
        return
    last = _last_call.get(cfg["id"])
    if last is not None:
        wait = last + 60.0 / rpm - time.monotonic()
        if wait > 0:
            time.sleep(wait)
    _last_call[cfg["id"]] = time.monotonic()


def _ollama(cfg, messages, cache=False):
    host = os.getenv("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
    r = httpx.post(
        f"{host}/api/chat",
        timeout=900,
        json={
            "model": cfg["id"],
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": cfg.get("num_ctx", 16384)},
        },
    )
    r.raise_for_status()
    d = r.json()
    return {
        "text": d["message"]["content"],
        "input_tokens": d.get("prompt_eval_count", 0),
        "output_tokens": d.get("eval_count", 0),
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


def _usage(u):
    prompt = getattr(u, "prompt_tokens", 0) or 0
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    out = getattr(u, "completion_tokens", 0) or 0
    return prompt, cached, out


def _reasoning(u):
    d = getattr(u, "completion_tokens_details", None)
    return (getattr(d, "reasoning_tokens", 0) or 0) if d else 0


def _kproxy_openai(cfg, messages, cache=False):
    global _client

    if _client is None:
        url, key = os.getenv("MODEL_PROXY_URL"), os.getenv("MODEL_PROXY_API_KEY")
        if not (url and key):
            raise ProxyAuthExpired(
                "MODEL_PROXY_URL / MODEL_PROXY_API_KEY are not set in this process. Run inside a Kaggle notebook "
                "started from the benchmark-task page, or set AAE_ROUTE=direct and export your vendor API keys."
            )
        from openai import OpenAI

        _client = OpenAI(base_url=url, api_key=key, timeout=300)
    return _complete(_client, cfg, messages)


def _api(cfg, messages, cache=False):
    env = cfg.get("api_key_env", "")
    key = os.getenv(env) if env else None
    base = cfg.get("base_url")
    if not (base and key):
        raise ProxyAuthExpired(f"{env or 'api_key_env'} is not set in this process. Export it or add it to .env, then rerun.")
    slot = f"{base}|{env}"
    if slot not in _api_clients:
        from openai import OpenAI

        _api_clients[slot] = OpenAI(base_url=base, api_key=key, timeout=300)
    return _complete(_api_clients[slot], cfg, messages)


def _complete(client, cfg, messages):
    q = _quirks.setdefault(cfg["id"], set())
    for _ in range(3):
        kw = {"model": cfg["id"], "messages": messages}
        kw["max_completion_tokens" if "max_completion_tokens" in q else "max_tokens"] = cfg.get("max_output_tokens", 4096)
        temp = cfg.get("temperature", 0)
        if temp is not None and "no_temperature" not in q:
            kw["temperature"] = temp
        if cfg.get("reasoning"):
            kw["reasoning_effort"] = cfg["reasoning"]
        try:
            r = client.chat.completions.create(**kw)
            break
        except Exception as e:
            s = str(e)
            if "max_completion_tokens" in s and "max_completion_tokens" not in q:
                q.add("max_completion_tokens")
            elif "temperature" in s and "no_temperature" not in q:
                q.add("no_temperature")
            else:
                raise
    else:
        raise RuntimeError("the hosted endpoint rejected the request settings three times")
    prompt, cached, out = _usage(r.usage)
    reas = _reasoning(r.usage)
    retried = False
    limit_key = "max_completion_tokens" if "max_completion_tokens" in kw else "max_tokens"
    if getattr(r.choices[0], "finish_reason", None) == "length" and kw[limit_key] < RETRY_OUTPUT_TOKENS:
        kw[limit_key] = RETRY_OUTPUT_TOKENS
        r = client.chat.completions.create(**kw)
        p2, c2, o2 = _usage(r.usage)
        prompt, cached, out, retried = prompt + p2, cached + c2, out + o2, True
        reas += _reasoning(r.usage)
    text = r.choices[0].message.content or ""
    return {
        "text": text,
        "input_tokens": prompt - cached,
        "output_tokens": out,
        "cache_read_tokens": cached,
        "cache_write_tokens": 0,
        "limit_retry": retried,
        "reasoning_tokens": reas,
    }


def _kproxy_sdk(cfg, messages, cache=False):
    import kaggle_benchmarks as kbench

    llm = kbench.llms[cfg["id"]]
    system, turns = messages[0]["content"], messages[1:]
    if len(turns) == 1:
        prompt = turns[0]["content"]
    else:
        earlier = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in turns[:-1])
        prompt = f"Conversation so far:\n{earlier}\n\nNew message from the user:\n{turns[-1]['content']}"
    with kbench.chats.new("aae", system_instructions=system):
        text = llm.prompt(prompt, temperature=cfg.get("temperature", 0) or 0)
    return {"text": str(text), "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}


def _kproxy(cfg, messages, cache=False):
    fn = _kproxy_sdk if os.getenv("AAE_PROXY_MODE", "openai") == "sdk" else _kproxy_openai
    return fn(cfg, messages, cache)


PROVIDERS = {"ollama": _ollama, "kproxy": _kproxy, "api": _api}


def chat(cfg, messages, repeat=0, cache=False, nonce="", use_disk_cache=True):
    if nonce:
        messages = [{"role": "system", "content": f"[run {nonce}]\n" + messages[0]["content"]}] + messages[1:]
    parts = [cfg["id"], messages, repeat, cache] + ([cfg["reasoning"]] if cfg.get("reasoning") else [])
    key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if use_disk_cache and f.exists():
        return json.loads(f.read_text())
    for attempt in range(6):
        try:
            _pace(cfg)
            t0 = time.time()
            out = PROVIDERS[cfg["provider"]](cfg, messages, cache)
            out["latency_s"] = round(time.time() - t0, 3)
            break
        except QuotaExhausted:
            raise
        except Exception as e:
            kind = classify_error(e)
            if kind == "quota":
                raise QuotaExhausted(str(e)[:300])
            if kind == "auth":
                raise ProxyAuthExpired(str(e)[:300])
            if attempt == (5 if kind == "rate" else 2):
                raise
            time.sleep(20 * (attempt + 1) if kind == "rate" else 3 * 2**attempt)
    f.write_text(json.dumps(out))
    return out
