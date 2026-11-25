import os

import yaml

from aae.env import load_env

load_env()


def route():
    choice = os.getenv("AAE_ROUTE", "auto")
    if choice == "auto":
        return "kproxy" if os.getenv("MODEL_PROXY_URL") and os.getenv("MODEL_PROXY_API_KEY") else "direct"
    if choice not in ("kproxy", "direct"):
        raise SystemExit("AAE_ROUTE must be 'auto', 'kproxy' or 'direct'.")
    return choice


def resolve(cfg, chosen=None):
    if cfg.get("provider") != "hosted":
        return cfg
    chosen = chosen or route()
    flat = {k: v for k, v in cfg.items() if k not in ("routes", "provider")}
    flat.update(cfg["routes"][chosen])
    flat["provider"] = "kproxy" if chosen == "kproxy" else "api"
    return flat


def load_models(path="models.yaml", chosen=None):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {key: resolve(value, chosen) for key, value in raw.items()}
