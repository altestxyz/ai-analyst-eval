import json
import sys
from pathlib import Path

import pandas as pd

from aae.config import CTX_DIR
from aae.cost import add_costs


def load(tag):
    p = Path(f"runs/{tag}.jsonl")
    return add_costs(pd.DataFrame([json.loads(l) for l in open(p)])) if p.exists() else pd.DataFrame()


def p95(s):
    return s.quantile(0.95)


def caching():
    cold, warm = load("cache_cold"), load("cache_warm")
    if cold.empty or warm.empty:
        print("No caching runs yet.")
        return
    cold["arm"], warm["arm"] = "cold", "warm"
    d = pd.concat([cold, warm])
    d = d[d.correct.notna()]
    d["prompt_total"] = d.input_tokens + d.cache_read_tokens + d.cache_write_tokens
    g = d.groupby(["model", "context", "arm"]).agg(
        runs=("run_id", "count"),
        acc=("correct", "mean"),
        usd_per_run=("cost_usd", "mean"),
        p50_s=("latency_s", "median"),
        p95_s=("latency_s", p95),
        cached=("cache_read_tokens", "sum"),
        total=("prompt_total", "sum"),
    )
    g["hit_rate"] = g.cached / g.total
    print("\n=== Prompt caching: cold vs warm ===")
    print(g.drop(columns=["cached", "total"]).round(4))
    w = g.usd_per_run.unstack("arm")
    print("\nCost saved by caching, % of cold cost (equivalent list price):")
    print(((1 - w["warm"] / w["cold"]) * 100).round(1))
    l = g.p50_s.unstack("arm")
    print("\nMedian latency saved, % of cold latency:")
    print(((1 - l["warm"] / l["cold"]) * 100).round(1))
    print("\nSanity check: accuracy should match between arms. A gap here is sampling noise, not caching.")


def latency():
    r = load("runs")
    r = r[r.correct.notna()] if not r.empty else r
    if r.empty:
        print("No main-run log yet.")
        return
    g = r.groupby(["model", "context"]).agg(
        prompt_tokens=("max_prompt_tokens", "median"),
        p50_s=("latency_s", "median"),
        p95_s=("latency_s", p95),
        usd_per_run=("cost_usd", "mean"),
    )
    print("\n=== Latency and cost by context (compare within a model, not across vendors) ===")
    print(g.round(3))


def scale():
    s = load("scale")
    if s.empty:
        print("No scale runs yet.")
        return
    s = s[s.correct.notna()]
    s["tables"] = s.db.map(lambda db: sum(1 for l in open(CTX_DIR / db / "L2.txt") if l.startswith("Table ")))
    g = s.groupby(["db", "tables", "model"]).agg(
        runs=("run_id", "count"),
        acc=("correct", "mean"),
        prompt_tokens=("max_prompt_tokens", "median"),
        usd_per_run=("cost_usd", "mean"),
        p95_s=("latency_s", p95),
        truncated=("likely_truncated", "mean"),
    )
    print("\n=== Schema scale: same questions, more tables in the prompt ===")
    print(g.round(3))
    print("\nIf 'truncated' is above 0 for a local model, its accuracy drop is partly the prompt being cut off.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("latency", latency), (("caching"), caching), ("scale", scale)):
        if which in (name, "all"):
            fn()
