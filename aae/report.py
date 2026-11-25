import json
import os

import numpy as np
import pandas as pd

from aae.cost import add_costs, price_note
from aae.score import paired_diff_ci, verdict

THRESHOLD = 5.0
df = add_costs(pd.DataFrame([json.loads(line) for line in open("runs/runs.jsonl")]))
df["heldout"] = df["db"] == "nyc_taxi"
clear = df[df["correct"].notna()].copy()
clear["correct"] = clear["correct"].astype(float)
STEPS = [("L1", "L2"), ("L2", "L3")]
BIGM = os.environ.get("AAE_BIG", "F1")


def line(label, a, b, n):
    if n < 2:
        print(f"{label:34} n={n}: too few questions yet")
        return
    d, lo, hi = paired_diff_ci(a, b)
    note = ""
    if hi < 0:
        note += "   [whole interval below zero: accuracy went DOWN]"
    if float(np.mean(a)) >= 0.95:
        note += "   [ceiling: the baseline is already 95% or more, so a 'not supported' here says little]"
    print(f"{label:34} {d:+5.1f} pts  [{lo:+5.1f}, {hi:+5.1f}]  n={n:3}  -> {verdict(d, lo, hi, THRESHOLD)}{note}")


print(f"{len(df)} runs in the log. Models: {sorted(df.model.unique())}")
for name, part in [("BIRD", clear[~clear.heldout]), ("NYC taxi held-out", clear[clear.heldout])]:
    if part.empty:
        continue
    per_q = part.groupby(["model", "context", "question_id"]).correct.mean().reset_index()
    print(f"\n=== {name}: accuracy % by model and context ===")
    print((per_q.groupby(["model", "context"]).correct.mean().unstack("context") * 100).round(1))
    print(f"\nquestions finished per model and context ({name}). The pooled comparison below uses only the questions every setup finished:")
    print(per_q.groupby(["model", "context"]).question_id.nunique().unstack("context"))

    print(f"\n=== {name}: primary comparison, average over models (paired, 95% interval) ===")
    n_cells = per_q.groupby(["model", "context"]).ngroups
    cover = per_q.groupby("question_id").size()
    full = per_q[per_q.question_id.isin(cover[cover == n_cells].index)]
    pooled = full.groupby(["context", "question_id"]).correct.mean().unstack("context").dropna()
    for a, b in STEPS:
        if a in pooled and b in pooled:
            line(f"{b} minus {a}, all models", pooled[a].values, pooled[b].values, len(pooled))
    if len(pooled) == 0:
        print("(no question has every model and context yet)")
    _author = os.environ.get("AAE_SEMANTIC_AUTHOR", "")
    if _author in set(per_q.model) and per_q.model.nunique() > 2:
        oth = per_q[per_q.model != _author]
        cov_o = oth.groupby("question_id").size()
        full_o = oth[oth.question_id.isin(cov_o[cov_o == oth.groupby(["model", "context"]).ngroups].index)]
        pooled_o = full_o.groupby(["context", "question_id"]).correct.mean().unstack("context").dropna()
        if "L2" in pooled_o and "L3" in pooled_o:
            line(f"L3 minus L2, without {_author}", pooled_o["L2"].values, pooled_o["L3"].values, len(pooled_o))
            print(f"  ({_author} drafted the L3 semantic models, so this row leaves out the model that wrote them)")

    print(f"\n=== {name}: per model ===")
    wide = {
        m: per_q[per_q.model == m].pivot(index="question_id", columns="context", values="correct").dropna()
        for m in sorted(per_q.model.unique())
    }
    for m, p in wide.items():
        for a, b in STEPS:
            if a in p and b in p:
                line(f"{m}  {b} minus {a}", p[a].values, p[b].values, len(p))

    if {"M1", BIGM} <= set(wide) and all(c in wide[m] for m in ("M1", BIGM) for c in ("L2", "L3")):
        print(f"\n=== {name}: does the semantic model help M1 more than {BIGM}? (L3 minus L2, M1 gain minus {BIGM} gain) ===")
        both = wide["M1"][["L2", "L3"]].join(wide[BIGM][["L2", "L3"]], lsuffix="_1", rsuffix="_4", how="inner")
        g1, g4 = both.L3_1 - both.L2_1, both.L3_4 - both.L2_4
        d, lo, hi = paired_diff_ci(g4.values, g1.values)
        print(f"{d:+5.1f} pts  [{lo:+5.1f}, {hi:+5.1f}]  n={len(both)}")
        print(f"\n=== {name}: {BIGM} minus M1 at each context, and whether the gap shrinks ===")
        gaps = {}
        for c in ("L1", "L2", "L3"):
            p = per_q[per_q.context == c].pivot(index="question_id", columns="model", values="correct").dropna()
            if "M1" not in p or BIGM not in p or len(p) == 0:
                continue
            gaps[c] = p[BIGM] - p["M1"]
            d, lo, hi = paired_diff_ci(p["M1"].values, p[BIGM].values)
            print(f"at {c}: {d:+5.1f} pts  [{lo:+5.1f}, {hi:+5.1f}]  n={len(p)}")
        if "L1" in gaps and "L3" in gaps:
            g = pd.concat([gaps["L1"], gaps["L3"]], axis=1, keys=["L1", "L3"]).dropna()
            d, lo, hi = paired_diff_ci(g["L1"].values, g["L3"].values)
            print(f"gap at L3 minus gap at L1: {d:+5.1f} pts  [{lo:+5.1f}, {hi:+5.1f}]  (negative = gap shrinks)")

print("\n=== Cost per correct answer (equivalent list-price USD, from logged tokens). Billed estimate, then list price with no caching ===")
print(price_note(clear))
g = clear.groupby(["model", "context"]).agg(billed=("cost_usd", "sum"), listed=("cost_list_usd", "sum"), right=("correct", "sum"))
from aae.cost import MODELS as _PRICED

_paid = [m for m in g.index.get_level_values(0).unique() if _PRICED.get(m, {}).get("provider", "ollama") != "ollama"]
if _paid and g.loc[_paid].billed.sum() == 0:
    print(
        "WARNING: every frontier cost reads 0. The prices in FRONTIER are 0 or missing, or the runs logged no token counts (sdk proxy mode)."
    )
print((g.billed / g.right).unstack("context").round(4))
print((g.listed / g.right).unstack("context").round(4))

print("\nEquivalent list-price spend so far, US$ by model (the free quota's own accounting is in dollars too, so this sizes a slice):")
print(g.groupby(level=0).billed.sum().round(2).to_string())

print("\n=== Product maturity ===")
c = clear.copy()
for col in ("had_error", "had_suspicious", "asked", "format_fail"):
    c[col] = c[col].fillna(False).astype(bool)
c["silent_wrong"] = (c.correct == 0) & ~c.had_error & ~c.had_suspicious & ~c.asked & ~c.format_fail
print("\nSilent wrong rate: wrong answers where nothing flagged a problem, % of all runs")
print((c.groupby(["model", "context"]).silent_wrong.mean().unstack("context") * 100).round(1))
print("\nRecovery after an error: share of runs that hit an error and still ended right")
print(c[c.had_error].groupby("model").correct.agg(["mean", "count"]).round(2))
print("\nRecovery after a 'looks wrong' nudge: share that ended right")
print(c[c.had_suspicious].groupby("model").correct.agg(["mean", "count"]).round(2))
_nud = c[c.had_suspicious]
if len(_nud):
    print(
        f"\nAll models together: {int(_nud.correct.sum())} of {len(_nud)} nudged runs ended right ({_nud.correct.mean() * 100:.0f}%). "
        f"Mean turns per run: {_nud.n_turns.mean():.1f} when nudged, {c[~c.had_suspicious].n_turns.mean():.1f} when not."
    )
from aae.diagnose import load_shapes

if "shape" not in c:
    c["shape"] = None
c["shape"] = c["shape"].where(c["shape"].notna(), c.run_id.map(load_shapes()))
_wrong = c[(c.correct == 0) & c["shape"].notna()]
if len(_wrong):
    print(
        f"\nHow the wrong answers differ from the gold answer ({len(_wrong)} of {int((c.correct == 0).sum())} wrong runs labelled; "
        "older runs need `python -m aae.diagnose backfill`). partial_overlap points at a filter, disjoint_values at a metric or a join:"
    )
    print(pd.crosstab([_wrong.model, _wrong.context], _wrong["shape"]))
print("\nMedian output tokens per run, by model and context:")
print(df[df.correct.notna()].groupby(["model", "context"]).output_tokens.median().unstack("context").round(0))
if "reasoning_tokens" in df and df.reasoning_tokens.notna().any():
    print("\nShare of runs with reasoning tokens above 0, % (0 means no thinking budget was used, or the proxy does not report it):")
    print((df[df.reasoning_tokens.notna()].assign(r=lambda x: x.reasoning_tokens > 0).groupby("model").r.mean() * 100).round(0).to_string())
else:
    print(
        "\nNote: these runs did not log reasoning tokens (made before v11), so a difference in the models' default reasoning effort can't be seen here."
    )
print("\nFormat failures (no SQL block, no ASK:), % of runs")
print((c.groupby("model").format_fail.mean() * 100).round(1))
print("\nUnneeded questions: share of clear questions where the agent asked, %")
print((c.groupby(["model", "context"]).asked.mean().unstack("context") * 100).round(1))
if "limit_retries" in df and df.limit_retries.fillna(0).sum():
    print("\nReplies that hit the lower max_output_tokens and were retried at 4,096 (runs with at least one):")
    print(df[df.limit_retries.fillna(0) > 0].groupby("model").size().to_string())
trunc = df[df.likely_truncated.eq(True)] if "likely_truncated" in df else df.iloc[0:0]
if len(trunc):
    print(f"\nWARNING: {len(trunc)} local runs likely had their prompt cut off (context window too small):")
    print(trunc.groupby(["model", "context"]).size())
amb = df[df["correct"].isna()]
if not amb.empty:
    print("\nAmbiguous set: share of runs where the agent asked, %")
    print((amb.groupby(["model", "context"]).asked.mean() * 100).round(0).unstack("context"))

author = os.environ.get("AAE_SEMANTIC_AUTHOR", "")
if author:
    print(
        f"\nNote: the L3 semantic models were drafted by {author}. Read that model's L3 gain with that in mind: it may favour its own phrasing."
    )
