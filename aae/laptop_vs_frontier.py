import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aae.cost import add_costs
from aae.models import load_models
from aae.score import noninferior_verdict, paired_diff_ci

MARGIN = float(os.environ.get("AAE_LAPTOP_MARGIN", "5.0"))
MODELS = load_models()
LABEL = {k: v.get("label", k) for k, v in MODELS.items()}
CTXS = ["L1", "L2", "L3"]
Path("docs").mkdir(exist_ok=True)

df = add_costs(pd.DataFrame([json.loads(line) for line in open("runs/runs.jsonl")]))
df = df[df.correct.notna()].copy()
df["correct"] = df.correct.astype(float)
have = set(df.model.unique())
LAPTOP = [k for k, v in MODELS.items() if v["provider"] == "ollama" and k in have and k not in ("M1L", "JUDGE")]
FRONT = [k for k, v in MODELS.items() if v["provider"] != "ollama" and k in have]
if not LAPTOP or not FRONT:
    print("Need runs from at least one laptop-class and one frontier model. Have:", sorted(have))
    sys.exit(0)


def show(pq, label, frontier_col, laptop_col):
    if frontier_col not in pq.columns or laptop_col not in pq.columns:
        return
    x = pq[[frontier_col, laptop_col]].dropna()
    if len(x) < 2:
        print(f"{label:62} too few questions yet")
        return
    d, lo, hi = paired_diff_ci(x[frontier_col].values, x[laptop_col].values)
    tag = " (ahead)" if lo > 0 else ""
    print(f"{label:62} {d:+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]  n={len(x):3}  {noninferior_verdict(d, lo, hi, MARGIN)}{tag}")


def gap_shrink(pq, lap, fr):
    need = [(fr, "L1"), (lap, "L1"), (fr, "L3"), (lap, "L3")]
    if not all(n in pq.columns for n in need):
        return
    x = pq[need].dropna()
    if len(x) < 2:
        return
    g1 = (x[need[0]] - x[need[1]]).values
    g3 = (x[need[2]] - x[need[3]]).values
    d, lo, hi = paired_diff_ci(g1, g3)
    print(
        f"{LABEL[fr]} minus {LABEL[lap]}: gap at L3 minus gap at L1 {d:+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]  n={len(x)}"
        "   (negative = context shrinks the gap)"
    )


def table(part):
    rows = []
    for m in LAPTOP + FRONT:
        r = part[(part.model == m) & (part.context == "L3")]
        if r.empty:
            continue
        right = r.correct.sum()
        rows.append(
            {
                "model": LABEL[m],
                "kind": "laptop-class" if m in LAPTOP else "frontier",
                "acc_L3_%": round(r.correct.mean() * 100, 1),
                "list_$_per_correct": round(r.cost_list_usd.sum() / right, 4) if right else None,
                "p50_latency_s": round(r.latency_s.median(), 1),
                "p95_latency_s": round(r.latency_s.quantile(0.95), 1),
                "data_leaves_your_machine": "no" if m in LAPTOP else "yes",
            }
        )
    return pd.DataFrame(rows).set_index("model")


def chart(tag, name, pq):
    order = [m for m in LAPTOP + FRONT]
    means = pq.mean() * 100
    x, w = np.arange(len(order)), 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for j, c in enumerate(CTXS):
        ax.bar(x + (j - 1) * w, [means.get((m, c), np.nan) for m in order], w, label=c)
    l1 = [means.get((f, "L1"), np.nan) for f in FRONT]
    if not np.all(np.isnan(l1)):
        top = np.nanmax(l1)
        ax.axhline(top, ls="--", color="k", lw=1)
        ax.axhspan(top - MARGIN, top, alpha=0.12, color="k")
        ax.text(
            len(order) - 0.55, min(top + 1.5, 97), "best frontier model, bare schema (L1), and the margin below it", ha="right", fontsize=8
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LABEL[m]}\n({'laptop-class' if m in LAPTOP else 'frontier'})" for m in order], fontsize=8)
    ax.set_ylabel("accuracy %")
    ax.set_ylim(0, 100)
    ax.set_title(f"Can a laptop-class model with context catch a bare frontier model? ({name})")
    ax.legend(title="context", loc="lower right")
    fig.tight_layout()
    fig.savefig(f"docs/laptop_vs_frontier_{tag}.png", dpi=130)
    plt.close(fig)
    print("wrote", f"docs/laptop_vs_frontier_{tag}.png")


for name, tag, part in [("BIRD", "bird", df[df.db != "nyc_taxi"]), ("NYC taxi held-out", "taxi", df[df.db == "nyc_taxi"])]:
    if part.empty:
        continue
    pq = part.groupby(["model", "context", "question_id"]).correct.mean().unstack(["model", "context"])
    print(
        f"\n{'=' * 100}\n{name}: laptop-class minus frontier, accuracy points [95% paired interval], verdict against a {MARGIN:.0f}-point margin"
    )

    print("\n--- Headline: laptop-class + semantic model (L3) against the frontier model with the bare schema (L1)")
    for lap in LAPTOP:
        for fr in FRONT:
            show(pq, f"{LABEL[lap]} @L3  vs  {LABEL[fr]} @L1", (fr, "L1"), (lap, "L3"))
    print("\n--- Descriptions only (L2) against the bare-schema frontier model (L1)")
    for lap in LAPTOP:
        for fr in FRONT:
            show(pq, f"{LABEL[lap]} @L2  vs  {LABEL[fr]} @L1", (fr, "L1"), (lap, "L2"))
    print("\n--- Same context on both sides")
    for c in ("L1", "L3"):
        for lap in LAPTOP:
            for fr in FRONT:
                show(pq, f"{LABEL[lap]}  vs  {LABEL[fr]}, both @{c}", (fr, c), (lap, c))
    print("\n--- Does context close the gap?")
    for lap in LAPTOP:
        for fr in FRONT:
            gap_shrink(pq, lap, fr)

    print(f"\n--- {name} at L3: accuracy, equivalent cost, latency, privacy")
    print(table(part).to_string())
    chart(tag, name, pq)

print("\nCaveats to put next to these numbers:")
print("- Laptop-class latency depends on the hardware that ran the models. Only accuracy transfers to other machines.")
print("- Equivalent cost = every run's logged tokens x the prices in FRONTIER now. List prices change: give the date.")
print("- Frontier models ran through hosted endpoints at temperature 0 with no vendor-specific tuning.")
print("- With one repeat per frontier model and about 150 questions, call 'inconclusive' what is inconclusive.")
