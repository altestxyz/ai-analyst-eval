import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Path("docs").mkdir(exist_ok=True)
CTXS = ["L1", "L2", "L3"]


def boot_ci(v, n=2000):
    rng = np.random.default_rng(0)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


def accuracy():
    df = pd.DataFrame([json.loads(l) for l in open("runs/runs.jsonl")])
    clear = df[df.correct.notna()].copy()
    clear["correct"] = clear.correct.astype(float)
    for name, part in [("bird", clear[clear.db != "nyc_taxi"]), ("taxi_heldout", clear[clear.db == "nyc_taxi"])]:
        if part.empty:
            continue
        per_q = part.groupby(["model", "context", "question_id"]).correct.mean().reset_index()
        models = sorted(per_q.model.unique())
        x, w = np.arange(len(models)), 0.26
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for j, c in enumerate(CTXS):
            means, lo_err, hi_err = [], [], []
            for m in models:
                v = per_q[(per_q.model == m) & (per_q.context == c)].correct.values
                if len(v) == 0:
                    means.append(0)
                    lo_err.append(0)
                    hi_err.append(0)
                    continue
                lo, hi = boot_ci(v)
                means.append(v.mean() * 100)
                lo_err.append((v.mean() - lo) * 100)
                hi_err.append((hi - v.mean()) * 100)
            ax.bar(x + (j - 1) * w, means, w, yerr=[lo_err, hi_err], capsize=3, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("accuracy %")
        ax.set_ylim(0, 100)
        ax.set_title(f"Accuracy by model and context ({name}), 95% bootstrap intervals")
        ax.legend(title="context")
        fig.tight_layout()
        fig.savefig(f"docs/accuracy_{name}.png", dpi=130)
        plt.close(fig)
        print("wrote", f"docs/accuracy_{name}.png")


def failures():
    p = Path("labels/judge.jsonl")
    if not p.exists():
        print("no judge labels yet")
        return
    runs = pd.DataFrame([json.loads(l) for l in open("runs/runs.jsonl")])[["run_id", "context", "model"]]
    lab = pd.DataFrame([json.loads(l) for l in open(p)]).drop_duplicates("run_id")
    d = lab.merge(runs, on="run_id")
    if d.empty:
        return
    t = d.groupby(["label", "context"]).size().unstack("context").reindex(columns=CTXS).fillna(0)
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t.plot.barh(ax=ax)
    ax.set_xlabel("failed runs (labelled by the judge)")
    ax.set_title("Failure causes by context")
    fig.tight_layout()
    fig.savefig("docs/failure_causes.png", dpi=130)
    plt.close(fig)
    print(t.astype(int))
    print("wrote docs/failure_causes.png")


if __name__ == "__main__":
    accuracy()
    failures()
