import json

import pandas as pd

from aae.cost import add_costs

ARMS = {"base (nudge on)": "maturity_base", "nudge off": "maturity_nonudge", "verify on": "maturity_verify"}


def load(tag):
    d = add_costs(pd.DataFrame([json.loads(l) for l in open(f"runs/{tag}.jsonl")]))
    d = d[d.correct.notna()].copy()
    d["correct"] = d.correct.astype(int)
    return d


frames = {arm: load(tag) for arm, tag in ARMS.items()}
base = frames["base (nudge on)"].set_index(["question_id", "model"]).correct
print(f"{'arm':18} {'model':6} {'acc %':>6} {'wrong->right':>13} {'right->wrong':>13} {'turns':>6} {'$ per run':>10}")
for arm, d in frames.items():
    x = d.set_index(["question_id", "model"]).correct
    joined = pd.concat([base.rename("base"), x.rename("arm")], axis=1, join="inner")
    for m, g in d.groupby("model"):
        j = joined.xs(m, level="model")
        fixed, broke = int(((j.arm - j.base) == 1).sum()), int(((j.arm - j.base) == -1).sum())
        print(f"{arm:18} {m:6} {g.correct.mean() * 100:6.1f} {fixed:13} {broke:13} {g.n_turns.mean():6.2f} {g.cost_usd.mean():10.4f}")
print("\nRead it as: did the extra rule fix more answers than it broke, and what did the extra turns cost?")
print("With about 50 questions per arm, a difference of a few answers is noise. Report the counts.")
