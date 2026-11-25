import json
import math
import os
import sys

import pandas as pd

from aae.config import QUESTIONS_FILE, RUNS_FILE
from aae.models import load_models

MODELS = load_models()
BIG = os.getenv("AAE_BIG", "F1")
ADDONS = 600


def main(limits):
    n_q = sum(1 for _ in open(QUESTIONS_FILE))
    df = pd.DataFrame([json.loads(l) for l in open(RUNS_FILE)])
    print(f"{n_q} questions x 3 contexts. One API request per model turn.")
    print(f"{'model':6} {'req/run':>8} {'repeats':>8} {'requests':>9} {'hours at rpm':>13} {'days at daily limit':>20}")
    for m in [k for k, v in MODELS.items() if v["provider"] != "ollama"]:
        d = df[df.model == m] if not df.empty else df
        if d.empty:
            print(m, "has no pilot runs yet")
            continue
        per_run = d.n_turns.mean()
        per_day = limits.get(m)
        for rep in (1, 2):
            reqs = per_run * n_q * 3 * rep + (ADDONS if m == BIG else 0)
            hours = reqs / MODELS[m].get("rpm", 10) / 60
            days = str(math.ceil(reqs / per_day)) if per_day else "?"
            print(f"{m:6} {per_run:8.2f} {rep:8d} {reqs:9.0f} {hours:13.1f} {days:>20}")
    print(f"\n{BIG} includes about {ADDONS} requests for the add-ons.")
    print("'?' means you didn't pass that model's daily limit. Look it up in your provider's docs, then rerun with F1=<n> ...")
    print("If the days are too many: use 1 repeat, then run fewer questions, then drop a frontier model. Do it BEFORE the freeze.")


if __name__ == "__main__":
    main({a.split("=")[0]: int(a.split("=")[1]) for a in sys.argv[1:] if "=" in a})
