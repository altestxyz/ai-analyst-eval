import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from aae.config import GOLD_FILE, QUESTIONS_FILE, RUNS_FILE


def stories(model="F1", low="L1", high="L3"):
    r = pd.DataFrame([json.loads(l) for l in open(RUNS_FILE)])
    r = r[(r.model == model) & r.correct.notna()]
    p = r.groupby(["question_id", "context"]).correct.mean().unstack("context")
    if low not in p or high not in p:
        return r, []
    return r, p[(p[low] == 0) & (p[high] == 1)].index.tolist()


def cards(model="F1", limit=5):
    runs, ids = stories(model)
    qs = {json.loads(l)["question_id"]: json.loads(l) for l in open(QUESTIONS_FILE)}
    out = [f"# Questions that flipped from wrong to right ({model})\n"]
    for qid in ids[:limit]:
        q = qs[qid]
        out += [
            f"## {q['question']}\n",
            f"Database: {q['db']}, difficulty: {q.get('difficulty')}\n",
            f"Gold SQL:\n```sql\n{q.get('gold_sql')}\n```\n",
        ]
        for lvl in ("L1", "L2", "L3"):
            row = runs[(runs.question_id == qid) & (runs.context == lvl) & (runs.repeat == 0)].iloc[0]
            out.append(f"**{lvl}**, correct={row.correct}:\n```sql\n{row.final_sql}\n```\n")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/flip_cards.md").write_text("\n".join(out))
    print(f"{len(ids)} questions flipped for {model}; wrote {min(limit, len(ids))} to docs/flip_cards.md")


def live(qid, mkey):
    from aae.config import DB_DIR
    from aae.guard import open_connection
    from aae.runner import LEVELS, run_one

    q = next(json.loads(l) for l in open(QUESTIONS_FILE) if json.loads(l)["question_id"] == qid)
    gold = pickle.load(open(GOLD_FILE, "rb"))
    conns = {q["db"]: open_connection(DB_DIR / f"{q['db']}.duckdb")}
    print(f"\nQuestion: {q['question']}\n")
    for lvl in LEVELS:
        rec = run_one(q, lvl, mkey, 0, gold, conns, {"tag": "demo"})
        print(f"--- {lvl}  correct={rec['correct']}  turns={rec['n_turns']}  {rec['latency_s']}s")
        print(rec["final_sql"] or "(no SQL)", "\n")


if __name__ == "__main__":
    if sys.argv[1] == "cards":
        cards(*(sys.argv[2:3] or ["F1"]))
    else:
        live(sys.argv[2], sys.argv[3])
