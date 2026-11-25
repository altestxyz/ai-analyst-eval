import csv
import json
from pathlib import Path

import pandas as pd

from aae.config import QUESTIONS_FILE, RUNS_FILE

Path("labels").mkdir(exist_ok=True)
qs = {json.loads(l)["question_id"]: json.loads(l) for l in open(QUESTIONS_FILE)}
r = pd.DataFrame([json.loads(l) for l in open(RUNS_FILE)])
r = r[r.correct.notna()].copy()
r["correct"] = r.correct.astype(float)
g = r.groupby("question_id").agg(runs=("correct", "size"), right=("correct", "sum"))
bad = g[(g.right == 0) & (g.runs >= 6)].index
with open("labels/audit_all_wrong.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["question_id", "db", "question", "gold_sql", "example_final_sql", "example_model", "your_decision"])
    for qid in bad:
        ex = r[r.question_id == qid].iloc[0]
        q = qs.get(qid, {})
        w.writerow([qid, q.get("db"), q.get("question"), q.get("gold_sql"), ex.final_sql, ex.model, ""])
print(f"{len(bad)} questions had every run wrong (of {len(g)}). Review labels/audit_all_wrong.csv by hand:")
print("decide for each whether the answer key is wrong, the question is unclear, or the models really fail.")
