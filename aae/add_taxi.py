import json
import pickle

import duckdb

from aae.config import DB_DIR, GOLD_FILE, QUESTIONS_FILE, TAXI_QUESTIONS_FILE
from aae.score import has_top_level_order

gold = pickle.load(open(GOLD_FILE, "rb"))
con = duckdb.connect(str(DB_DIR / "nyc_taxi.duckdb"), read_only=True)
existing = [json.loads(line) for line in open(QUESTIONS_FILE)]
existing = [q for q in existing if q["db"] != "nyc_taxi"]
new, dropped = [], []
for line in open(TAXI_QUESTIONS_FILE):
    q = json.loads(line)
    if not q.get("ambiguous"):
        try:
            rows = con.execute(q["gold_sql"]).fetchall()
            if not rows:
                raise ValueError("empty result")
            ordered = has_top_level_order(q["gold_sql"], "duckdb")
        except Exception as e:
            dropped.append((q["question_id"], f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        gold[q["question_id"]] = {"rows": rows, "ordered": ordered}
    new.append(q)
with open(QUESTIONS_FILE, "w") as f:
    for q in existing + new:
        f.write(json.dumps(q) + "\n")
pickle.dump(gold, open(GOLD_FILE, "wb"))
clear = sum(1 for q in new if not q.get("ambiguous"))
print(f"{clear} clear and {len(new) - clear} ambiguous taxi questions added")
for d in dropped:
    print("  DROPPED", d)
