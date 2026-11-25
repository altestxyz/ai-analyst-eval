import collections
import json
import pickle
import sqlite3
import time

import duckdb

from aae.config import ART_DIR, BIRD_DIR, DB_DIR, DBS, GOLD_FILE, KEPT_FILE
from aae.score import has_top_level_order, same_result


def run_sqlite(db_id, sql, timeout=30):
    path = BIRD_DIR / "dev_databases" / db_id / f"{db_id}.sqlite"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    start = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - start > timeout else 0, 100_000)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def run_duck(db_id, sql):
    con = duckdb.connect(str(DB_DIR / f"{db_id}.duckdb"), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def main():
    ART_DIR.mkdir(exist_ok=True)
    sqlite_qs = json.load(open(BIRD_DIR / "mini_dev_sqlite.json"))
    pg = {q["question_id"]: q for q in json.load(open(BIRD_DIR / "mini_dev_postgresql.json"))}
    kept, gold, dropped = [], {}, collections.Counter()
    for q in sqlite_qs:
        if q["db_id"] not in DBS:
            continue
        qid = q["question_id"]
        if qid not in pg:
            dropped["no postgres version"] += 1
            continue
        try:
            want = run_sqlite(q["db_id"], q["SQL"])
            got = run_duck(q["db_id"], pg[qid]["SQL"])
        except Exception as e:
            dropped[f"error: {type(e).__name__}"] += 1
            continue
        if not want:
            dropped["empty gold result"] += 1
            continue
        ordered = has_top_level_order(q["SQL"], "sqlite")
        if not same_result(got, want, ordered):
            dropped["results differ"] += 1
            continue
        kept.append(
            {
                "question_id": qid,
                "db": q["db_id"],
                "question": q["question"],
                "difficulty": q["difficulty"],
                "gold_sql": pg[qid]["SQL"],
            }
        )
        gold[qid] = {"rows": got, "ordered": ordered}

    json.dump(kept, open(KEPT_FILE, "w"))
    pickle.dump(gold, open(GOLD_FILE, "wb"))
    print("kept", len(kept), "| dropped", dict(dropped))
    by = collections.Counter((q["db"], q["difficulty"]) for q in kept)
    for k, v in sorted(by.items()):
        print(" ", k, v)


if __name__ == "__main__":
    main()
