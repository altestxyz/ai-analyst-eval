import json
import pickle
import sys

import duckdb

from aae.build_db import build
from aae.config import BIRD_DIR, CTX_DIR, DB_DIR, DBS, GOLD_FILE, QUESTIONS_FILE, SCALE_QUESTIONS_FILE
from aae.contexts import example_values, keys, load_descriptions


def all_bird():
    return sorted(p.name for p in (BIRD_DIR / "dev_databases").iterdir() if p.is_dir())


def make_db(name, target, extras):
    out = DB_DIR / f"{name}.duckdb"
    out.unlink(missing_ok=True)
    con = duckdb.connect(str(out))
    for db, prefix in [(target, "")] + [(d, f"{d}__") for d in extras]:
        src = DB_DIR / f"{db}.duckdb"
        if not src.exists():
            try:
                build(db, verbose=False)
            except Exception as e:
                print(f"  skipping {db}: {type(e).__name__}")
                continue
        con.execute(f"ATTACH '{src}' AS s (READ_ONLY)")
        tables = [r[0] for r in con.execute("SELECT table_name FROM duckdb_tables() WHERE database_name = 's'").fetchall()]
        for t in tables:
            con.execute(f'CREATE TABLE main."{prefix}{t}" AS SELECT * FROM s."{t}"')
        con.execute("DETACH s")
    con.close()


def make_context(name, target):
    con = duckdb.connect(str(DB_DIR / f"{name}.duckdb"), read_only=True)
    cols = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
    ).fetchall()
    known, keyinfo = set(all_bird()), {}

    def origin(t):
        head, sep, rest = t.partition("__")
        return (head, rest) if sep and head in known else (target, t)

    lines, current, descs = [], None, {}
    for table, col, dtype in cols:
        orig, base = origin(table)
        if orig not in keyinfo:
            keyinfo[orig] = keys(orig)
        if table != current:
            current = table
            descs = load_descriptions(orig, base)
            lines.append(f"\nTable {table}")
        flag = " [primary key]" if col in keyinfo[orig][0].get(base, []) else ""
        d = descs.get(col.lower(), {})
        extra = []
        if d.get("desc"):
            extra.append(d["desc"])
        if d.get("values"):
            extra.append("values: " + d["values"])
        ex = example_values(con, table, col)
        if ex:
            extra.append("examples: " + ", ".join(ex))
        lines.append(f"  - {col} ({dtype}){flag}" + (": " + " | ".join(extra) if extra else ""))
    fks = []
    for orig, (_, fk) in keyinfo.items():
        p = "" if orig == target else f"{orig}__"
        fks += [f"  - {p}{t}.{c} -> {p}{rt}.{rc}" for t, c, rt, rc in fk]
    text = "Database schema with descriptions:" + "\n".join(lines)
    text += ("\n\nForeign keys:\n" + "\n".join(fks)) if fks else ""
    out = CTX_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "L2.txt").write_text(text)
    n_tables = sum(1 for l in lines if l.startswith("\nTable "))
    print(f"  {name}: {n_tables} tables, about {len(text) // 4} tokens")


def main(target):
    others = [d for d in all_bird() if d != target]
    arms = {"scaled_0": [], "scaled_1": [d for d in DBS if d != target], "scaled_2": others}
    gold = pickle.load(open(GOLD_FILE, "rb"))
    mine = [json.loads(l) for l in open(QUESTIONS_FILE)]
    mine = [q for q in mine if q["db"] == target and not q.get("ambiguous")]
    with open(SCALE_QUESTIONS_FILE, "w") as f:
        for name, extras in arms.items():
            print("building", name)
            make_db(name, target, extras)
            make_context(name, target)
            for q in mine:
                new = {**q, "question_id": f"{q['question_id']}@{name}", "db": name}
                gold[new["question_id"]] = gold[q["question_id"]]
                f.write(json.dumps(new) + "\n")
    pickle.dump(gold, open(GOLD_FILE, "wb"))
    print(len(mine), "questions per arm written to", SCALE_QUESTIONS_FILE)
    print("Compare the token counts above with num_ctx in models.yaml before you run.")


if __name__ == "__main__":
    main(sys.argv[1])
