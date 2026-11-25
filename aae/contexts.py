import csv
import sqlite3
import sys

import duckdb

from aae.config import BIRD_DIR, CTX_DIR, DB_DIR, DBS, TAXI_DIR

TAXI = "nyc_taxi"


def desc_dir(db_id):
    if db_id == TAXI:
        return TAXI_DIR / "database_description"
    return BIRD_DIR / "dev_databases" / db_id / "database_description"


def keys(db_id):
    if db_id == TAXI:
        return {}, [("trips", "PULocationID", "zones", "LocationID"), ("trips", "DOLocationID", "zones", "LocationID")]
    path = BIRD_DIR / "dev_databases" / db_id / f"{db_id}.sqlite"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    pk, fk = {}, []
    for t in tables:
        pk[t] = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")') if r[5]]
        for r in con.execute(f'PRAGMA foreign_key_list("{t}")'):
            fk.append((t, r[3], r[2], r[4]))
    con.close()
    return pk, fk


def load_descriptions(db_id, table):
    folder = desc_dir(db_id)
    match = next((p for p in folder.glob("*.csv") if p.stem.lower() == table.lower()), None)
    out = {}
    if match is None:
        return out
    with open(match, encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("original_column_name") or "").strip().lower()
            out[name] = {
                "desc": (row.get("column_description") or "").strip(),
                "values": (row.get("value_description") or "").strip(),
            }
    return out


def example_values(con, table, col, n=3):
    rows = con.execute(
        f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL AND CAST("{col}" AS VARCHAR) <> \'\' ORDER BY 1 LIMIT {n}'
    ).fetchall()
    return [str(r[0])[:40] for r in rows]


def assemble_l3(db_id):
    out = CTX_DIR / db_id
    sem, l2 = out / "semantic.yaml", out / "L2.txt"
    if sem.exists() and l2.exists():
        (out / "L3.txt").write_text(l2.read_text() + "\n\nSemantic model:\n" + sem.read_text())
        return True
    return False


def build(db_id):
    con = duckdb.connect(str(DB_DIR / f"{db_id}.duckdb"), read_only=True)
    cols = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
    ).fetchall()
    pk, fk = keys(db_id)
    l1, l2, current = [], [], None
    descs = {}
    for table, col, dtype in cols:
        if table != current:
            current = table
            descs = load_descriptions(db_id, table)
            l1.append(f"\nTable {table}")
            l2.append(f"\nTable {table}")
        flag = " [primary key]" if col in pk.get(table, []) else ""
        l1.append(f"  - {col} ({dtype}){flag}")
        d = descs.get(col.lower(), {})
        extra = []
        if d.get("desc"):
            extra.append(d["desc"])
        if d.get("values"):
            extra.append("values: " + d["values"])
        ex = example_values(con, table, col)
        if ex:
            extra.append("examples: " + ", ".join(ex))
        l2.append(f"  - {col} ({dtype}){flag}" + (": " + " | ".join(extra) if extra else ""))
    fk_lines = [f"  - {t}.{c} -> {rt}.{rc}" for t, c, rt, rc in fk]
    tail = ("\n\nForeign keys:\n" + "\n".join(fk_lines)) if fk_lines else ""
    out = CTX_DIR / db_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "L1.txt").write_text("Database schema:" + "\n".join(l1) + tail)
    (out / "L2.txt").write_text("Database schema with descriptions:" + "\n".join(l2) + tail)
    if not assemble_l3(db_id):
        print(f"  no semantic.yaml for {db_id} yet, skipping L3")
    for name in ("L1", "L2", "L3"):
        p = out / f"{name}.txt"
        if p.exists():
            print(f"  {db_id} {name}: about {len(p.read_text()) // 4} tokens")


def all_dbs():
    return DBS + ([TAXI] if (DB_DIR / f"{TAXI}.duckdb").exists() else [])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "l3":
        for db in all_dbs():
            print(db, "L3 assembled" if assemble_l3(db) else "no semantic.yaml")
    else:
        for db in all_dbs():
            print("building contexts for", db)
            build(db)
