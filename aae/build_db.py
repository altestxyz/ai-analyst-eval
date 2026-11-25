import sys

import duckdb

from aae.config import BIRD_DIR, DB_DIR, DBS


def build(db_id, verbose=True):
    src = BIRD_DIR / "dev_databases" / db_id / f"{db_id}.sqlite"
    out = DB_DIR / f"{db_id}.duckdb"
    DB_DIR.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    con = duckdb.connect(str(out))
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{src}' AS src (TYPE sqlite, READ_ONLY)")
    tables = [
        r[0]
        for r in con.execute("SELECT table_name FROM duckdb_tables() WHERE database_name = 'src'").fetchall()
        if not r[0].startswith("sqlite_")
    ]
    for t in tables:
        con.execute(f'CREATE TABLE main."{t}" AS SELECT * FROM src."{t}"')
        if verbose:
            n = con.execute(f'SELECT count(*) FROM main."{t}"').fetchone()[0]
            print(f"  {db_id}.{t}: {n} rows")
    con.execute("DETACH src")
    con.close()


if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or DBS
    for db in wanted:
        if (DB_DIR / f"{db}.duckdb").exists() and "--force" not in sys.argv:
            print("already built:", db)
            continue
        print("building", db)
        build(db)
