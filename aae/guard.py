import numbers
import threading

import duckdb
import sqlglot
from sqlglot import exp

ROW_CAP = 10_000
TIMEOUT_S = 30
SET_OPS = getattr(exp, "SetOperation", exp.Union)


class GuardError(Exception):
    pass


def open_connection(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def check_sql(sql):
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise GuardError(f"That isn't valid SQL: {e}")
    if len(statements) != 1:
        raise GuardError("Send exactly one SQL statement.")
    if not isinstance(statements[0], (exp.Select, SET_OPS)):
        raise GuardError("This connection is read-only. Rewrite the query as a SELECT.")


def run_sql(con, sql):
    check_sql(sql)
    timer = threading.Timer(TIMEOUT_S, con.interrupt)
    timer.start()
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(ROW_CAP + 1)
        columns = [d[0] for d in cur.description]
    except duckdb.InterruptException:
        raise GuardError(f"The query stopped after {TIMEOUT_S} seconds. Add filters or aggregate earlier.")
    except duckdb.Error as e:
        raise GuardError(f"SQL error: {e}")
    finally:
        timer.cancel()
    return {"columns": columns, "rows": rows[:ROW_CAP], "truncated": len(rows) > ROW_CAP}


def suspicious(res, nudge=True):
    rows = res["rows"]
    if not rows:
        return "The query returned no rows. Check your filters and joins, then try again."
    if not nudge:
        return None
    if res["truncated"]:
        return f"The result was cut off at {ROW_CAP:,} rows. Aggregate or filter so the answer fits, then try again."
    if len(rows) == 1 and all(v is None for v in rows[0]):
        return "The query returned a single NULL. Check your filters and joins, and whether you used the right column, then try again."
    if len(rows) == 1 and all(isinstance(v, numbers.Number) and not isinstance(v, bool) and v == 0 for v in rows[0]):
        return (
            "The query returned a single zero. If that is unexpected, check your filters and "
            "joins and try again. If it is right, send the same query again."
        )
    return None
