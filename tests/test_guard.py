import duckdb
import pytest

from aae import guard
from aae.guard import GuardError, open_connection, run_sql, suspicious


@pytest.fixture
def con(tmp_path):
    path = str(tmp_path / "t.duckdb")
    c = duckdb.connect(path)
    c.execute("CREATE TABLE t AS SELECT range AS id FROM range(20000)")
    c.close()
    return open_connection(path)


NASTY = [
    "DROP TABLE t",
    "INSERT INTO t VALUES (1)",
    "CREATE TABLE x AS SELECT 1",
    "COPY t TO '/tmp/out.csv'",
    "ATTACH '/tmp/other.duckdb' AS o",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT 1; DROP TABLE t",
    "PRAGMA database_list",
    "INSTALL httpfs",
    "SET enable_external_access = true",
]


@pytest.mark.parametrize("sql", NASTY)
def test_nasty_queries_are_blocked(con, sql):
    with pytest.raises(GuardError):
        run_sql(con, sql)


def test_select_and_cte_and_union_pass(con):
    assert run_sql(con, "SELECT 1")["rows"] == [(1,)]
    assert run_sql(con, "WITH a AS (SELECT 2 AS x) SELECT x FROM a")["rows"] == [(2,)]
    assert len(run_sql(con, "SELECT 1 UNION SELECT 2")["rows"]) == 2


def test_row_cap(con):
    r = run_sql(con, "SELECT * FROM t")
    assert len(r["rows"]) == 10_000 and r["truncated"]


def test_timeout(con, monkeypatch):
    monkeypatch.setattr(guard, "TIMEOUT_S", 1)
    with pytest.raises(GuardError, match="stopped after"):
        run_sql(con, "SELECT sum(a.range * b.range) FROM range(100000000) a, range(100000000) b")


def ok(rows, truncated=False):
    return {"rows": rows, "truncated": truncated}


def test_suspicious_results():
    assert suspicious(ok([(5,)])) is None
    assert suspicious(ok([(0, 5)])) is None
    assert "no rows" in suspicious(ok([]))
    assert "single zero" in suspicious(ok([(0,)]))
    assert "single NULL" in suspicious(ok([(None,)]))
    assert "cut off" in suspicious(ok([(1,)], truncated=True))
    assert suspicious(ok([(0,)]), nudge=False) is None
    assert "no rows" in suspicious(ok([]), nudge=False)
