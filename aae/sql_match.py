import argparse
import collections
import csv
import re
import sys
from pathlib import Path

from aae.merge import discover, read_records

try:
    import sqlglot
    from sqlglot import exp
    from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

    HAVE_GLOT = True
except ImportError:
    HAVE_GLOT = False


def text_norm(sql):
    s = (sql or "").strip().rstrip(";").strip()
    return re.sub(r"\s+", " ", s).lower()


def canon_aliases(tree):
    mapping = {}
    for t in tree.find_all(exp.Table):
        alias = t.alias or t.name
        if alias and alias.lower() not in mapping:
            mapping[alias.lower()] = f"t{len(mapping) + 1}"
    for t in tree.find_all(exp.Table):
        key = (t.alias or t.name).lower()
        if t.alias:
            t.set("alias", exp.TableAlias(this=exp.to_identifier(mapping[key])))
    for c in tree.find_all(exp.Column):
        if c.table and c.table.lower() in mapping:
            c.set("table", exp.to_identifier(mapping[c.table.lower()]))
    return tree


def ast_norm(sql, ignore_select_order=False):
    if not HAVE_GLOT or not sql:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
        tree = normalize_identifiers(tree, dialect="duckdb")
        tree = canon_aliases(tree)
        if ignore_select_order:
            for sel in tree.find_all(exp.Select):
                exprs = [e.this if isinstance(e, exp.Alias) else e for e in sel.expressions]
                sel.set("expressions", sorted(exprs, key=lambda e: e.sql(dialect="duckdb")))
        return tree.sql(dialect="duckdb", normalize=True)
    except Exception:
        return None


def load_runs(paths):
    files = []
    for p in map(Path, paths):
        files.extend(discover(p) if p.is_dir() else [p])
    seen = {}
    for f in files:
        for rec in read_records(f):
            seen.setdefault(rec["run_id"], rec)
    return list(seen.values())


def compare(runs, gold):
    rows = []
    for r in runs:
        qid = str(r["question_id"])
        if qid not in gold or r.get("asked") or not r.get("final_sql"):
            continue
        g, m = gold[qid], r["final_sql"]
        gt = ast_norm(g)
        rows.append(
            {
                "run_id": r["run_id"],
                "question_id": qid,
                "model": r["model"],
                "context": r["context"],
                "exec_correct": bool(r.get("correct")),
                "text_exact": text_norm(g) == text_norm(m),
                "ast_exact": (gt == ast_norm(m)) if gt else None,
                "ast_select": (ast_norm(g, True) == ast_norm(m, True)) if gt else None,
                "gold_sql": g,
                "model_sql": m,
            }
        )
    return rows


def report(rows):
    levels = ["text_exact"] + (["ast_exact", "ast_select"] if HAVE_GLOT else [])
    print(f"runs compared: {len(rows)}   sqlglot available: {HAVE_GLOT}")
    for lvl in levels:
        ok = [r for r in rows if r[lvl] is not None]
        both = sum(r[lvl] and r["exec_correct"] for r in ok)
        sql_only = sum(r[lvl] and not r["exec_correct"] for r in ok)
        exec_only = sum((not r[lvl]) and r["exec_correct"] for r in ok)
        print(f"\n[{lvl}]  n={len(ok)}")
        print(f"  SQL matches gold and result correct : {both}")
        print(f"  SQL matches gold but result wrong   : {sql_only}")
        print(f"  SQL differs but result correct      : {exec_only}")
        print(f"  SQL differs and result wrong        : {len(ok) - both - sql_only - exec_only}")
        print(
            f"  SQL-match accuracy {100 * (both + sql_only) / max(1, len(ok)):.1f}%  "
            f"vs execution accuracy {100 * (both + exec_only) / max(1, len(ok)):.1f}%"
        )
    last = levels[-1]
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r[last] is None:
            continue
        k = (r["model"], r["context"])
        agg[k][0] += 1
        agg[k][1] += r["exec_correct"]
        agg[k][2] += r[last]
    print(f"\nBy model and context (execution accuracy vs {last}):")
    for k in sorted(agg):
        n, e, s = agg[k]
        print(f"  {k[0]} {k[1]}  n={n:4d}  exec {100 * e / n:5.1f}%  sql-match {100 * s / n:5.1f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description="compare final SQL with gold SQL without running any query")
    ap.add_argument("--questions", nargs="+", default=["artifacts/questions.jsonl", "artifacts/taxi_questions.jsonl"])
    ap.add_argument("--runs", nargs="+", default=["runs"])
    ap.add_argument("--out", default="results/sql_match")
    args = ap.parse_args(argv)
    gold = {}
    for p in args.questions:
        if Path(p).exists():
            for q in read_records(p):
                if q.get("gold_sql"):
                    gold[str(q["question_id"])] = q["gold_sql"]
    rows = compare(load_runs(args.runs), gold)
    if not rows:
        raise SystemExit("no scorable runs found")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out + "_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
