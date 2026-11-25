import argparse
import collections
import itertools
import json
import numbers
import pickle
import sys
from pathlib import Path

from aae.models import load_models
from aae.score import same_result

RESCORED = ("ok", "sql_error")


def truthy(value):
    return value is True or str(value).lower() == "true"


def _vkey(v):
    if isinstance(v, numbers.Number) and not isinstance(v, bool):
        return (1, f"{float(v):.6g}")
    return (0, "") if v is None else (2, str(v))


def tolerant_same(pred, gold, ordered):
    if same_result(pred, gold, ordered):
        return True
    if not gold or len(pred) != len(gold):
        return False
    gc, pc = len(gold[0]), len(pred[0])
    if pc <= gc:
        return False
    candidates = []
    for j in range(gc):
        gcol = sorted(_vkey(r[j]) for r in gold)
        match = [i for i in range(pc) if sorted(_vkey(r[i]) for r in pred) == gcol]
        if not match:
            return False
        candidates.append(match)
    for combo in itertools.product(*candidates):
        if len(set(combo)) < gc:
            continue
        if same_result([tuple(r[i] for i in combo) for r in pred], gold, ordered):
            return True
    return False


def load_gold(paths):
    gold = {}
    for p in paths:
        with open(p, "rb") as f:
            for key, value in pickle.load(f).items():
                gold[str(key)] = value
    return gold


def rescore(runs, gold, db_dir, groups):
    from aae.guard import GuardError, open_connection, run_sql

    connections = {}

    def con_for(db):
        if db not in connections:
            path = Path(db_dir) / f"{db}.duckdb"
            connections[db] = open_connection(path) if path.exists() else None
        return connections[db]

    results = []
    for n, r in enumerate(runs, 1):
        rec = {
            "group": groups.get(r["model"], "other"),
            "run_id": r["run_id"],
            "q": str(r["question_id"]),
            "db": r["db"],
            "model": r["model"],
            "ctx": r["context"],
            "repeat": str(r.get("repeat")),
            "logged_correct": truthy(r.get("correct")),
        }
        g, sql = gold.get(str(r["question_id"])), r.get("final_sql")
        if truthy(r.get("asked")) or not sql:
            rec["status"] = "no_sql"
        elif g is None:
            rec["status"] = "no_gold"
        elif con_for(r["db"]) is None:
            rec["status"] = "no_db"
        else:
            try:
                res = run_sql(con_for(r["db"]), sql)
                gr = [tuple(x) for x in g["rows"]]
                pr = [tuple(x) for x in res["rows"]]
                ordered = truthy(g["ordered"])
                rec["strict"] = bool(same_result(pr, gr, ordered)) and not res["truncated"]
                rec["tol"] = bool(tolerant_same(pr, gr, ordered)) and not res["truncated"]
                rec["status"] = "ok"
            except GuardError:
                rec["status"] = "sql_error"
                rec["strict"] = rec["tol"] = False
        results.append(rec)
        if n % 500 == 0:
            print(f"  {n}/{len(runs)}", flush=True)
    return results


def summarize(results):
    ok = [x for x in results if x["status"] == "ok"]
    agree = sum(x["strict"] == x["logged_correct"] for x in ok)
    flips = [x for x in ok if x["tol"] and not x["strict"]]
    return {
        "rescored_runs": len(ok),
        "strict_agree": agree,
        "strict_agreement_pct": round(100 * agree / max(1, len(ok)), 2),
        "status_counts": {f"{g}|{s}": n for (g, s), n in sorted(collections.Counter((x["group"], x["status"]) for x in results).items())},
        "lenient_only_right": len(flips),
        "disagreements_by_model": dict(collections.Counter(x["model"] for x in ok if x["strict"] != x["logged_correct"])),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="re-run the saved final SQL of every run and score it strictly and leniently")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--runs", default="runs/runs.jsonl")
    run.add_argument("--gold", nargs="+", default=["artifacts/gold.pkl"])
    run.add_argument("--db-dir", default="data/duckdb")
    run.add_argument("--out", default="results/rescored.json")
    run.add_argument("--summary", default="results/rescore_summary.json")
    summ = sub.add_parser("summarize")
    summ.add_argument("rescored")
    summ.add_argument("--summary", default="results/rescore_summary.json")
    args = ap.parse_args(argv)
    if args.cmd == "summarize":
        results = json.load(open(args.rescored))
    else:
        models = load_models()
        groups = {k: ("local" if v.get("provider") == "ollama" else "frontier") for k, v in models.items()}
        runs = [json.loads(line) for line in open(args.runs) if line.strip()]
        results = rescore(runs, load_gold(args.gold), args.db_dir, groups)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"))
    summary = summarize(results)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.summary, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
