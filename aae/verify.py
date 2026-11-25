import argparse
import json
import sys
from pathlib import Path

from aae.headline import compute_metrics, half_up, load_summary, load_table


def tolerance(claim):
    if "tolerance" in claim:
        return float(claim["tolerance"])
    return 0.5 * 10 ** (-int(claim.get("decimals", 0)))


def check(claims, metrics):
    results = []
    for c in claims:
        got = metrics.get(c["metric"])
        if got is None:
            results.append((c, None, "MISSING"))
            continue
        if "tolerance" in c:
            ok = abs(got - float(c["expected"])) <= tolerance(c) + 1e-9
        else:
            places = int(c.get("decimals", 0))
            ok = half_up(got, places) == half_up(float(c["expected"]), places)
        results.append((c, got, "PASS" if ok else "FAIL"))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="compare the numbers in the post with the results table")
    ap.add_argument("--table", default="results/runs_table.csv")
    ap.add_argument("--claims", default="plan/claims.json")
    ap.add_argument("--rescore-summary", default="results/rescore_summary.json")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    claims = json.load(open(args.claims))["claims"]
    df = load_table(args.table)
    metrics = compute_metrics(df, rescore_summary=load_summary(args.rescore_summary))
    results = check(claims, metrics)
    width = max(len(c["metric"]) for c, _, _ in results)
    for c, got, status in results:
        shown = "n/a" if got is None else f"{got:.3f}"
        print(f"{status:7} {c['metric']:{width}}  post says {c['expected']:<8} table gives {shown:>9}   {c.get('where', '')}")
    bad = [r for r in results if r[2] != "PASS"]
    print(f"\n{len(results) - len(bad)} of {len(results)} claims match")
    if args.json:
        Path(args.json).write_text(
            json.dumps([{"metric": c["metric"], "expected": c["expected"], "got": g, "status": s} for c, g, s in results], indent=2)
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
