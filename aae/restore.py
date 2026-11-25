import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from aae.merge import read_records

FOLDERS = ("artifacts", "runs", "contexts", "labels", "docs")


def nlines(path):
    p = Path(path)
    return sum(1 for _ in open(p)) if p.exists() else 0


def find_states(bases):
    found = []
    for base in map(Path, bases):
        if not base.exists():
            continue
        for dp, dn, _ in os.walk(base):
            if len(Path(dp).relative_to(base).parts) > 6:
                dn[:] = []
                continue
            if {"artifacts", "runs"} <= set(dn):
                found.append(Path(dp))
                dn[:] = []
    return sorted(found, key=lambda s: -nlines(s / "runs" / "runs.jsonl"))


def copy_missing(src, dst):
    n = 0
    for p in Path(src).rglob("*"):
        if p.is_file():
            target = Path(dst) / p.relative_to(src)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
                n += 1
    return n


def merge_jsonl(dst, sources):
    seen, lines = set(), []
    for p in [Path(dst), *map(Path, sources)]:
        if not p.exists():
            continue
        for rec in read_records(p):
            rid = rec.get("run_id")
            if rid is None or rid in seen:
                continue
            seen.add(rid)
            lines.append(json.dumps(rec, default=str) + "\n")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("".join(lines))
    return len(lines)


def frozen_parts(state):
    p = Path(state) / "artifacts" / "frozen.json"
    return json.load(open(p))["parts"] if p.exists() else None


def restore(bases, dest="."):
    dest = Path(dest)
    states = find_states(bases)
    summary = {"states": [str(s) for s in states], "files_copied": 0, "merged": {}, "plans_consistent": True}
    already = (dest / "runs" / "runs.jsonl").exists() or (dest / "artifacts" / "questions.jsonl").exists()
    if not states:
        return summary
    if not already:
        for s in states:
            for folder in FOLDERS:
                if (s / folder).exists():
                    summary["files_copied"] += copy_missing(s / folder, dest / folder)
    names = sorted({p.name for s in states for p in (s / "runs").glob("*.jsonl") if p.name != "sample.jsonl"})
    for name in names:
        summary["merged"][name] = merge_jsonl(dest / "runs" / name, [s / "runs" / name for s in states])
    plans = [pp for pp in (frozen_parts(s) for s in states) if pp]
    summary["plans_consistent"] = all(pp == plans[0] for pp in plans)
    return summary


def snapshot(dest="."):
    snap = {}
    for p in (Path(dest) / "runs").glob("*.jsonl"):
        if p.name != "sample.jsonl":
            snap[p.name] = [r["run_id"] for r in read_records(p) if "run_id" in r]
    out = Path(tempfile.gettempdir()) / "aae_restored_ids.json"
    json.dump(snap, open(out, "w"))
    return {k: len(v) for k, v in snap.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="restore saved sessions and merge their run logs by run id")
    ap.add_argument("--from", dest="bases", action="append", default=[])
    ap.add_argument("--to", default=".")
    args = ap.parse_args(argv)
    bases = args.bases or ["/kaggle/input", str(Path(tempfile.gettempdir()) / "aae_pub")]
    summary = restore(bases, args.to)
    print(json.dumps(summary, indent=2))
    if not summary["plans_consistent"]:
        print("WARNING: the saved sessions were frozen with different plans. Check before you report them together.")
    print("run ids already present at the start:", snapshot(args.to))
    return 0


if __name__ == "__main__":
    sys.exit(main())
