import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUN_KEYS = {"run_id", "question_id", "model", "context"}
SKIP_NAMES = {"sample.jsonl", "errors.log"}


def read_records(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def is_run_log(path):
    for rec in read_records(path):
        return RUN_KEYS <= set(rec)
    return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(folder, exclude=()):
    skip = {Path(e).resolve() for e in exclude}
    found = []
    for p in sorted(Path(folder).rglob("*.jsonl")):
        if p.name in SKIP_NAMES or p.resolve() in skip:
            continue
        if is_run_log(p):
            found.append(p)
    return found


def plan_fingerprints(folder):
    seen = {}
    for p in Path(folder).rglob("frozen.json"):
        try:
            seen[str(p)] = json.load(open(p))["parts"]
        except (OSError, ValueError, KeyError):
            continue
    for p in Path(folder).rglob("shard_*.json"):
        try:
            seen[str(p)] = json.load(open(p)).get("plan_hash")
        except (OSError, ValueError):
            continue
    return seen


def pick(old, new, prefer):
    a, b = str(old.get("timestamp", "")), str(new.get("timestamp", ""))
    if prefer == "latest":
        return new if b >= a else old
    return old if a <= b else new


def merge(folder, prefer="first", exclude=()):
    sources = discover(folder, exclude)
    merged, origin = {}, {}
    per_source, conflicts = [], []
    for path in sources:
        n = new = same = differ = 0
        for rec in read_records(path):
            if "run_id" not in rec:
                continue
            n += 1
            rid = rec["run_id"]
            if rid not in merged:
                merged[rid], origin[rid] = rec, path.name
                new += 1
            elif merged[rid] == rec:
                same += 1
            else:
                differ += 1
                conflicts.append({"run_id": rid, "kept_from": origin[rid], "other": path.name})
                chosen = pick(merged[rid], rec, prefer)
                if chosen is rec:
                    merged[rid], origin[rid] = rec, path.name
        per_source.append(
            {
                "file": str(path.relative_to(folder)),
                "sha256": sha256(path),
                "records": n,
                "new_unique": new,
                "identical_duplicates": same,
                "conflicting_duplicates": differ,
            }
        )
    records = sorted(merged.values(), key=lambda r: (str(r.get("timestamp", "")), r["run_id"]))
    cells = defaultdict(set)
    for r in records:
        cells[(r["model"], r["context"])].add(str(r["question_id"]))
    manifest = {
        "input_folder": Path(folder).name,
        "prefer_on_conflict": prefer,
        "sources": per_source,
        "unique_runs": len(records),
        "records_read": sum(s["records"] for s in per_source),
        "identical_duplicates": sum(s["identical_duplicates"] for s in per_source),
        "conflicts": len(conflicts),
        "conflict_examples": conflicts[:20],
        "runs_by_model": dict(sorted(Counter(r["model"] for r in records).items())),
        "questions_by_model_context": {f"{m}|{c}": len(q) for (m, c), q in sorted(cells.items())},
        "plan_fingerprints": plan_fingerprints(folder),
    }
    prints = {json.dumps(v, sort_keys=True) for v in manifest["plan_fingerprints"].values()}
    manifest["plan_consistent"] = len(prints) <= 1
    return records, manifest


def write(records, manifest, out, manifest_path):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    manifest["output"] = {"file": out.name, "sha256": sha256(out), "lines": len(records)}
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="merge every run log under a folder into one log, by run_id")
    ap.add_argument("folder")
    ap.add_argument("--out", default="runs/runs.jsonl")
    ap.add_argument("--manifest", default="runs/merge_manifest.json")
    ap.add_argument("--prefer", choices=["first", "latest"], default="first")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder")
    records, manifest = merge(folder, args.prefer, exclude=[args.out])
    for s in manifest["sources"]:
        print(
            f"{s['file']:55} {s['records']:6} records, {s['new_unique']:6} new, "
            f"{s['identical_duplicates']:6} duplicates, {s['conflicting_duplicates']} conflicts"
        )
    print(
        f"\n{manifest['records_read']} records read, {manifest['unique_runs']} unique runs, "
        f"{manifest['identical_duplicates']} identical duplicates, {manifest['conflicts']} conflicts"
    )
    print("runs by model:", manifest["runs_by_model"])
    if not manifest["plan_consistent"]:
        print("WARNING: the sessions in this folder carry different plan fingerprints. Check before you report them together.")
    if args.dry_run:
        return 0
    write(records, manifest, args.out, args.manifest)
    print(f"wrote {args.out} and {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
