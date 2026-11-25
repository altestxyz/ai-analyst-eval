import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    "run_id",
    "question_id",
    "part",
    "db",
    "difficulty",
    "model",
    "model_id",
    "context",
    "repeat",
    "tag",
    "correct_logged",
    "strict",
    "lenient",
    "rescore_status",
    "asked",
    "had_error",
    "had_suspicious",
    "format_fail",
    "verify_changed",
    "shape",
    "n_turns",
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "max_prompt_tokens",
    "latency_s",
    "timestamp",
]
RESCORED_OK = ("ok", "sql_error")


def part_of(rec):
    qid = str(rec.get("question_id", ""))
    if qid.startswith("taxi-amb") or rec.get("ambiguous"):
        return "ambiguous"
    return "taxi" if rec.get("db") == "nyc_taxi" else "bird"


def load_rescored(path):
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return {x["run_id"]: x for x in data}


def tri(value):
    if value is None:
        return ""
    return "1" if bool(value) else "0"


def row_for(rec, rescored):
    logged = rec.get("correct")
    part = part_of(rec)
    rs = rescored.get(rec["run_id"])
    if part == "ambiguous":
        strict = lenient = None
        status = "not_scored"
    elif rs and rs.get("status") in RESCORED_OK:
        strict, lenient, status = rs.get("strict"), rs.get("tol"), rs["status"]
    else:
        strict = lenient = logged
        status = rs["status"] if rs else "logged"
    return {
        "run_id": rec["run_id"],
        "question_id": rec["question_id"],
        "part": part,
        "db": rec.get("db"),
        "difficulty": rec.get("difficulty") or "",
        "model": rec["model"],
        "model_id": rec.get("model_id", ""),
        "context": rec["context"],
        "repeat": rec.get("repeat", 0),
        "tag": rec.get("tag", ""),
        "correct_logged": tri(logged),
        "strict": tri(strict),
        "lenient": tri(lenient),
        "rescore_status": status,
        "asked": tri(rec.get("asked")),
        "had_error": tri(rec.get("had_error")),
        "had_suspicious": tri(rec.get("had_suspicious")),
        "format_fail": tri(rec.get("format_fail")),
        "verify_changed": tri(rec.get("verify_changed")),
        "shape": rec.get("shape") or "",
        "n_turns": rec.get("n_turns", ""),
        "input_tokens": rec.get("input_tokens", 0),
        "cache_read_tokens": rec.get("cache_read_tokens", 0),
        "cache_write_tokens": rec.get("cache_write_tokens", 0),
        "output_tokens": rec.get("output_tokens", 0),
        "reasoning_tokens": rec.get("reasoning_tokens", ""),
        "max_prompt_tokens": rec.get("max_prompt_tokens", ""),
        "latency_s": rec.get("latency_s", ""),
        "timestamp": rec.get("timestamp", ""),
    }


def build(runs_path, rescored_path=None, shapes_path=None):
    rescored = load_rescored(rescored_path)
    shapes = {}
    if shapes_path and Path(shapes_path).exists():
        for line in open(shapes_path):
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                if "run_id" in d and "shape" in d:
                    shapes[d["run_id"]] = d["shape"]
    rows = []
    with open(runs_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("shape") and rec["run_id"] in shapes:
                rec["shape"] = shapes[rec["run_id"]]
            rows.append(row_for(rec, rescored))
    return rows


def write(rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="one row per run, with no SQL text and no answer keys")
    ap.add_argument("--runs", default="runs/runs.jsonl")
    ap.add_argument("--rescored", default="")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--out", default="results/runs_table.csv")
    args = ap.parse_args(argv)
    rows = build(args.runs, args.rescored or None, args.shapes or None)
    write(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
