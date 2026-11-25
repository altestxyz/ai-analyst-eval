import datetime
import hashlib
import json
import math
import random
from pathlib import Path

LEVELS = ("L1", "L2", "L3")


def part_of(q):
    return "ambiguous" if q.get("ambiguous") else ("taxi" if q["db"] == "nyc_taxi" else "bird")


def ordered_questions(questions, seed, priority=("bird", "taxi", "ambiguous")):
    qs = list(questions)
    random.Random(seed).shuffle(qs)
    rank = {p: i for i, p in enumerate(priority)}
    return sorted(qs, key=lambda q: rank.get(part_of(q), len(rank)))


def run_id(qid, level, model, repeat):
    return f"{qid}|{level}|{model}|{repeat}"


def job_ids(q, models, repeats, levels=LEVELS):
    return [run_id(q["question_id"], l, m, r) for l in levels for m in models for r in range(repeats)]


def missing(q, done, models, repeats):
    return [j for j in job_ids(q, models, repeats) if j not in done]


def read_runs(path):
    p = Path(path)
    if not p.exists():
        return
    for line in open(p):
        try:
            yield json.loads(line)
        except Exception:
            continue


def done_ids(path):
    return {r["run_id"] for r in read_runs(path) if not r.get("tag")}


def baseline_ids(path, cutoff_iso):
    return {r["run_id"] for r in read_runs(path) if not r.get("tag") and str(r.get("timestamp", "")) < cutoff_iso}


def make_slices(qids, per_slice):
    if not qids:
        return []
    n = math.ceil(len(qids) / per_slice)
    base, extra = divmod(len(qids), n)
    out, i = [], 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        out.append(qids[i : i + size])
        i += size
    return out


def make_plan(questions, baseline, models, repeats, per_slice, seed, priority=("bird", "taxi", "ambiguous"), addons=()):
    ordered = ordered_questions(questions, seed, priority)
    remaining = [q["question_id"] for q in ordered if missing(q, baseline, models, repeats)]
    slices = make_slices(remaining, per_slice)
    plan = {
        "slices": slices,
        "addons": list(addons),
        "models": list(models),
        "repeats": repeats,
        "n_remaining": len(remaining),
        "per_slice": per_slice,
    }
    plan["hash"] = hashlib.sha256(json.dumps([slices, list(addons), list(models), repeats]).encode()).hexdigest()[:12]
    return plan


def slice_missing(plan, idx, done, questions_by_id):
    out = {m: 0 for m in plan["models"]}
    for qid in plan["slices"][idx]:
        for j in missing(questions_by_id[qid], done, plan["models"], plan["repeats"]):
            out[j.split("|")[2]] += 1
    return out


def pick(plan, done, questions_by_id, mode="auto", start="2026-01-01", today=None, have_shards=False, addon_done=None):
    addon_done = addon_done or {}
    n = len(plan["slices"])
    total = n + len(plan["addons"])

    def entry(i, why):
        if i < n:
            return {"kind": "main", "index": i, "qids": plan["slices"][i], "why": why}
        return {"kind": plan["addons"][i - n], "index": i, "qids": [], "why": why}

    def incomplete(i):
        if i < n:
            return sum(slice_missing(plan, i, done, questions_by_id).values()) > 0
        return not addon_done.get(plan["addons"][i - n], False)

    if str(mode).strip().isdigit():
        i = int(mode)
        if not 0 <= i < total:
            return {"kind": "none", "index": i, "qids": [], "why": f"SHARD={i} is out of range (0 to {total - 1})"}
        return entry(i, f"SHARD={i} set by hand")
    today = today or datetime.datetime.now(datetime.timezone.utc).date()
    day = (today - datetime.date.fromisoformat(start)).days
    if day < 0:
        return {
            "kind": "none",
            "index": -1,
            "qids": [],
            "why": f"today ({today}) is before SCHEDULE_START ({start}). Set SHARD to a slice number to run one by hand",
        }
    if day < total and incomplete(day):
        return entry(day, f"day {day} since SCHEDULE_START")
    if day < total:
        why = f"day {day}'s slice is already complete"
    else:
        why = f"the schedule has run its {total} slices"
    if have_shards:
        for i in range(total):
            if incomplete(i):
                return entry(i, f"{why}; filling the first incomplete slice")
        return {"kind": "none", "index": -1, "qids": [], "why": f"{why}, and nothing is missing"}
    return {
        "kind": "none",
        "index": -1,
        "qids": [],
        "why": f"{why}. Earlier shards are not visible here, so nothing is re-run. Set SHARD to a slice number to force one",
    }


def describe(plan, done, questions_by_id):
    lines = [
        f"plan {plan['hash']}: {plan['n_remaining']} questions still missing from the baseline, "
        f"{len(plan['slices'])} main slices of about {plan['per_slice']}" + (f", then add-ons {plan['addons']}" if plan["addons"] else "")
    ]
    for i, s in enumerate(plan["slices"]):
        parts = {}
        for qid in s:
            p = part_of(questions_by_id[qid])
            parts[p] = parts.get(p, 0) + 1
        miss = slice_missing(plan, i, done, questions_by_id)
        lines.append(f"  slice {i}: {len(s)} questions {parts}, jobs missing now {miss}")
    for j, a in enumerate(plan["addons"]):
        lines.append(f"  slice {len(plan['slices']) + j}: add-on '{a}'")
    return "\n".join(lines)


def check_plans(art_dir="artifacts"):
    hs = {}
    for p in Path(art_dir).glob("shard_*.json"):
        try:
            hs[p.name] = json.load(open(p)).get("plan_hash")
        except Exception:
            pass
    if len({h for h in hs.values()}) > 1:
        print("WARNING: shards were made from different plans (the baseline or the settings changed between runs):", hs)
        print("  Runs are still merged by run id, so nothing is counted twice, but some questions may have run on two days.")
        return False
    return True


if __name__ == "__main__":
    check_plans()
