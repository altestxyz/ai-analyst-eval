import csv
import json
import pickle
import random
import re
import sys
from pathlib import Path

from aae.config import GOLD_FILE, QUESTIONS_FILE, RUNS_FILE
from aae.llm import chat
from aae.models import load_models

LABELS = [
    "wrong table",
    "wrong join",
    "wrong filter",
    "wrong metric definition",
    "unresolved ambiguity",
    "unneeded clarification",
    "SQL error",
    "timeout",
]
MODELS = load_models()
Path("labels").mkdir(exist_ok=True)


def failures():
    qs = {json.loads(l)["question_id"]: json.loads(l) for l in open(QUESTIONS_FILE)}
    runs = [json.loads(l) for l in open(RUNS_FILE)]
    out = [(r, qs[r["question_id"]]) for r in runs if r["correct"] is False and r["question_id"] in qs]
    out.sort(key=lambda p: p[0]["run_id"])
    random.Random(1).shuffle(out)
    return out


def cohen_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def sample():
    pairs = failures()
    with open("labels/to_label.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "question", "gold_sql", "final_sql", "attempts", "result_preview", "label"])
        for r, q in pairs[:50]:
            w.writerow(
                [r["run_id"], q["question"], q.get("gold_sql"), r["final_sql"], json.dumps(r["attempts"])[:600], r["result_preview"], ""]
            )
    print(f"Wrote labels/to_label.csv ({min(50, len(pairs))} failures). Fill in the label column using one of:")
    print(LABELS)


def run_judge(limit=0):
    gold = pickle.load(open(GOLD_FILE, "rb"))
    path = Path("labels/judge.jsonl")
    seen = {json.loads(l)["run_id"] for l in open(path)} if path.exists() else set()
    pairs = failures()
    if limit:
        pairs = pairs[:limit]
    out = open(path, "a")
    n = 0
    for r, q in pairs:
        if r["run_id"] in seen:
            continue
        g = gold.get(q["question_id"], {}).get("rows", [])[:5]
        prompt = (
            f"Question: {q['question']}\nGold SQL: {q.get('gold_sql')}\nGold result (first rows): {g}\n"
            f"Generated SQL: {r['final_sql']}\nGenerated result (first rows): {r['result_preview']}\n"
            f"Attempts and errors: {json.dumps(r['attempts'])[:800]}\nAsked a clarifying question: {r['asked']}\n\n"
            f"Pick the single best label for why this run failed, from: {LABELS}. "
            'Reply as JSON: {"label": "...", "reason": "one sentence"}'
        )
        reply = chat(
            MODELS["JUDGE"], [{"role": "system", "content": "You label text-to-SQL failures."}, {"role": "user", "content": prompt}]
        )
        m = re.search(r"\{.*\}", reply["text"], re.S)
        try:
            lab = json.loads(m.group(0))
        except Exception:
            lab = {"label": "unparsed", "reason": reply["text"][:100]}
        out.write(json.dumps({"run_id": r["run_id"], **lab}) + "\n")
        out.flush()
        n += 1
        if n % 50 == 0:
            print(f"  judged {n} failures", flush=True)
    print(f"judged {n} new failures, {len(seen)} were already done")


def kappa():
    p = Path("labels/to_label.csv")
    hand = {r["run_id"]: r["label"] for r in csv.DictReader(open(p)) if r["label"]} if p.exists() else {}
    if not hand:
        print("No hand labels yet. Fill in labels/to_label.csv (see the notebook), then rerun this.")
        return
    judge = {}
    for line in open("labels/judge.jsonl"):
        j = json.loads(line)
        judge[j["run_id"]] = j["label"]
    ids = [i for i in hand if i in judge]
    k = cohen_kappa([hand[i] for i in ids], [judge[i] for i in ids])
    print(f"kappa = {k:.2f} on {len(ids)} runs. Need at least 0.70.")
    if k < 0.7:
        print("Below the gate: do not use the judge's labels for headline claims. Label those by hand.")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run_judge(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        {"sample": sample, "kappa": kappa}[cmd]()
