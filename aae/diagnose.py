import json
import pickle
import re
import sys
import time
from pathlib import Path

import pandas as pd

STOP = {
    "what",
    "which",
    "that",
    "with",
    "from",
    "have",
    "were",
    "the",
    "and",
    "for",
    "how",
    "many",
    "give",
    "just",
    "name",
    "each",
    "than",
    "more",
    "most",
    "least",
    "average",
    "total",
    "number",
    "between",
    "their",
    "does",
    "when",
    "where",
}


def _load(path="runs/runs.jsonl"):
    p = Path(path)
    return pd.DataFrame([json.loads(l) for l in open(p)]) if p.exists() else pd.DataFrame()


def words(text):
    return {w for w in re.findall(r"[a-z_]{4,}", str(text).lower()) if w not in STOP}


def regressions(df, min_models=2, base="L2"):
    c = df[df.correct.notna() & df.tag.fillna("").eq("")].copy() if "tag" in df else df[df.correct.notna()].copy()
    c["correct"] = c.correct.astype(float)
    per = c.groupby(["question_id", "model", "context"]).correct.mean().unstack("context")
    if base not in per or "L3" not in per:
        return []
    per = per.dropna(subset=[base, "L3"])
    hit = per[(per[base] >= 0.5) & (per["L3"] < 0.5)].reset_index()
    out = []
    for qid, g in hit.groupby("question_id"):
        models = sorted(g.model)
        tried = per.xs(qid, level="question_id").shape[0]
        if len(models) >= min_models or (tried == 1 and len(models) == 1):
            out.append({"question_id": qid, "models": models, "of": tried})
    return sorted(out, key=lambda r: (-len(r["models"]), str(r["question_id"])))


def semantic_hits(db, text, top=6):
    p = Path("contexts") / db / "semantic.yaml"
    if not p.exists():
        return []
    want = words(text)
    scored = []
    for line in p.read_text().splitlines():
        s = len(want & words(line))
        if s:
            scored.append((s, line.strip()))
    scored.sort(key=lambda t: -t[0])
    return [l for _, l in scored[:top]]


def l3_report():
    df = _load()
    if df.empty:
        print("No runs yet.")
        return
    qs = {json.loads(l)["question_id"]: json.loads(l) for l in open("artifacts/questions.jsonl")}
    gold = pickle.load(open("artifacts/gold.pkl", "rb")) if Path("artifacts/gold.pkl").exists() else {}
    shapes = load_shapes()
    regs = regressions(df)
    lines = [
        "# Where the semantic model (L3) broke an answer that L2 got right\n",
        "A question is listed when at least two models were right at L2 and wrong at L3. Models failing the same question "
        "the same way points at the L3 text, not at model noise. Read the L3 lines below against the failing SQL by hand.\n",
    ]
    if not regs:
        lines.append("No question was broken by L3 for two or more models yet.")
    for r in regs[:25]:
        q = qs.get(r["question_id"], {})
        lines.append(
            f"## {r['question_id']}  ({q.get('db')}, {q.get('difficulty')})  models that regressed: {', '.join(r['models'])} of {r['of']}\n"
        )
        lines.append(f"Question: {q.get('question')}\n")
        if r["question_id"] in gold:
            lines.append(f"Gold result (first rows): {[list(x) for x in gold[r['question_id']]['rows'][:3]]}\n")
        sub = df[(df.question_id == r["question_id"]) & df.model.isin(r["models"]) & df.context.isin(["L2", "L3"])]
        text = q.get("question", "")
        for _, row in sub.sort_values(["model", "context"]).iterrows():
            sh = row.get("shape") if isinstance(row.get("shape"), str) else shapes.get(row["run_id"], "n/a")
            lines.append(
                f"- {row['model']} {row['context']}  correct={row['correct']}  shape={sh}\n  ```sql\n  {str(row.get('final_sql'))[:400]}\n  ```"
            )
            text += " " + str(row.get("final_sql"))
        hits = semantic_hits(q.get("db", ""), text)
        if hits:
            lines.append("\nLines in this database's semantic model that mention the same words:\n")
            lines += [f"    {h}" for h in hits]
        lines.append("")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/l3_regressions.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))
    print(f"\nwrote docs/l3_regressions.md ({len(regs)} questions)")


def load_shapes(pattern="labels/shape*.jsonl"):
    out = {}
    for p in sorted(Path(".").glob(pattern)):
        for line in open(p):
            try:
                r = json.loads(line)
                out[r["run_id"]] = r["shape"]
            except Exception:
                pass
    return out


def backfill():
    from aae.config import DB_DIR, GOLD_FILE
    from aae.guard import GuardError, open_connection, run_sql
    from aae.score import shape_of

    df = _load()
    have = load_shapes()
    gold = pickle.load(open(GOLD_FILE, "rb"))
    todo = df[df.correct.notna() & df.correct.eq(False) & ~df.run_id.isin(have)]
    if "shape" in df:
        todo = todo[todo["shape"].isna()]
    conns, n, t0 = {}, 0, time.time()
    Path("labels").mkdir(exist_ok=True)
    with open("labels/shape.jsonl", "a") as f:
        for _, r in todo.iterrows():
            if time.time() - t0 > 900:
                print("backfill: 15 minute budget reached. The rest is labelled on a later run.")
                break
            if r["asked"]:
                s = "asked"
            elif not isinstance(r["final_sql"], str) or not r["final_sql"].strip():
                s = "no_result"
            else:
                db = r["db"]
                if db not in conns:
                    p = DB_DIR / f"{db}.duckdb"
                    conns[db] = open_connection(p) if p.exists() else None
                if conns[db] is None:
                    continue
                try:
                    res = run_sql(conns[db], r["final_sql"])
                    s = "truncated" if res["truncated"] else shape_of(res["rows"], gold[r["question_id"]]["rows"])
                except GuardError:
                    s = "sql_error"
            f.write(json.dumps({"run_id": r["run_id"], "shape": s}) + "\n")
            n += 1
    print(f"labelled {n} wrong runs in labels/shape.jsonl ({len(todo) - n} skipped because their database file is not here)")


def effort():
    main = _load()
    for p in sorted(Path("runs").glob("effort_*.jsonl")):
        probe = _load(p)
        if probe.empty:
            continue
        m = probe.model.iloc[0]
        base = main[(main.model == m) & main.question_id.isin(probe.question_id) & main.context.isin(probe.context.unique())]
        both = probe.merge(base, on=["question_id", "context", "model"], suffixes=("_probe", "_default"))
        both = both[both.correct_probe.notna() & both.correct_default.notna()]
        n = len(both)
        if not n:
            print(f"{p.name}: no overlapping default runs")
            continue
        a, b = both.correct_default.astype(float), both.correct_probe.astype(float)
        print(
            f"\n=== {m} at default effort vs {probe.reasoning_setting.iloc[0] if 'reasoning_setting' in probe else '?'} ({p.name}), n={n} same question/context pairs ==="
        )
        print(
            f"accuracy {a.mean() * 100:.1f}% -> {b.mean() * 100:.1f}%   fixed {int(((a == 0) & (b == 1)).sum())}, broke {int(((a == 1) & (b == 0)).sum())}"
        )
        print(
            f"median latency {both.latency_s_default.median():.1f}s -> {both.latency_s_probe.median():.1f}s; "
            f"median output tokens {both.output_tokens_default.median():.0f} -> {both.output_tokens_probe.median():.0f}"
        )
        if "reasoning_tokens_probe" in both:
            print(f"runs with reasoning tokens > 0: {(both.reasoning_tokens_probe.fillna(0) > 0).mean() * 100:.0f}% of probe runs")


if __name__ == "__main__":
    {"l3": l3_report, "backfill": backfill, "effort": effort}[sys.argv[1]]()
