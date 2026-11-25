import argparse
import fcntl
import hashlib
import json
import os
import pickle
import random
import re
import subprocess
import time
from datetime import datetime, timezone

from aae.config import CTX_DIR, DB_DIR, EXP, GOLD_FILE, MAX_ATTEMPTS, QUESTIONS_FILE, RUNS_DIR, RUNS_FILE
from aae.guard import GuardError, open_connection, run_sql, suspicious
from aae.llm import QuotaExhausted, chat
from aae.models import load_models
from aae.prompts import RETRY_RULE, SYSTEM
from aae.score import same_result, shape_of

MODELS = load_models()
LEVELS = ["L1", "L2", "L3"]
SQL_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
ASK_RE = re.compile(r"^\s*ASK:\s*(.+)", re.M)

DEFAULT_OPT = {"nudge": True, "verify": False, "cache": False, "nonce": False, "disk_cache": True, "tag": ""}


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_reply(text):
    code = SQL_RE.search(text)
    if code:
        return "sql", code.group(1).strip()
    ask = ASK_RE.search(text)
    if ask:
        return "ask", ask.group(1).strip()
    return "none", text


def cost_of(cfg, fresh, cache_read, cache_write, out):
    p_in = cfg["price_in"]
    billed = (
        fresh * p_in
        + cache_read * cfg.get("price_cache_read", p_in)
        + cache_write * cfg.get("price_cache_write", p_in)
        + out * cfg["price_out"]
    ) / 1e6
    listed = ((fresh + cache_read + cache_write) * p_in + out * cfg["price_out"]) / 1e6
    return billed, listed


def run_one(q, level, mkey, repeat, gold, conns, opt=None):
    opt = {**DEFAULT_OPT, **(opt or {})}
    cfg = MODELS[mkey]
    context = (CTX_DIR / q["db"] / f"{level}.txt").read_text()
    rule = RETRY_RULE["verify" if opt["verify"] else "nudge" if opt["nudge"] else "plain"]
    messages = [
        {"role": "system", "content": SYSTEM.format(context=context, max_attempts=MAX_ATTEMPTS, retry_rule=rule)},
        {"role": "user", "content": q["question"]},
    ]
    run_id = f"{q['question_id']}|{level}|{mkey}|{repeat}" + (f"|{opt['tag']}" if opt["tag"] else "")
    nonce = hashlib.md5(run_id.encode()).hexdigest()[:8] if opt["nonce"] else ""

    attempts = []
    fresh = cache_read = cache_write = out_tok = max_prompt = reasoning_tok = 0
    model_s = 0.0
    limit_retries = 0
    final_sql, result = None, None
    asked = had_error = had_suspicious = verify_changed = format_fail = False
    verifying = False
    for turn in range(MAX_ATTEMPTS):
        reply = chat(cfg, messages, repeat, cache=opt["cache"], nonce=nonce, use_disk_cache=opt["disk_cache"])
        fresh += reply["input_tokens"]
        cache_read += reply.get("cache_read_tokens", 0)
        cache_write += reply.get("cache_write_tokens", 0)
        out_tok += reply["output_tokens"]
        reasoning_tok += reply.get("reasoning_tokens", 0)
        max_prompt = max(max_prompt, reply["input_tokens"] + reply.get("cache_read_tokens", 0) + reply.get("cache_write_tokens", 0))
        model_s += reply.get("latency_s", 0.0)
        limit_retries += int(bool(reply.get("limit_retry")))
        messages.append({"role": "assistant", "content": reply["text"]})

        was_verifying, verifying = verifying, False
        if was_verifying and reply["text"].strip().upper().startswith("OK"):
            break
        kind, payload = parse_reply(reply["text"])
        if kind == "ask":
            asked = True
            break
        if kind == "none":
            format_fail = True
            break
        if was_verifying and payload != final_sql:
            verify_changed = True

        confirmed = result is not None and payload == final_sql
        t1 = time.time()
        try:
            res = run_sql(conns[q["db"]], payload)
        except GuardError as e:
            had_error = True
            attempts.append({"sql": payload, "error": str(e), "ms": int((time.time() - t1) * 1000)})
            if result is None:
                final_sql = payload
            messages.append({"role": "user", "content": str(e)})
            continue
        attempts.append(
            {"sql": payload, "error": None, "rows": len(res["rows"]), "truncated": res["truncated"], "ms": int((time.time() - t1) * 1000)}
        )
        final_sql = payload
        if res["rows"]:
            result = res
        why = suspicious(res, nudge=opt["nudge"])
        if why and not confirmed:
            had_suspicious = True
            attempts[-1]["suspicious"] = why
            messages.append({"role": "user", "content": why})
            continue
        if opt["verify"] and not was_verifying and not why and turn < MAX_ATTEMPTS - 1:
            preview = str(res["rows"][:5])[:500]
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The query returned {len(res['rows'])} rows. First rows: {preview}\n"
                        "Does this result answer the question? Reply with exactly OK if it does. "
                        "If it doesn't, reply with one corrected query in a ```sql block."
                    ),
                }
            )
            verifying = True
            continue
        break

    shape = None
    if q.get("ambiguous"):
        correct = None
    elif asked or result is None or result["truncated"]:
        correct = False
        shape = "asked" if asked else "no_result" if result is None else "truncated"
    else:
        g = gold[q["question_id"]]
        correct = same_result(result["rows"], g["rows"], g["ordered"])
        shape = "match" if correct else shape_of(result["rows"], g["rows"])

    billed, listed = cost_of(cfg, fresh, cache_read, cache_write, out_tok)
    num_ctx = cfg.get("num_ctx", 16384)
    return {
        "run_id": run_id,
        "question_id": q["question_id"],
        "db": q["db"],
        "difficulty": q.get("difficulty"),
        "context": level,
        "model": mkey,
        "model_id": cfg["id"],
        "repeat": repeat,
        "tag": opt["tag"],
        "attempts": attempts,
        "final_sql": final_sql,
        "asked": asked,
        "correct": correct,
        "had_error": had_error,
        "had_suspicious": had_suspicious,
        "verify_changed": verify_changed,
        "format_fail": format_fail,
        "n_turns": len([m for m in messages if m["role"] == "assistant"]),
        "limit_retries": limit_retries,
        "result_preview": [list(r) for r in result["rows"][:5]] if result else None,
        "input_tokens": fresh,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": out_tok,
        "max_prompt_tokens": max_prompt,
        "reasoning_tokens": reasoning_tok,
        "shape": shape,
        "reasoning_setting": cfg.get("reasoning"),
        "max_output_tokens": cfg.get("max_output_tokens", 4096) if cfg["provider"] != "ollama" else None,
        "likely_truncated": cfg["provider"] == "ollama" and max_prompt >= 0.98 * num_ctx,
        "cost_usd": billed,
        "cost_list_usd": listed,
        "latency_s": round(model_s + sum(a["ms"] for a in attempts) / 1000, 2),
        "opt": {k: v for k, v in opt.items() if k != "tag"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
    }


def total_spend():
    total = 0.0
    for p in RUNS_DIR.glob("*.jsonl"):
        if p.name == "sample.jsonl":
            continue
        for line in open(p):
            try:
                total += json.loads(line)["cost_usd"]
            except Exception:
                pass
    return total


def done_ids(path):
    ids = set()
    if path.exists():
        for l in open(path):
            try:
                ids.add(json.loads(l)["run_id"])
            except Exception:
                pass
    return ids


def append_record(path, rec):
    line = json.dumps(rec, default=str) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(k for k in MODELS if k not in ("M1L", "JUDGE")))
    ap.add_argument("--levels", default="L1,L2,L3")
    ap.add_argument("--questions", default=str(QUESTIONS_FILE))
    ap.add_argument("--only", default="", help="comma list of question parts to run: bird, taxi, ambiguous")
    ap.add_argument("--limit", type=int, default=0, help="first N questions overall (after --only), for the pilot and the frontier rounds")
    ap.add_argument("--qids", default="", help="comma list of question ids, or @file holding one. Used by the shard slices")
    ap.add_argument("--reasoning", default="", help="override the reasoning setting of these models (needs --tag): the effort probe")
    ap.add_argument("--per-db", type=int, default=0, help="N clear questions per database")
    ap.add_argument("--repeats", type=int, default=0, help="override experiment.yaml")
    ap.add_argument("--tag", default="", help="write to runs/<tag>.jsonl instead of runs.jsonl")
    ap.add_argument("--no-nudge", action="store_true", help="only retry on errors and empty results")
    ap.add_argument("--verify", action="store_true", help="ask the model to check its first result")
    ap.add_argument("--cache", action="store_true", help="accepted but ignored: the proxy gives no cache controls")
    ap.add_argument("--nonce", action="store_true", help="defeat vendor prompt caching")
    ap.add_argument("--no-disk-cache", action="store_true", help="always call the model")
    ap.add_argument("--order-by-db", action="store_true", help="group runs so prefixes repeat")
    ap.add_argument("--allow-hosted", action="store_true", help="required to call hosted frontier models, which spend quota or money")
    ap.add_argument("--deadline", type=float, default=0, help="unix time after which to stop cleanly")
    ap.add_argument("--print-every", type=int, default=1, help="print one progress line per N new runs")
    args = ap.parse_args()

    keys = args.models.split(",")
    if args.reasoning:
        if not args.tag:
            raise SystemExit("--reasoning needs --tag, so the probe runs are logged apart from the main grid.")
        for k in keys:
            MODELS[k] = {**MODELS[k], "reasoning": args.reasoning}
    levels = args.levels.split(",")
    hosted = [k for k in keys if MODELS[k]["provider"] != "ollama"]
    if hosted and not args.allow_hosted:
        raise SystemExit(
            f"{hosted} call hosted models and spend quota or money. Re-run with --allow-hosted once you have checked your limits."
        )

    questions = [json.loads(line) for line in open(args.questions)]
    random.Random(EXP["seed"]).shuffle(questions)
    if args.only:
        want = set(args.only.split(","))
        questions = [q for q in questions if ("ambiguous" if q.get("ambiguous") else "taxi" if q["db"] == "nyc_taxi" else "bird") in want]
    if args.qids:
        raw = open(args.qids[1:]).read() if args.qids.startswith("@") else args.qids
        want_ids = {x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()}
        questions = [q for q in questions if str(q["question_id"]) in want_ids]
    if args.per_db:
        seen, picked = {}, []
        for q in questions:
            if q.get("ambiguous"):
                continue
            if seen.get(q["db"], 0) < args.per_db:
                seen[q["db"]] = seen.get(q["db"], 0) + 1
                picked.append(q)
        questions = picked
    if args.limit:
        questions = questions[: args.limit]

    gold = pickle.load(open(GOLD_FILE, "rb"))
    conns = {db: open_connection(DB_DIR / f"{db}.duckdb") for db in {q["db"] for q in questions}}
    RUNS_DIR.mkdir(exist_ok=True)
    log = RUNS_DIR / f"{args.tag}.jsonl" if args.tag else RUNS_FILE
    done, spend = done_ids(log), total_spend()
    cap = float(os.getenv("SPEND_CAP_USD", "300"))
    opt = {
        "nudge": not args.no_nudge,
        "verify": args.verify,
        "cache": args.cache,
        "nonce": args.nonce,
        "disk_cache": not args.no_disk_cache,
        "tag": args.tag,
    }

    repeats = args.repeats or EXP["repeats"]
    order = {m: i for i, m in enumerate(keys)}
    jobs = [(q, lvl, m, r) for q in questions for lvl in levels for m in keys for r in range(repeats)]
    if args.order_by_db:
        jobs.sort(key=lambda j: (j[0]["db"], j[1], order[j[2]], j[3]))
    else:
        jobs.sort(key=lambda j: order[j[2]])
    todo = sum(1 for q, lvl, m, r in jobs if f"{q['question_id']}|{lvl}|{m}|{r}" + (f"|{args.tag}" if args.tag else "") not in done)
    print(f"{len(jobs)} jobs, {len(jobs) - todo} already done, {todo} to run  models={keys}", flush=True)

    n_new = fails = 0
    session_cost = 0.0
    t_start = time.time()
    for i, (q, lvl, m, r) in enumerate(jobs, 1):
        rid = f"{q['question_id']}|{lvl}|{m}|{r}" + (f"|{args.tag}" if args.tag else "")
        if rid in done:
            continue
        if args.deadline and time.time() > args.deadline:
            print("STOP: time budget reached. Everything so far is saved. Rerun to continue.", flush=True)
            break
        if spend >= cap:
            print(f"STOP: spend cap ${cap} reached. Raise SPEND_CAP_USD to continue.", flush=True)
            break
        try:
            rec = run_one(q, lvl, m, r, gold, conns, opt)
        except QuotaExhausted as e:
            print(
                f"STOP: {type(e).__name__}: {e}\nEverything so far is saved. Rerun the same "
                "command after the quota resets or the credential is refreshed.",
                flush=True,
            )
            break
        except Exception as e:
            fails += 1
            with open(RUNS_DIR / "errors.log", "a") as ef:
                ef.write(f"{rid}\t{type(e).__name__}: {str(e)[:300]}\n")
            print(f"  ! {rid} failed ({type(e).__name__}: {str(e)[:120]}). It will be retried on the next run.", flush=True)
            if fails >= 8:
                print("STOP: 8 failures in a row. Check the model server or your key.", flush=True)
                break
            continue
        fails = 0
        spend += rec["cost_usd"]
        session_cost += rec["cost_usd"]
        append_record(log, rec)
        n_new += 1
        if n_new % args.print_every == 0:
            rate = (time.time() - t_start) / n_new
            print(
                f"[{i}/{len(jobs)}] {m} {lvl} q={q['question_id']} correct={rec['correct']} {rate:.1f}s/run equiv=${spend:.2f}", flush=True
            )
    print(
        f"finished: {n_new} new runs in {(time.time() - t_start) / 60:.1f} min, equivalent list-price cost of this call ${session_cost:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
