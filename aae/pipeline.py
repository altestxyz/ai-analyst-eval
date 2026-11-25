import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import yaml

from aae.models import load_models, route

PY = sys.executable
BIRD_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
TLC_BASE = "https://d37ci6vzurychx.cloudfront.net"
LEVELS = ("L1", "L2", "L3")
PARTS = ("bird", "taxi", "ambiguous")
DATA = Path(os.getenv("AAE_DATA", "data"))
ART = Path("artifacts")
TAXI_SOURCE = Path("taxi")
LOCAL_DEFAULT = ("M1", "M2")


def sh(args, check=True):
    cmd = [PY, "-u", *map(str, args)]
    rc = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    if check and rc:
        raise SystemExit(f"command failed (exit {rc}): {' '.join(cmd)}")
    return rc


def mod(name, *args, check=True):
    return sh(["-m", f"aae.{name}", *args], check=check)


def nlines(path):
    p = Path(path)
    return sum(1 for _ in open(p)) if p.exists() else 0


def part_of(q):
    return "ambiguous" if q.get("ambiguous") else ("taxi" if q["db"] == "nyc_taxi" else "bird")


def questions():
    return [json.loads(line) for line in open(ART / "questions.jsonl")]


def deadline_from(hours):
    return int(time.time() + hours * 3600) if hours else 0


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size < 10_000:
        raise SystemExit(f"{dest.name} did not download. The link may have moved: {url}")
    return dest


def find_bird(explicit=""):
    roots = [Path(explicit)] if explicit else []
    if os.getenv("BIRD_DIR"):
        roots.append(Path(os.environ["BIRD_DIR"]))
    roots += [DATA, Path("/kaggle/input")]
    for base in roots:
        if not base.exists():
            continue
        for dp, dn, fn in os.walk(base):
            if len(Path(dp).relative_to(base).parts) > 6:
                dn[:] = []
                continue
            if "mini_dev_sqlite.json" in fn and (Path(dp) / "dev_databases").exists():
                return Path(dp)
    return None


def ensure_bird(explicit=""):
    bird = find_bird(explicit)
    if bird is None:
        zpath = download(BIRD_URL, DATA / "minidev.zip")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATA)
        zpath.unlink()
        bird = find_bird()
    if bird is None:
        raise SystemExit("BIRD Mini-Dev not found. Pass --bird-dir with a folder holding mini_dev_sqlite.json and dev_databases/.")
    os.environ["BIRD_DIR"] = str(bird)
    return bird


def cmd_setup(args):
    bird = ensure_bird(args.bird_dir)
    print("BIRD at", bird)
    mod("build_db")
    if not (ART / "gold.pkl").exists():
        mod("parity")
        mod("select_questions")
    print(nlines(ART / "questions.jsonl"), "questions in artifacts/questions.jsonl")


def cmd_taxi(args):
    source = TAXI_SOURCE / "taxi_questions.jsonl"
    questions_file = ART / "taxi_questions.jsonl"
    if not source.exists() and not questions_file.exists():
        print("Skipping the taxi set: its questions and column descriptions are not in this repository (see README.md).")
        return
    if source.exists():
        ART.mkdir(exist_ok=True)
        shutil.copy(source, questions_file)
        shutil.copytree(TAXI_SOURCE / "database_description", DATA / "taxi" / "database_description", dirs_exist_ok=True)
    files = {f"yellow_tripdata_2025-{m:02d}.parquet": f"{TLC_BASE}/trip-data/yellow_tripdata_2025-{m:02d}.parquet" for m in args.months}
    files["taxi_zone_lookup.csv"] = f"{TLC_BASE}/misc/taxi_zone_lookup.csv"
    for name, url in files.items():
        download(url, DATA / "taxi" / name)
    mod("build_taxi")
    if not (ART / "questions.jsonl").exists():
        raise SystemExit("Run `python -m aae setup` first.")
    mod("add_taxi")


def cmd_contexts(args):
    mod("contexts")
    if args.author != "manual":
        mod("semantic", args.author)
    mod("contexts", "l3")


SMOKE = """
import sys
from aae.llm import chat
from aae.models import load_models
M = load_models()
msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hi in five words."}]
bad = []
for k in sys.argv[1:]:
    try:
        r = chat(M[k], msgs, use_disk_cache=False)
        print(k, M[k]["id"], "->", repr(r["text"][:60]), "tokens in/out:", r["input_tokens"], r["output_tokens"], f"{r['latency_s']}s")
        if not r["text"].strip():
            bad.append(k)
    except Exception as e:
        print(k, "FAILED:", type(e).__name__, str(e)[:300])
        bad.append(k)
sys.exit(1 if bad else 0)
"""


def targets(target, models, use_m2=True):
    cfg = load_models()
    pool = models or list(cfg)
    local = [k for k in pool if cfg.get(k, {}).get("provider") == "ollama" and k not in ("M1L", "JUDGE")]
    front = [k for k in pool if cfg.get(k, {}).get("provider", "ollama") != "ollama"]
    if not use_m2:
        local = [k for k in local if k != "M2"]
    if target == "laptop":
        return local, []
    if target == "frontier":
        return [], front
    return local, front


def cmd_smoke(args):
    local, front = targets(args.target, args.models)
    ids = local + front
    if ids:
        sh(["-c", SMOKE, *ids])


def cmd_freeze(args):
    mod("freeze", args.action, check=args.action == "write")


def check_freeze(allow_changes=False):
    if not (ART / "frozen.json").exists():
        return
    rc = mod("freeze", "check", check=False)
    if rc and not allow_changes:
        raise SystemExit(
            "Something frozen has changed (see above). Undo the change, or pass --allow-changes and record why in your write-up."
        )


def cmd_pilot(args):
    local, front = targets(args.target, args.models)
    if local:
        mod("runner", "--models", ",".join(local), "--limit", 10, "--print-every", 5)
    if front:
        mod("runner", "--models", ",".join(front), "--limit", 10, "--repeats", 1, "--allow-hosted", "--print-every", 5)
        mod("quota", check=False)
    n_clear = sum(1 for q in questions() if not q.get("ambiguous"))
    mod("power", n_clear, "0.20")


def log_size(tag):
    p = Path("runs/logs") / f"{tag}.log"
    return p.stat().st_size if p.exists() else 0


def log_since(tag, offset):
    p = Path("runs/logs") / f"{tag}.log"
    if not p.exists():
        return ""
    with open(p) as f:
        f.seek(offset)
        return f.read()


def rounds(models, batches, repeats, deadline, keep_going=True):
    Path("runs/logs").mkdir(parents=True, exist_ok=True)
    active, stops = list(models), {}
    for i, batch_args in enumerate(batches, 1):
        if deadline and time.time() > deadline:
            print("frontier: time budget reached. Everything so far is saved.", flush=True)
            break
        offsets = {m: log_size(f"main_{m}") for m in active}
        procs = {}
        for m in active:
            log = open(Path("runs/logs") / f"main_{m}.log", "a")
            cmd = [
                PY,
                "-u",
                "-m",
                "aae.runner",
                "--models",
                m,
                "--allow-hosted",
                "--repeats",
                str(repeats),
                "--print-every",
                "10",
                *map(str, batch_args),
            ]
            if deadline:
                cmd += ["--deadline", str(deadline)]
            procs[m] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        for p in procs.values():
            p.wait()
        stopped = {}
        for m, p in procs.items():
            out = log_since(f"main_{m}", offsets[m])
            if p.returncode:
                tail = "\n".join(out.splitlines()[-10:])
                raise SystemExit(f"frontier runner {m} crashed (exit {p.returncode}). End of runs/logs/main_{m}.log:\n{tail}")
            stop = next((ln for ln in out.splitlines() if ln.startswith("STOP:")), None)
            if stop:
                stopped[m] = stop
        print(
            f"frontier round {i}/{len(batches)} done" + (f" (still running: {', '.join(active)})" if len(active) < len(models) else ""),
            flush=True,
        )
        for m, stop in stopped.items():
            print(f"  {m} stopped: {stop[:220]}", flush=True)
        stops.update(stopped)
        if any("time budget" in s for s in stopped.values()):
            break
        if stopped:
            if not keep_going:
                break
            active = [m for m in active if m not in stopped]
            if not active:
                print("frontier: every model has stopped (quota or credential). Rerun after it resets.", flush=True)
                break
    return stops


def main_batches(front, batch, repeats):
    qs = questions()
    have = {}
    rp = Path("runs/runs.jsonl")
    if rp.exists():
        for line in open(rp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("tag") and r.get("model") in front:
                have.setdefault(r["question_id"], set()).add((r["model"], r["context"], r["repeat"]))
    need = {(m, c, r) for m in front for c in LEVELS for r in range(repeats)}
    out = []
    for part in PARTS:
        ids = [q["question_id"] for q in qs if part_of(q) == part]
        k = (sum(1 for i in ids if need <= have.get(i, set())) // batch) * batch
        while k < len(ids):
            k = min(k + batch, len(ids))
            out.append(["--only", part, "--limit", k])
    return out


def cmd_run(args):
    local, front = targets(args.target, args.models, not args.no_m2)
    check_freeze(args.allow_changes)
    deadline = deadline_from(args.hours)
    repeats_front = args.frontier_repeats
    thread, errors = None, []
    if front:
        batches = main_batches(front, args.batch, repeats_front)

        def work():
            try:
                rounds(front, batches, repeats_front, deadline)
            except SystemExit as e:
                errors.append(e)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
    if local:
        extra = ["--deadline", deadline] if deadline else []
        mod("runner", "--models", ",".join(local), "--print-every", 25, *extra, check=False)
    if thread:
        thread.join()
        if errors:
            raise errors[0]
    print("runs in the log:", nlines("runs/runs.jsonl"))


def cmd_shard(args):
    from aae import shard as S

    check_freeze(args.allow_changes)
    cfg = load_models()
    front = [k for k, v in cfg.items() if v["provider"] != "ollama"]
    qs = questions()
    by_id = {q["question_id"]: q for q in qs}
    exp = yaml.safe_load(open("experiment.yaml"))
    cutoff = args.cutoff or (args.schedule_start + "T00:00:00+00:00")
    base = S.baseline_ids("runs/runs.jsonl", cutoff)
    addons = (["cache"] if not args.no_cache else []) + ([] if args.no_effort else ["effort"])
    plan = S.make_plan(qs, base, front, args.frontier_repeats, args.slice_questions, exp["seed"], PARTS, addons)
    done = S.done_ids("runs/runs.jsonl")
    print(S.describe(plan, done, by_id))
    addon_done = {
        "cache": nlines("runs/cache_warm.jsonl") >= 43 and nlines("runs/cache_cold.jsonl") >= 43,
        "effort": nlines(f"runs/effort_{args.effort_level}_{args.effort_model}.jsonl") >= 27,
    }
    choice = S.pick(plan, done, by_id, args.slice, args.schedule_start, have_shards=any(ART.glob("shard_*.json")), addon_done=addon_done)
    print(f"\nTHIS RUN: {choice['kind']} slice {choice['index']}: {choice['why']}")
    S.check_plans(str(ART))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M")
    info = {
        "index": choice["index"],
        "kind": choice["kind"],
        "stamp": stamp,
        "plan_hash": plan["hash"],
        "why": choice["why"],
        "shard_setting": str(args.slice),
        "schedule_start": args.schedule_start,
        "questions": len(choice["qids"]),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    deadline = deadline_from(args.hours)
    if choice["kind"] == "main":
        dbs = sorted({by_id[i]["db"] for i in choice["qids"]})
        prepare_databases(dbs, args.bird_dir)
        batches = []
        for i in range(0, len(choice["qids"]), args.batch):
            chunk = choice["qids"][i : i + args.batch]
            f = ART / f"slice_batch_{i}.txt"
            f.write_text(",".join(str(c) for c in chunk))
            batches.append(["--qids", f"@{f}"])
        info["stopped"] = rounds(front, batches, args.frontier_repeats, deadline)
        info["missing_after"] = S.slice_missing(plan, choice["index"], S.done_ids("runs/runs.jsonl"), by_id)
        print("jobs still missing after this run, by model:", info["missing_after"])
    elif choice["kind"] == "cache":
        prepare_databases([d for d in exp["databases"]], args.bird_dir)
        common = [
            "--no-disk-cache",
            "--order-by-db",
            "--per-db",
            6,
            "--levels",
            "L2,L3",
            "--repeats",
            1,
            "--models",
            args.big,
            "--allow-hosted",
            "--print-every",
            10,
        ]
        mod("runner", "--tag", "cache_cold", "--nonce", *common, check=False)
        mod("runner", "--tag", "cache_warm", *common, check=False)
        mod("infra_report", "caching", check=False)
    elif choice["kind"] == "effort":
        probe = [q["question_id"] for q in S.ordered_questions(qs, exp["seed"], PARTS) if S.part_of(q) == "bird"][:30]
        prepare_databases(sorted({by_id[i]["db"] for i in probe}), args.bird_dir)
        mod(
            "runner",
            "--tag",
            f"effort_{args.effort_level}_{args.effort_model}",
            "--reasoning",
            args.effort_level,
            "--models",
            args.effort_model,
            "--allow-hosted",
            "--qids",
            ",".join(str(x) for x in probe),
            "--levels",
            "L2",
            "--repeats",
            1,
            "--no-disk-cache",
            "--print-every",
            10,
            check=False,
        )
        mod("diagnose", "effort", check=False)
    info["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ART.mkdir(exist_ok=True)
    json.dump(info, open(ART / "this_shard.json", "w"), indent=2, default=str)
    print("runs in the log now:", nlines("runs/runs.jsonl"))


def prepare_databases(dbs, bird_dir=""):
    bird_dbs = [d for d in dbs if d != "nyc_taxi"]
    todo = [d for d in bird_dbs if not (DATA / "duckdb" / f"{d}.duckdb").exists()]
    if todo:
        ensure_bird(bird_dir)
        mod("build_db", *todo)
    if "nyc_taxi" in dbs and not (DATA / "duckdb" / "nyc_taxi.duckdb").exists():
        for m in (1, 2, 3):
            download(f"{TLC_BASE}/trip-data/yellow_tripdata_2025-{m:02d}.parquet", DATA / "taxi" / f"yellow_tripdata_2025-{m:02d}.parquet")
        download(f"{TLC_BASE}/misc/taxi_zone_lookup.csv", DATA / "taxi" / "taxi_zone_lookup.csv")
        mod("build_taxi")


def cmd_report(args):
    if nlines("runs/runs.jsonl"):
        mod("report")
        mod("charts", check=False)
        mod("laptop_vs_frontier", check=False)
    if Path(args.table).exists():
        mod("figures", "--table", args.table)
        mod("headline", "--table", args.table, check=False)


def cmd_doctor(args):
    problems = []
    need = ["yaml", "numpy", "pandas", "matplotlib", "duckdb", "sqlglot", "httpx"]
    local, front = targets(args.target, args.models)
    if front:
        need.append("openai")
    for name in need:
        try:
            __import__(name)
            print(f"ok      python package {name}")
        except ImportError:
            problems.append(f"missing python package {name}")
            print(f"MISSING python package {name}")
    cfg = load_models()
    if local:
        host = os.getenv("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
        try:
            tags = json.load(urllib.request.urlopen(f"{host}/api/tags", timeout=3))
            have = {m["name"] for m in tags.get("models", [])}
            for k in local:
                want = cfg[k]["id"]
                if want in have:
                    print(f"ok      ollama model {want}")
                else:
                    problems.append(f"ollama model {want} not pulled")
                    print(f"MISSING ollama model {want}   (ollama pull {want})")
        except Exception as e:
            problems.append(f"ollama not reachable at {host}")
            print(f"MISSING ollama at {host} ({type(e).__name__})")
    if front:
        r = route()
        print(f"route   {r}")
        for k in front:
            if r == "kproxy":
                ok = bool(os.getenv("MODEL_PROXY_URL") and os.getenv("MODEL_PROXY_API_KEY"))
                name = "MODEL_PROXY_URL / MODEL_PROXY_API_KEY"
            else:
                name = cfg[k].get("api_key_env", "")
                ok = bool(name and os.getenv(name))
            print(f"{'ok     ' if ok else 'MISSING'} {k} credential {name} {'is set' if ok else 'is not set'}")
            if not ok:
                problems.append(f"{k} credential not set")
        if args.live and not problems:
            sh(["-c", SMOKE, *front], check=False)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        return 1
    print("\nready")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m aae")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name, **kw)
        p.set_defaults(fn=fn)
        return p

    def common(p):
        p.add_argument("--target", choices=["laptop", "frontier", "all"], default="all")
        p.add_argument("--models", default="", type=lambda s: [x for x in s.split(",") if x])

    p = add("setup", cmd_setup, help="BIRD data, DuckDB copies, answer-key parity, question set")
    p.add_argument("--bird-dir", default="")
    p = add("taxi", cmd_taxi, help="NYC taxi held-out set and ambiguous questions")
    p.add_argument("--months", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3])
    p = add("contexts", cmd_contexts, help="L1, L2 and L3 context files")
    p.add_argument("--author", default="F1")
    p = add("smoke", cmd_smoke, help="one request to each selected model")
    common(p)
    p = add("freeze", cmd_freeze, help="write or check the locked plan")
    p.add_argument("action", choices=["write", "check"])
    p = add("pilot", cmd_pilot, help="ten questions on every selected model")
    common(p)
    p = add("run", cmd_run, help="the main grid on laptop-class models, frontier models, or both")
    common(p)
    p.add_argument("--hours", type=float, default=0)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--frontier-repeats", type=int, default=1)
    p.add_argument("--no-m2", action="store_true")
    p.add_argument("--allow-changes", action="store_true")
    p = add("shard", cmd_shard, help="one independent slice of the remaining frontier questions")
    p.add_argument("--schedule-start", required=True)
    p.add_argument("--cutoff", default="")
    p.add_argument("--slice", default="auto")
    p.add_argument("--slice-questions", type=int, default=40)
    p.add_argument("--hours", type=float, default=0)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--frontier-repeats", type=int, default=1)
    p.add_argument("--big", default="F1")
    p.add_argument("--effort-model", default="F3")
    p.add_argument("--effort-level", default="medium")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-effort", action="store_true")
    p.add_argument("--bird-dir", default="")
    p.add_argument("--allow-changes", action="store_true")
    p = add("report", cmd_report, help="rebuild charts and tables")
    p.add_argument("--table", default="results/runs_table.csv")
    p = add("doctor", cmd_doctor, help="check packages, Ollama and credentials")
    common(p)
    p.add_argument("--live", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
