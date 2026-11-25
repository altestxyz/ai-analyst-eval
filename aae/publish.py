import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RESTORED = Path(tempfile.gettempdir()) / "aae_restored_ids.json"
OUT = Path("shard_out")


def new_records():
    seen = json.load(open(RESTORED)) if RESTORED.exists() else {}
    out = {}
    for p in sorted(Path("runs").glob("*.jsonl")):
        if p.name == "sample.jsonl":
            continue
        have = set(seen.get(p.name, []))
        lines = []
        for line in open(p):
            try:
                rid = json.loads(line)["run_id"]
            except Exception:
                continue
            if rid not in have:
                lines.append(line if line.endswith("\n") else line + "\n")
        if lines:
            out[p.name] = lines
    return out


def tag():
    meta = Path("artifacts/this_shard.json")
    if meta.exists():
        m = json.load(open(meta))
        return (
            f"shard_{m.get('index', 'x'):02d}_{m.get('stamp', '')}"
            if isinstance(m.get("index"), int) and m["index"] >= 0
            else f"shard_x_{m.get('stamp', '')}"
        )
    return "shard_x_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M")


def build():
    recs = new_records()
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "artifacts").mkdir(parents=True)
    (OUT / "runs").mkdir()
    meta = json.load(open("artifacts/this_shard.json")) if Path("artifacts/this_shard.json").exists() else {}
    t = tag()
    meta["new_records"] = {k: len(v) for k, v in recs.items()}
    json.dump(meta, open(OUT / "artifacts" / f"{t}.json", "w"), indent=2)
    for name, lines in recs.items():
        (OUT / "runs" / name).write_text("".join(lines))
    if Path("labels/shape.jsonl").exists():
        (OUT / "labels").mkdir()
        shutil.copy2("labels/shape.jsonl", OUT / "labels" / f"shape_{t}.jsonl")
    print(f"shard_out/ holds {sum(len(v) for v in recs.values())} new records: {meta['new_records']}")
    return t, sum(len(v) for v in recs.values())


def credentials():
    if os.environ.get("KAGGLE_USERNAME") and (os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")):
        return os.environ["KAGGLE_USERNAME"]
    try:
        from kaggle_secrets import UserSecretsClient

        sec = UserSecretsClient()
    except Exception:
        return ""
    got = {}
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        try:
            got[k] = sec.get_secret(k)
        except Exception:
            pass
    if not got.get("KAGGLE_USERNAME") or not (got.get("KAGGLE_KEY") or got.get("KAGGLE_API_TOKEN")):
        return ""
    os.environ.update(got)
    return got["KAGGLE_USERNAME"]


def unpack_zips(folder):
    for z in list(Path(folder).rglob("*.zip")):
        try:
            shutil.unpack_archive(str(z), str(z.with_suffix("")))
            z.unlink()
        except Exception as e:
            print(f"publish: could not unpack {z.name} ({e})")


def _kaggle(args, **kw):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=1800, **kw)


def push():
    dataset = os.environ.get("AAE_SHARDS_DATASET", "aae-shards")
    owner = os.environ.get("AAE_KAGGLE_USER") or credentials()
    if not owner:
        print(
            "publish: no Kaggle API secret found (KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_API_TOKEN). Skipping.\n"
            "  This run's new records are in shard_out/ in the notebook Output. Nothing is lost."
        )
        return 0
    t, n = build()
    if not n:
        print("publish: this session added no records. Nothing to push.")
        return 0
    stage = Path(tempfile.gettempdir()) / "aae_publish"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    slug = f"{owner}/{dataset}"
    r = _kaggle(["datasets", "download", "-d", slug, "-p", str(stage), "--unzip"])
    exists = r.returncode == 0
    if exists:
        unpack_zips(stage)
    print(f"publish: {slug} " + ("exists, adding to it" if exists else f"not found ({(r.stderr or r.stdout).strip()[:120]}), creating it"))
    shutil.copytree(OUT, stage / t)
    json.dump(
        {"title": dataset, "id": slug, "licenses": [{"name": "other"}], "isPrivate": True}, open(stage / "dataset-metadata.json", "w")
    )
    verb = ["datasets", "version", "-m", f"{t}: {n} new records"] if exists else ["datasets", "create"]
    r = _kaggle([*verb, "-p", str(stage), "--dir-mode", "zip"])
    ok = r.returncode == 0
    print("publish:", "done" if ok else "FAILED", (r.stdout + r.stderr).strip()[-300:])
    if not ok:
        print("  shard_out/ in the notebook Output has this run's records. Nothing is lost.")
    return 0


def pull():
    owner = os.environ.get("AAE_KAGGLE_USER") or credentials()
    dataset = os.environ.get("AAE_SHARDS_DATASET", "aae-shards")
    if not owner:
        print("pull: no Kaggle API secret found, so earlier shards are only seen if they are attached as inputs.")
        return 0
    dest = Path(tempfile.gettempdir()) / "aae_pub"
    shutil.rmtree(dest, ignore_errors=True)
    r = _kaggle(["datasets", "download", "-d", f"{owner}/{dataset}", "-p", str(dest), "--unzip"])
    unpack_zips(dest)
    print("pull:", "downloaded" if r.returncode == 0 else f"none yet or not reachable ({(r.stderr or r.stdout).strip()[:120]})")
    return 0


if __name__ == "__main__":
    try:
        {"build": build, "push": push, "pull": pull}[sys.argv[1]]()
    except Exception as e:
        print(f"publish step failed ({type(e).__name__}: {str(e)[:200]}). The records are still in the notebook Output.")
