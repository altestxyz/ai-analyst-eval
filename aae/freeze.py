import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

FILE = Path("artifacts/frozen.json")


def h(b):
    return hashlib.sha256(b).hexdigest()[:16]


def compute():
    exp = yaml.safe_load(open("experiment.yaml"))
    models = yaml.safe_load(open("models.yaml"))
    ids = {k: {"provider": v["provider"], "id": v.get("id") or v.get("routes"), "num_ctx": v.get("num_ctx")} for k, v in models.items()}
    ctx = b"".join(p.read_bytes() for p in sorted(Path("contexts").glob("*/L[123].txt")) if not p.parent.name.startswith("scaled_"))
    return {
        "experiment": h(json.dumps(exp, sort_keys=True).encode()),
        "models": h(json.dumps(ids, sort_keys=True).encode()),
        "prompt": h(Path("aae/prompts.py").read_bytes()),
        "questions": h(Path("artifacts/questions.jsonl").read_bytes()),
        "contexts": h(ctx),
    }


def write():
    FILE.write_text(json.dumps({"frozen_at": datetime.now(timezone.utc).isoformat(), "parts": compute()}, indent=2))
    print("frozen:", FILE.read_text())


def check():
    if not FILE.exists():
        print("not frozen yet")
        return 0
    old = json.loads(FILE.read_text())["parts"]
    new = compute()
    changed = [k for k in old if old[k] != new.get(k)]
    if changed:
        print("CHANGED since the freeze:", changed)
        return 1
    print("frozen files unchanged")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "write":
        write()
    else:
        sys.exit(check())
