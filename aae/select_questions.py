import json
import random

from aae.config import DBS, EXP, KEPT_FILE, QUESTIONS_FILE


def main():
    kept = json.load(open(KEPT_FILE))
    rng = random.Random(EXP["seed"])
    quotas, per_db = EXP["quotas"], EXP["questions_per_db"]
    chosen = []
    for db in DBS:
        pool = {t: [q for q in kept if q["db"] == db and q["difficulty"] == t] for t in quotas}
        for t in pool:
            rng.shuffle(pool[t])
        picked = []
        for t, n in quotas.items():
            picked += pool[t][:n]
            pool[t] = pool[t][n:]
        leftovers = [q for t in ("moderate", "challenging", "simple") for q in pool.get(t, [])]
        picked += leftovers[: max(0, per_db - len(picked))]
        chosen += picked[:per_db]
    with open(QUESTIONS_FILE, "w") as f:
        for q in chosen:
            f.write(json.dumps(q) + "\n")
    print(len(chosen), "questions written to", QUESTIONS_FILE)


if __name__ == "__main__":
    main()
