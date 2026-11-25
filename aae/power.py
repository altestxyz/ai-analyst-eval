import sys

import numpy as np


def simulate(n=150, flip=0.20, gain=0.05, reps=400, seed=0):
    rng = np.random.default_rng(seed)
    up, down = (flip + gain) / 2, (flip - gain) / 2
    hits, widths = 0, []
    for _ in range(reps):
        d = rng.choice([1, 0, -1], size=n, p=[up, 1 - flip, down])
        boots = d[rng.integers(0, n, size=(2000, n))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        hits += lo > 0
        widths.append((hi - lo) / 2 * 100)
    return hits / reps, float(np.mean(widths))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    flip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    print(f"n={n} questions, flip rate {flip:.0%}")
    print("true gain   chance the interval excludes zero")
    for gain in (0.03, 0.05, 0.08, 0.10, 0.15):
        if gain > flip:
            continue
        power, half = simulate(n, flip, gain)
        print(f"  {gain * 100:4.0f} pts    {power:5.0%}      (interval about +/-{half:.1f} pts)")
