import math
import numbers

import numpy as np


def _cell_equal(a, b, rel=1e-6):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-12)
    return str(a) == str(b)


def _sort_key(row):
    out = []
    for v in row:
        if v is None:
            out.append((0, ""))
        elif isinstance(v, numbers.Number):
            out.append((1, f"{float(v):.6g}"))
        else:
            out.append((2, str(v)))
    return out


def same_result(pred, gold, ordered):
    if len(pred) != len(gold):
        return False
    if pred and len(pred[0]) != len(gold[0]):
        return False
    if not ordered:
        pred = sorted(pred, key=_sort_key)
        gold = sorted(gold, key=_sort_key)
    return all(len(p) == len(g) and all(_cell_equal(a, b) for a, b in zip(p, g)) for p, g in zip(pred, gold))


def has_top_level_order(sql, dialect="sqlite"):
    import sqlglot

    return sqlglot.parse_one(sql, read=dialect).args.get("order") is not None


def paired_diff_ci(a, b, n_boot=10_000, seed=0):
    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean() * 100, lo * 100, hi * 100


def verdict(d, lo, hi, threshold=5.0):
    if lo > 0 and d >= threshold:
        return "supported"
    if hi < threshold:
        return "not supported"
    return "inconclusive"


def noninferior_verdict(d, lo, hi, margin=5.0):
    if lo > -margin:
        return "within margin"
    if hi < -margin:
        return "behind"
    return "inconclusive"


def _norm_row(row):
    return tuple(_sort_key(row))


def shape_of(pred, gold):
    if not gold and not pred:
        return "match"
    if not pred:
        return "empty"
    if gold and len(pred[0]) != len(gold[0]):
        return "wrong_columns"
    if [_norm_row(r) for r in pred] == [_norm_row(r) for r in gold]:
        return "match"
    p = sorted(_norm_row(r) for r in pred)
    g = sorted(_norm_row(r) for r in gold)
    if p == g:
        return "order_only"
    if len(pred) > len(gold):
        return "too_many_rows"
    if len(pred) < len(gold):
        return "too_few_rows"
    shared = len(set(p) & set(g))
    return "partial_overlap" if shared else "disjoint_values"
