import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from aae.models import load_models
from aae.score import paired_diff_ci

LEVELS = ("L1", "L2", "L3")
FLAGS = ("correct_logged", "strict", "lenient", "asked", "had_error", "had_suspicious", "format_fail", "verify_changed")


def half_up(value, places=0):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    q = Decimal(1).scaleb(-places)
    return float(Decimal(str(round(float(value), 9))).quantize(q, rounding=ROUND_HALF_UP))


def load_table(path):
    df = pd.read_csv(path, dtype={"question_id": str}, keep_default_na=False)
    for c in FLAGS:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    for c in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "max_prompt_tokens"):
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    df["repeat"] = pd.to_numeric(df["repeat"], errors="coerce")
    return df


def split_models(models):
    local = [k for k, v in models.items() if v.get("provider") == "ollama" and k not in ("M1L", "JUDGE")]
    front = [k for k, v in models.items() if v.get("provider") != "ollama"]
    return local, front


def main_grid(df):
    return df[df["tag"].fillna("") == ""]


def per_question(df, scorer):
    d = df[df[scorer].notna()]
    return d.groupby(["model", "context", "question_id"])[scorer].mean().reset_index()


def shared_questions(pq, models):
    sets = []
    for m in models:
        part = pq[pq.model == m]
        have = part.groupby("question_id").context.nunique()
        sets.append(set(have[have == len(LEVELS)].index))
    return set.intersection(*sets) if sets else set()


def accuracy_table(df, scorer, local, front, part="bird", extra=None):
    d = main_grid(df)
    d = d[d.part == part]
    if extra is not None:
        d = d[extra(d)]
    pq = per_question(d, scorer)
    out, n = {}, {}
    shared = shared_questions(pq, front) if front else set()
    for m in local + front:
        p = pq[pq.model == m]
        if m in front and part != "taxi" and shared:
            p = p[p.question_id.isin(shared)]
        for c in LEVELS:
            vals = p[p.context == c][scorer]
            out[(m, c)] = 100 * vals.mean() if len(vals) else np.nan
            n[(m, c)] = len(vals)
    return out, n


def list_cost(df, prices):
    pin = df.model.map(lambda m: float(prices.get(m, {}).get("price_in") or 0))
    pout = df.model.map(lambda m: float(prices.get(m, {}).get("price_out") or 0))
    toks = df.input_tokens.fillna(0) + df.cache_read_tokens.fillna(0) + df.cache_write_tokens.fillna(0)
    return (toks * pin + df.output_tokens.fillna(0) * pout) / 1e6


def compute_metrics(df, models=None, rescore_summary=None):
    models = models or load_models()
    local, front = split_models(models)
    present = set(df.model.unique())
    local = [m for m in local if m in present]
    front = [m for m in front if m in present]
    m = {}
    grid = main_grid(df)
    m["runs.total"] = len(df)
    m["runs.frontier"] = int(df.model.isin(front).sum())
    m["runs.local"] = int(df.model.isin(local).sum())
    if rescore_summary:
        m["rescore.agreement_pct"] = rescore_summary.get("strict_agreement_pct")
    scorers = ("strict", "lenient", "correct_logged")
    tables = {s: accuracy_table(df, s, local, front)[0] for s in scorers}
    for s in scorers:
        for (mod, c), v in tables[s].items():
            m[f"acc.{mod}.{c}.{s}"] = v
        for mod in local + front:
            for a, b in (("L1", "L2"), ("L2", "L3")):
                m[f"gain.{mod}.{b}-{a}.{s}"] = tables[s][(mod, b)] - tables[s][(mod, a)]
    for mod in local + front:
        diffs = [tables["lenient"][(mod, c)] - tables["strict"][(mod, c)] for c in LEVELS]
        m[f"lenient_minus_strict.{mod}.min"] = min(diffs)
        m[f"lenient_minus_strict.{mod}.max"] = max(diffs)
    others = [x for x in local + front if x not in ("F1", "F3")]
    allv = [tables["lenient"][(x, c)] - tables["strict"][(x, c)] for x in others for c in LEVELS]
    if allv:
        m["lenient_minus_strict.others.min"] = min(allv)
        m["lenient_minus_strict.others.max"] = max(allv)
    for c in ("L1", "L2"):
        for mod in ("M1",):
            if mod in present:
                t, _ = accuracy_table(df, "strict", local, front, extra=lambda d: d.difficulty == "challenging")
                m[f"acc_challenging.{mod}.{c}.strict"] = t[(mod, c)]
    t_taxi, _ = accuracy_table(df, "strict", local, front, part="taxi")
    for mod in local + front:
        m[f"acc_taxi.{mod}.L1.strict"] = t_taxi[(mod, "L1")]
        m[f"gain_taxi.{mod}.L2-L1.strict"] = t_taxi[(mod, "L2")] - t_taxi[(mod, "L1")]
    if front:
        bird = grid[grid.part == "bird"]
        pq = per_question(bird, "strict")
        shared = shared_questions(pq, front)
        for mod in local:
            for c in LEVELS:
                vals = pq[(pq.model == mod) & (pq.context == c) & pq.question_id.isin(shared)].strict
                m[f"acc_shared.{mod}.{c}.strict"] = 100 * vals.mean() if len(vals) else np.nan
        for mod in local:
            for fm in front:
                m[f"gap_shared.{fm}.L1-{mod}.L2.strict"] = tables["strict"][(fm, "L1")] - m[f"acc_shared.{mod}.L2.strict"]
    clear = grid[(grid.part.isin(["bird", "taxi"])) & grid.correct_logged.notna()].copy()
    fr = clear[clear.model.isin(front)].copy()
    fr["list_usd"] = list_cost(fr, models)
    m["list_cost_usd.frontier"] = float(list_cost(df[df.model.isin(front)], models).sum())
    for mod in front:
        for c in LEVELS:
            x = fr[(fr.model == mod) & (fr.context == c)]
            right = x.correct_logged.sum()
            m[f"cents_per_correct.{mod}.{c}"] = 100 * x.list_usd.sum() / right if right else np.nan
            allx = df[(df.model == mod) & (df.context == c)]
            m[f"prompt_tokens_median.{mod}.{c}"] = float(
                (allx.input_tokens.fillna(0) + allx.cache_read_tokens.fillna(0) + allx.cache_write_tokens.fillna(0)).median()
            )
        for s in ("strict", "lenient"):
            a, b = [fr[(fr.model == mod) & (fr.context == c)] for c in ("L1", "L2")]
            ra, rb = a[s].sum(), b[s].sum()
            m[f"extra_correct.{mod}.{s}"] = float(rb - ra)
            m[f"usd_per_extra_correct.{mod}.{s}"] = float((b.list_usd.sum() - a.list_usd.sum()) / (rb - ra)) if rb != ra else np.nan
    nud = df[df.had_suspicious == 1]
    m["nudged.runs"] = len(nud)
    m["nudged.correct"] = int((nud.correct_logged == 1).sum())
    sc = grid[grid.correct_logged.notna()]
    sw = (sc.correct_logged == 0) & (sc.had_error == 0) & (sc.had_suspicious == 0) & (sc.asked == 0) & (sc.format_fail == 0)
    rate = sw.groupby([sc.model, sc.context]).mean() * 100
    for (mod, c), v in rate.items():
        m[f"silent_wrong_pct.{mod}.{c}"] = float(v)
    if len(rate):
        m["silent_wrong_pct.min"] = float(rate.min())
        m["silent_wrong_pct.max"] = float(rate.max())
    lat = pd.to_numeric(df["latency_s"].replace("", np.nan), errors="coerce")
    m["latency.max_s"] = float(lat.max())
    for mod in front:
        for c in LEVELS:
            x = lat[(df.model == mod) & (df.context == c)].dropna()
            m[f"latency_median.{mod}.{c}"] = float(x.median())
            m[f"latency_p95.{mod}.{c}"] = float(x.quantile(0.95))
    if front:
        bird_front = grid[(grid.part == "bird") & grid.model.isin(front)]
        shared_front = shared_questions(per_question(bird_front, "strict"), front)
        for s in ("strict", "lenient"):
            pq = per_question(bird_front, s)
            worst = pq[pq.question_id.isin(shared_front)].groupby("question_id")[s].max()
            m[f"all_wrong_shared.{s}"] = int((worst == 0).sum())
        sorted_wrong = bird_front[(bird_front.strict == 0) & (bird_front["shape"] != "") & (bird_front["shape"] != "match")]
        m["wrong_by_shape.total"] = len(sorted_wrong)
        m["wrong_by_shape.wrong_columns"] = int((sorted_wrong["shape"] == "wrong_columns").sum())
        missed = bird_front[bird_front.strict == 0]
        fixed = int((missed.lenient == 1).sum())
        m["frontier_misses.strict"] = len(missed)
        m["frontier_misses.fixed_by_lenient"] = fixed
        m["frontier_misses.fixed_by_lenient_pct"] = 100 * fixed / len(missed) if len(missed) else np.nan
        pq_strict = per_question(bird_front, "strict")
        for mod in front:
            p = pq_strict[(pq_strict.model == mod) & pq_strict.question_id.isin(shared_front)]
            wide = p.pivot(index="question_id", columns="context", values="strict").sort_index()
            for a, b in (("L1", "L2"), ("L2", "L3")):
                if a in wide and b in wide:
                    _, low, high = paired_diff_ci(wide[a].values, wide[b].values)
                    m[f"gain_ci.{mod}.{b}-{a}.strict.low"] = low
                    m[f"gain_ci.{mod}.{b}-{a}.strict.high"] = high
        pq_lenient = per_question(grid[grid.part == "bird"], "lenient")
        for mod in local:
            vals = pq_lenient[(pq_lenient.model == mod) & (pq_lenient.context == "L2") & pq_lenient.question_id.isin(shared_front)]
            for fm in front:
                gap = tables["lenient"][(fm, "L1")] - 100 * vals.lenient.mean() if len(vals) else np.nan
                m[f"gap_shared.{fm}.L1-{mod}.L2.lenient"] = gap
    for name, group in (("local", local), ("frontier", front)):
        vague = grid[(grid.part == "ambiguous") & grid.model.isin(group) & grid.asked.notna()]
        if len(vague):
            cells = vague.groupby(["model", "context"]).asked
            m[f"asked_pct.{name}.min"] = float(100 * cells.mean().min())
            m[f"asked_pct.{name}.max"] = float(100 * cells.mean().max())
            m[f"asked_n.{name}.min"] = float(cells.count().min())
            m[f"asked_n.{name}.max"] = float(cells.count().max())
    taxi = grid[(grid.part == "taxi") & grid.model.isin(front) & grid.strict.notna()]
    if len(taxi):
        cells = taxi.groupby(["context", "model"]).strict
        for c in LEVELS:
            if c in taxi.context.values:
                m[f"taxi_frontier.{c}.acc_min"] = float(100 * cells.mean()[c].min())
                m[f"taxi_frontier.{c}.acc_max"] = float(100 * cells.mean()[c].max())
                m[f"taxi_frontier.{c}.n_min"] = float(cells.count()[c].min())
                m[f"taxi_frontier.{c}.n_max"] = float(cells.count()[c].max())
    return {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in m.items()}


def print_headline(df, models=None):
    models = models or load_models()
    local, front = split_models(models)
    present = set(df.model.unique())
    local = [x for x in local if x in present]
    front = [x for x in front if x in present]
    for scorer in ("strict", "lenient"):
        t, n = accuracy_table(df, scorer, local, front)
        print(f"\nBIRD accuracy %, {scorer} scoring. Frontier models use the questions all of them finished.")
        print(f"{'model':8}" + "".join(f"{c:>9}" for c in LEVELS) + f"{'L2-L1':>9}{'L3-L2':>9}{'n(L1)':>8}")
        for mod in local + front:
            v = [t[(mod, c)] for c in LEVELS]
            print(f"{mod:8}" + "".join(f"{x:9.1f}" for x in v) + f"{v[1] - v[0]:9.1f}{v[2] - v[1]:9.1f}{n[(mod, 'L1')]:8d}")


def load_summary(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="print the headline accuracy tables from the results table")
    ap.add_argument("--table", default="results/runs_table.csv")
    args = ap.parse_args(argv)
    print_headline(load_table(args.table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
