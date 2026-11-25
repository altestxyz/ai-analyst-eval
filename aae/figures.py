import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aae.headline import LEVELS, accuracy_table, list_cost, load_table, main_grid, per_question, shared_questions, split_models
from aae.models import load_models
from aae.score import paired_diff_ci

COLORS = {"F1": "#eb6834", "F2": "#6250d6", "F3": "#eda100", "M1": "#1baf7a", "M2": "#2a78d6"}
INK = "#1f2328"
GRID = "#d9dde1"
LIGHTNESS = {"L1": 0.55, "L2": 0.25, "L3": 0.0}


def tint(hex_color, amount):
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], float)
    return tuple((rgb + (255 - rgb) * amount) / 255)


def style(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def models_present(df, models):
    local, front = split_models(models)
    present = set(df.model.unique())
    return [m for m in local if m in present], [m for m in front if m in present]


def label(models, m):
    return models.get(m, {}).get("label", m)


def fig_accuracy(df, models, out):
    local, front = models_present(df, models)
    order = front + local
    strict, _ = accuracy_table(df, "strict", local, front)
    lenient, _ = accuracy_table(df, "lenient", local, front)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = 0.26
    for i, m in enumerate(order):
        for j, c in enumerate(LEVELS):
            x = i + (j - 1) * width
            ax.bar(x, strict[(m, c)], width * 0.92, color=tint(COLORS[m], LIGHTNESS[c]))
            ax.scatter([x], [lenient[(m, c)]], marker="D", s=34, color=INK, zorder=3)
            ax.text(x, 1.5, f"{strict[(m, c)]:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label(models, m) for m in order])
    ax.set_ylim(0, 100)
    style(ax, "Correct answers, % of BIRD questions")
    ax.set_title("Accuracy by context level. Bars: strict grading. Diamonds: extra columns allowed.", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_layer(df, models, out):
    local, front = models_present(df, models)
    order = front + local
    d = main_grid(df)
    d = d[d.part == "bird"]
    pq = per_question(d, "strict")
    shared = shared_questions(pq, front) if front else set()
    lenient, _ = accuracy_table(df, "lenient", local, front)
    strict, _ = accuracy_table(df, "strict", local, front)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(order):
        p = pq[pq.model == m]
        if m in front:
            p = p[p.question_id.isin(shared)]
        w = p.pivot(index="question_id", columns="context", values="strict").dropna()
        diff, lo, hi = paired_diff_ci(w["L2"].values, w["L3"].values)
        ax.bar(i, diff, 0.55, color=COLORS[m])
        ax.errorbar([i], [diff], yerr=[[diff - lo], [hi - diff]], color=INK, capsize=4, linewidth=1.2)
        ax.scatter([i], [lenient[(m, "L3")] - lenient[(m, "L2")]], marker="D", s=40, color=INK, zorder=3)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label(models, m) for m in order])
    style(ax, "Change in accuracy, L3 minus L2 (points)")
    ax.set_title("Adding the drafted layer. Bars: strict with 95% interval. Diamonds: extra columns allowed.", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_shape(df, models, out):
    local, front = models_present(df, models)
    order = front + local
    strict, _ = accuracy_table(df, "strict", local, front)
    lenient, _ = accuracy_table(df, "lenient", local, front)
    rows = [(m, c) for m in order for c in LEVELS]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for y, (m, c) in enumerate(rows[::-1]):
        ax.plot([strict[(m, c)], lenient[(m, c)]], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.scatter([strict[(m, c)]], [y], color=COLORS[m], s=46, zorder=3)
        ax.scatter([lenient[(m, c)]], [y], color=COLORS[m], marker="D", s=46, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{label(models, m)} {c}" for m, c in rows[::-1]], fontsize=8)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("Correct answers, % of BIRD questions", color=INK)
    ax.set_title("Strict (dot) and extra-columns-allowed (diamond) scores", loc="left", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def cost_table(df, models):
    _, front = models_present(df, models)
    c = main_grid(df)
    c = c[c.part.isin(["bird", "taxi"]) & c.correct_logged.notna() & c.model.isin(front)].copy()
    c["usd"] = list_cost(c, models)
    g = c.groupby(["model", "context"]).agg(usd=("usd", "sum"), right=("correct_logged", "sum"))
    return (100 * g.usd / g.right).unstack("context")[list(LEVELS)]


def fig_cost(df, models, out):
    t = cost_table(df, models)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.26
    for i, m in enumerate(t.index):
        for j, c in enumerate(LEVELS):
            x = i + (j - 1) * width
            ax.bar(x, t.loc[m, c], width * 0.92, color=tint(COLORS[m], LIGHTNESS[c]))
            ax.text(x, t.loc[m, c] + 0.1, f"{t.loc[m, c]:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(t.index)))
    ax.set_xticklabels([label(models, m) for m in t.index])
    style(ax, "Cents per correct answer, list price, no caching")
    ax.set_title("Cost of a correct answer at L1, L2 and L3 (light to dark)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_tables(df, models, folder):
    local, front = models_present(df, models)
    for scorer in ("strict", "lenient"):
        t, n = accuracy_table(df, scorer, local, front)
        rows = [{"model": m, **{c: round(t[(m, c)], 2) for c in LEVELS}, "questions": n[(m, "L1")]} for m in local + front]
        pd.DataFrame(rows).to_csv(folder / f"accuracy_{scorer}.csv", index=False)
    cost_table(df, models).round(3).to_csv(folder / "cents_per_correct.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description="rebuild the post's charts and tables from the results table")
    ap.add_argument("--table", default="results/runs_table.csv")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args(argv)
    df = load_table(args.table)
    models = load_models()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    fig_accuracy(df, models, out / "figures" / "fig1_accuracy_by_level.png")
    fig_layer(df, models, out / "figures" / "fig2_drafted_layer_change.png")
    fig_shape(df, models, out / "figures" / "fig3_strict_vs_lenient.png")
    fig_cost(df, models, out / "figures" / "fig4_cost_per_correct.png")
    write_tables(df, models, out / "tables")
    print(f"wrote 4 figures to {out / 'figures'} and 3 tables to {out / 'tables'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
