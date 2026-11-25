import argparse
import sys

import pandas as pd

from aae.headline import list_cost, load_table
from aae.models import load_models


def main(argv=None):
    ap = argparse.ArgumentParser(description="which questions the semantic layer breaks, and exact cost per correct answer")
    ap.add_argument("--table", default="results/runs_table.csv")
    ap.add_argument("--scorer", default="strict", choices=["strict", "lenient", "correct_logged"])
    args = ap.parse_args(argv)
    df = load_table(args.table)
    c = df[(df.tag.fillna("") == "") & (df.part != "ambiguous") & df[args.scorer].notna()].copy()
    wide = c.pivot_table(index=["model", "question_id"], columns="context", values=args.scorer)
    broke = wide[(wide.L2 == 1) & (wide.L3 == 0)].reset_index()
    print("Questions right at L2 and wrong at L3, by model:")
    print(broke.groupby("model").size().to_string())
    print("\nQuestions broken for the most models:")
    print(broke.question_id.value_counts().head(15).to_string())
    models = load_models()
    paid = c[c.model.map(lambda m: models.get(m, {}).get("provider", "ollama") != "ollama")].copy()
    paid["usd"] = list_cost(paid, models)
    g = paid.groupby(["model", "context"]).agg(usd=("usd", "sum"), right=(args.scorer, "sum"))
    print("\nList-price dollars per correct answer, no caching discount:")
    print(pd.DataFrame((g.usd / g.right).unstack("context").round(4)).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
