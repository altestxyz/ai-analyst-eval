import re
import sys

import yaml

from aae.config import CTX_DIR
from aae.contexts import all_dbs, assemble_l3
from aae.llm import chat
from aae.models import load_models

MODELS = load_models()

SYSTEM = (
    "You are an analytics engineer writing a semantic model: the business definitions, join paths "
    "and metric rules that a SQL analyst agent needs but the raw schema does not state."
)

USER = """Below is the schema of a DuckDB database, with column descriptions and example values.
Write a semantic model for it as YAML.

Rules:
- Use ONLY table and column names that appear in the schema. Never invent a column.
- Base everything on the schema and the descriptions. Do not write SQL for particular questions.
- Sections: database, entities (name, table, key, synonyms), joins (from, to, use_when),
  metrics (name, expression, grain, filters, notes), value_codes (what coded values mean, taken from the
  descriptions), gotchas (data quirks, date formats, numbers stored as text, NULL handling, terms that
  could be read two ways and which reading to use).
- A metric's expression must be a SQL expression over real columns.
- Be concise: about 150 lines at most. Output only the YAML. No code fences, no commentary.

Example of the shape (these names are made up):
database: example_db
entities:
  - name: customer
    table: customers
    key: customer_id
    synonyms: ["client"]
joins:
  - from: orders.customer_id
    to: customers.customer_id
    use_when: "you need customer attributes next to order totals"
metrics:
  - name: net_revenue
    expression: "SUM(order_total - refunds)"
    grain: order
    filters: ["status = 'completed'"]
    notes: "Excludes cancelled orders. 'Sales' and 'revenue' both mean this."

SCHEMA:
{schema}
"""


def strip_fences(t):
    t = t.strip()
    m = re.search(r"```(?:ya?ml)?\s*(.*?)```", t, re.S | re.I)
    return (m.group(1) if m else t).strip()


def draft(db_id, cfg):
    out = CTX_DIR / db_id / "semantic.yaml"
    if out.exists():
        print(f"  {db_id}: semantic.yaml already there, keeping it")
        return
    schema = (CTX_DIR / db_id / "L2.txt").read_text()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(schema=schema)}]
    text = strip_fences(chat(cfg, msgs)["text"])
    try:
        yaml.safe_load(text)
    except Exception as e:
        msgs += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"That is not valid YAML ({str(e)[:200]}). Send the corrected YAML only."},
        ]
        text = strip_fences(chat(cfg, msgs)["text"])
        try:
            yaml.safe_load(text)
        except Exception:
            print(f"  warning: {db_id} semantic model is not valid YAML. It is still usable as text; read it.")
    if len(text) < 200:
        raise SystemExit(f"{db_id}: the model returned almost nothing ({len(text)} chars). Check the model and key.")
    out.write_text(text + "\n")
    print(f"  {db_id}: wrote semantic.yaml ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    author = sys.argv[1] if len(sys.argv) > 1 else "F1"
    cfg = dict(MODELS[author])
    if cfg["provider"] != "ollama":
        cfg["max_output_tokens"] = max(cfg.get("max_output_tokens", 4096), 8192)
    for db in all_dbs():
        draft(db, cfg)
        assemble_l3(db)
