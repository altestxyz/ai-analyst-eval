# Test plan

This is the plan the study was run against. It is reconstructed from the committed configuration (`experiment.yaml`, `models.yaml`, `aae/prompts.py`, `aae/score.py`) and the Kaggle session record. The hashes written by `python -m aae freeze write` at the start of the first run are not reproduced by this repository: comments were removed from the source after the run, and the prompt hash covers file bytes. Run `freeze write` again before any new run.

## Question

Does adding context to a text-to-SQL agent change its accuracy, and can a laptop-class model with good context match a frontier model that sees only the bare schema?

| Level | Context given to the agent |
| --- | --- |
| L1 | Table and column names |
| L2 | L1 plus column descriptions and sample values |
| L3 | L2 plus join rules and metric definitions drafted by F1 from the schema and descriptions only |

## Models

| Key | Model | Route |
| --- | --- | --- |
| M1 | Qwen2.5-Coder 7B, 4-bit | Ollama |
| M2 | Qwen2.5-Coder 14B, 4-bit | Ollama |
| F1 | Claude Opus 5.5 | Kaggle proxy or Anthropic API |
| F2 | Gemini 4 Argon | Kaggle proxy or Gemini API |
| F3 | GPT-5.6 Terra | Kaggle proxy or OpenAI API |
| JUDGE | Qwen2.5 7B | Ollama |

Model versions were pinned on 2026-11-18. Frontier models ran at their default reasoning setting.

## Questions

| Set | Count | Source |
| --- | --- | --- |
| BIRD | 152 | 38 from each of formula_1, card_games, superhero, european_football_2; quotas 10 challenging, 20 moderate, 8 simple per database; seed 7 |
| Taxi | 30 | held-out questions on three months of NYC taxi trips |
| Ambiguous | 20 | taxi questions that need a clarifying question or a stated assumption |

The identifiers are in `question_ids.json`. Question text and answer keys are not published.

## Harness

- Read-only DuckDB connection, 10,000 rows, 30 seconds, three SQL attempts per question.
- The agent never sees an answer key.
- Repeats: two for M1 and M2, one for F1, F2 and F3.
- A spending limit stopped each frontier model after a random 95 BIRD questions. Comparisons between small and frontier models use those 95.

## Scoring

- Strict: the result must match the answer key, extra columns count as wrong.
- Lenient: extra columns are accepted when every expected column is right. Added after the first results; always reported next to strict.
- Failure shapes for wrong answers are rule-based (`aae/score.py`). The judge model (kappa of at least 0.70 against 50 hand labels, or hand labelling) is used only for failure causes, never for accuracy.

## Decision rules

- Primary comparison: pooled L2 against L1.
- Intervals: paired bootstrap, 10,000 resamples, 95 percent.
- Verdict for a gain: supported when the lower bound is above zero and the gain is at least 5 points; not supported when the upper bound is below 5 points; inconclusive otherwise.
- Laptop against frontier: within margin when the lower bound is above minus 5 points; behind when the upper bound is below minus 5 points; inconclusive otherwise.
- Every other comparison is exploratory.

## Changes after the first partial run

See `CHANGELOG.md`.

## Claims

`claims.json` lists the numbers quoted in the post that the results table can check. `make verify` compares each one with the results table.
