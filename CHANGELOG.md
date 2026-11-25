# Changelog

## 1.0.0

First public release. One codebase, two ways to run it.

- One package, `aae`, drives laptop-class models through Ollama and frontier models through either the Kaggle model proxy or the vendors' own APIs. `models.yaml` lists both routes for each frontier model and `AAE_ROUTE` picks one.
- `python -m aae.merge` combines every run log in a plan folder by run id. Identical copies are counted once and conflicting copies are reported. The published log merges 7,742 records into 3,524 unique runs with no conflicts.
- `python -m aae table` writes one row per run with no SQL text and no answer keys.
- `python -m aae rescore` re-runs each saved query and scores it strictly and leniently.
- `python -m aae verify` compares the claims in `plan/claims.json` with the results table, rounding half up to each claim's decimals.
- Source files carry no comments or docstrings. The description of the study is in `README.md`, the notebook has an instruction header, and `models.yaml` explains its own fields.
- Added `LICENSE` (MIT) and `LICENSE-DATA.md` (CC BY 4.0 for the results, charts and plan).
- `python -m aae taxi` skips the taxi set when its questions are not present, so `make setup` completes on a fresh clone.
- Not included: the Kaggle Benchmarks task packaging, the raw run logs, and the taxi questions and column descriptions.

## Harness changes after the first partial run

The first partial run logged 809 frontier runs. Two harness versions followed. Neither changed the questions, the contexts or the scoring rule. The dates are when the notebook versions were saved (UTC), from the version names and file times.

### v10 (2026-11-18)

- Frontier models run in rounds, so every model answers the same questions before the quota runs out.
- Cost is recomputed from logged tokens at the prices in `models.yaml`.
- `max_output_tokens` for F1 and F3 changed from 4,096 to 1,024, with an automatic retry at 4,096 when a reply is cut off. It is not part of the plan hash and is logged per run.

### v11 (2026-11-19)

- The remaining frontier questions are split into independent slices that a scheduler can run on separate days (`python -m aae shard`).
- Every run logs reasoning tokens, the reasoning setting and `max_output_tokens`.
- Each wrong answer gets a rule-based shape (empty, wrong_columns, too_many_rows, too_few_rows, partial_overlap, disjoint_values, asked, no_result, truncated).
- The report flags intervals that sit wholly below zero and baselines at or above 95 percent.
