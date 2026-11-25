PY ?= python3
TARGET ?= all
PLAN_DIR ?= plan_runs
RESCORED ?=
SHAPES ?=
CLAIMS ?= plan/claims.json

.PHONY: install doctor setup taxi contexts smoke freeze run-laptop run-frontier merge table rescore report verify test lint format clean

install:
	$(PY) -m pip install -e ".[dev]"

doctor:
	$(PY) -m aae doctor --target $(TARGET)

setup:
	$(PY) -m aae setup
	$(PY) -m aae taxi
	$(PY) -m aae contexts

smoke:
	$(PY) -m aae smoke --target $(TARGET)

freeze:
	$(PY) -m aae freeze write

run-laptop:
	$(PY) -m aae run --target laptop

run-frontier:
	$(PY) -m aae run --target frontier

merge:
	$(PY) -m aae merge $(PLAN_DIR) --out runs/runs.jsonl --manifest results/merge_manifest.json

table:
	$(PY) -m aae table --runs runs/runs.jsonl --out results/runs_table.csv $(if $(RESCORED),--rescored $(RESCORED)) $(if $(SHAPES),--shapes $(SHAPES))

rescore:
	$(PY) -m aae rescore run --runs runs/runs.jsonl --out results/rescored.json --summary results/rescore_summary.json

report:
	$(PY) -m aae headline --table results/runs_table.csv
	$(PY) -m aae figures --table results/runs_table.csv --out docs

verify:
	$(PY) -m aae verify --table results/runs_table.csv --claims $(CLAIMS) --rescore-summary results/rescore_summary.json

test:
	$(PY) -m pytest -q

lint:
	ruff check aae tests
	ruff format --check aae tests

format:
	ruff format aae tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
