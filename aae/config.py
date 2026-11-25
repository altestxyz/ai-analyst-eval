import os
from pathlib import Path

import yaml

from aae.env import load_env

load_env()

DATA_DIR = Path(os.getenv("AAE_DATA", "data"))
ART_DIR = Path("artifacts")
BIRD_DIR = Path(os.getenv("BIRD_DIR", str(DATA_DIR / "minidev" / "MINIDEV")))
DB_DIR = DATA_DIR / "duckdb"
TAXI_DIR = DATA_DIR / "taxi"
CTX_DIR = Path("contexts")
RUNS_DIR = Path("runs")
RUNS_FILE = RUNS_DIR / "runs.jsonl"

KEPT_FILE = ART_DIR / "kept.json"
GOLD_FILE = ART_DIR / "gold.pkl"
QUESTIONS_FILE = ART_DIR / "questions.jsonl"
SCALE_QUESTIONS_FILE = ART_DIR / "scale_questions.jsonl"
TAXI_QUESTIONS_FILE = ART_DIR / "taxi_questions.jsonl"

EXP = yaml.safe_load(open("experiment.yaml"))
DBS = EXP["databases"]
MAX_ATTEMPTS = EXP["max_sql_attempts"]
