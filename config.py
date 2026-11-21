"""Shared paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SQL_DIR = ROOT / "sql"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TABLEAU = OUT / "tableau"
DB_PATH = OUT / "saas.duckdb"

for d in (TABLES, FIGURES, TABLEAU):
    d.mkdir(parents=True, exist_ok=True)

# Churn model: snapshots used for training, threshold tuning and the out-of-time test
TRAIN_SNAPSHOTS = ["2024-03-01", "2024-06-01", "2024-09-01", "2024-12-01", "2025-03-01"]
VALID_SNAPSHOT = "2025-06-01"
TEST_SNAPSHOT = "2025-09-01"
SCORE_SNAPSHOT = "2025-12-01"       # latest month: no label yet, used for scoring
RANDOM_STATE = 42
