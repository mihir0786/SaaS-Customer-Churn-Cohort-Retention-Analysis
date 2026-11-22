"""Shared paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "Data"
SQL_DIR = ROOT / "Sql"
OUT = ROOT / "Output"

TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TABLEAU = OUT / "tableau"
DB_PATH = OUT / "saas.duckdb"

for d in (TABLES, FIGURES, TABLEAU):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_SNAPSHOTS = [
    "2024-03-01",
    "2024-06-01",
    "2024-09-01",
    "2024-12-01",
    "2025-03-01"
]

VALID_SNAPSHOT = "2025-06-01"
TEST_SNAPSHOT = "2025-09-01"
SCORE_SNAPSHOT = "2025-12-01"

RANDOM_STATE = 42
