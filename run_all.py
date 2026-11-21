"""
Run the whole project end to end.

  1. generate synthetic data      scripts/generate_data.py
  2. SQL: cohorts, KPIs, features src/run_sql.py  (sql/01..05)
  3. charts                       src/cohort_charts.py
  4. churn model + scoring        src/churn_model.py
  5. A/B test design + readout    src/ab_test.py
  6. Tableau extracts             src/export_tableau.py

Usage:  python run_all.py            (add --skip-data to reuse data/raw)
"""
import subprocess
import sys

STEPS = [
    ["scripts/generate_data.py"],
    ["-m", "src.run_sql"],
    ["-m", "src.cohort_charts"],
    ["-m", "src.churn_model"],
    ["-m", "src.ab_test"],
    ["-m", "src.export_tableau"],
]

if __name__ == "__main__":
    steps = STEPS[1:] if "--skip-data" in sys.argv else STEPS
    for step in steps:
        print(f"\n=== python {' '.join(step)} ===")
        subprocess.run([sys.executable, *step], check=True)
    print("\nDone. Results are in outputs/.")
