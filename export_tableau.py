"""
Write the Tableau extracts (CSV) to outputs/tableau/.
Connect Tableau Public / Desktop to these files; see tableau/DASHBOARD_SPEC.md.

  cohort_retention.csv          cohort x month triangle (heatmap)
  retention_curves.csv          12-month curves by plan / channel / size / billing
  monthly_kpis.csv              churn rate, ARPU, MRR bridge per month
  segment_revenue_at_risk.csv   segment ARPU, churn and 12-month MRR at risk
  segment_model_risk.csv        model-based 90-day MRR at risk per segment
  customer_churn_scores.csv     one row per active customer (written by churn_model.py)

Usage:  python -m src.export_tableau
"""
import shutil

import pandas as pd

from src.config import TABLEAU, TABLES


def main():
    for name in ["cohort_retention", "retention_curves", "monthly_kpis", "segment_revenue_at_risk"]:
        shutil.copy(TABLES / f"{name}.csv", TABLEAU / f"{name}.csv")

    scores = pd.read_csv(TABLEAU / "customer_churn_scores.csv")
    seg = (scores.assign(segment=scores.plan + " / " + scores.company_size)
           .groupby(["segment", "plan", "company_size"], as_index=False)
           .agg(active_customers=("customer_id", "size"),
                current_mrr=("mrr", "sum"),
                avg_churn_probability=("churn_probability", "mean"),
                high_risk_customers=("risk_tier", lambda s: (s == "High").sum()),
                expected_mrr_at_risk_90d=("expected_mrr_at_risk_90d", "sum"))
           .sort_values("expected_mrr_at_risk_90d", ascending=False))
    seg["arpu"] = seg.current_mrr / seg.active_customers
    seg.round(4).to_csv(TABLEAU / "segment_model_risk.csv", index=False)

    for f in sorted(TABLEAU.glob("*.csv")):
        print(f"  {f.name:<32} {sum(1 for _ in open(f)) - 1:>7,} rows")


if __name__ == "__main__":
    main()
