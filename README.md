# SaaS Customer Churn & Cohort Retention Analysis

SQL and Python analysis of **50,000 B2B SaaS customers** (Jan 2023 - Dec 2025): monthly acquisition
cohorts and 12-month retention curves with SQL window functions, ARPU and churn KPIs with an MRR bridge,
revenue at risk by segment, a logistic-regression churn model, an A/B testing framework for retention
campaigns, and Tableau-ready extracts.

![Retention curves](outputs/figures/retention_curves_by_plan.png)

## Results (synthetic data)
| | |
|---|---|
| 12-month logo retention | 56% overall; Basic 45%, Pro 62%, Enterprise 80% |
| Price increase effect | monthly churn 3.9% → 5.6% after the Jul-2024 Basic increase |
| Top 3 high-value segments at risk | Enterprise/1000+, Enterprise/201-1000, Pro/201-1000: 49% of 12-month MRR at risk |
| Churn model (out-of-time test) | ROC-AUC 0.873; top risk decile holds 46% of churners (4.6x lift) |
| A/B design | 808 per arm to detect a 15% churn reduction (α 0.05, power 0.8) |

![Revenue at risk](outputs/figures/segment_revenue_at_risk.png)

Full write-up: **[REPORT.md](REPORT.md)** · Dashboards: **[tableau/DASHBOARD_SPEC.md](tableau/DASHBOARD_SPEC.md)**

## Quick start
```bash
pip install -r requirements.txt
python run_all.py                 # data -> SQL -> charts -> model -> A/B -> Tableau extracts
duckdb outputs/saas.duckdb        # optional: explore the SQL tables
```

## Repository
```
scripts/generate_data.py          synthetic customers, usage, cancellations
sql/01..05_*.sql                  DuckDB SQL: base table, cohorts, KPIs, segment risk, model features
src/run_sql.py                    runs the SQL and exports tables
src/cohort_charts.py              figures
src/churn_model.py                logistic regression, out-of-time test, customer scoring
src/ab_test.py                    power analysis, hash assignment, balance check, readout
src/export_tableau.py             Tableau data sources
tableau/DASHBOARD_SPEC.md         3 dashboards, calculated fields
outputs/                          tables, figures, Tableau CSVs
```

Data note: synthetic data with built-in churn drivers; see REPORT.md section 9.
