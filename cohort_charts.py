"""
Static charts of the SQL outputs (for the README and to preview the Tableau views).

  outputs/figures/cohort_retention_heatmap.png
  outputs/figures/retention_curves_by_plan.png
  outputs/figures/churn_and_arpu_trend.png
  outputs/figures/segment_revenue_at_risk.png

Usage:  python -m src.cohort_charts
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.config import FIGURES, TABLES


def heatmap():
    c = pd.read_csv(TABLES / "cohort_retention.csv", parse_dates=["cohort_month"])
    m = c.pivot(index="cohort_month", columns="months_since_signup", values="logo_retention")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(m.values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(m.shape[1]), m.columns)
    ax.set_yticks(range(m.shape[0]), [d.strftime("%b %Y") for d in m.index], fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5)
    ax.set(xlabel="Months since signup", ylabel="Signup cohort",
           title="Logo retention by monthly signup cohort")
    fig.colorbar(im, ax=ax, format=mtick.PercentFormatter(1.0), shrink=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "cohort_retention_heatmap.png", dpi=150)
    plt.close(fig)


def curves_by_plan():
    r = pd.read_csv(TABLES / "retention_curves.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seg in ["Enterprise", "Pro", "Basic"]:
        d = r[(r.dimension == "Plan") & (r.segment == seg)]
        ax.plot(d.months_since_signup, d.logo_retention, marker="o", lw=2, label=seg)
    d = r[r.dimension == "Overall"]
    ax.plot(d.months_since_signup, d.logo_retention, ls="--", c="black", label="All customers")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set(xlabel="Months since signup", ylabel="Customers still paying",
           title="12-month logo retention by plan", xticks=range(13), ylim=(0.3, 1.02))
    ax.annotate("Annual renewals\ncome due", xy=(12, 0.62), xytext=(8.6, 0.42),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "retention_curves_by_plan.png", dpi=150)
    plt.close(fig)


def churn_and_arpu():
    k = pd.read_csv(TABLES / "monthly_kpis.csv", parse_dates=["month"]).iloc[1:]
    fig, ax1 = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1[0].plot(k.month, k.logo_churn_rate, marker=".", label="Monthly logo churn")
    ax1[0].plot(k.month, k.logo_churn_rate_3m_avg, lw=2, label="3-month average")
    ax1[0].axvline(pd.Timestamp("2024-07-01"), c="grey", ls=":")
    ax1[0].text(pd.Timestamp("2024-07-10"), k.logo_churn_rate.max() * 0.97,
                "Basic price +20%", fontsize=8)
    ax1[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
    ax1[0].set(ylabel="Churn rate", title="Monthly churn rate and ARPU")
    ax1[0].legend(fontsize=8)
    ax1[1].plot(k.month, k.arpu, c="#2e86c1", lw=2)
    ax1[1].yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax1[1].set(ylabel="ARPU (MRR per customer)")
    fig.tight_layout()
    fig.savefig(FIGURES / "churn_and_arpu_trend.png", dpi=150)
    plt.close(fig)


def segment_risk():
    s = pd.read_csv(TABLES / "segment_revenue_at_risk.csv").sort_values("mrr_at_risk_12m")
    colors = np.where(s.risk_rank_high_value.le(3), "#c0392b",
                      np.where(s.is_high_value, "#e59866", "#aab7b8"))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(s.segment, s.mrr_at_risk_12m / 1e6, color=colors)
    for y, (v, ch) in enumerate(zip(s.mrr_at_risk_12m, s.monthly_mrr_churn)):
        ax.text(v / 1e6 + 0.02, y, f"${v / 1e6:.2f}M  ({ch:.1%}/mo)", va="center", fontsize=8)
    ax.set(xlabel="MRR expected to churn in the next 12 months ($M)",
           title="Revenue at risk by segment (red = top 3 high-ARPU segments)")
    ax.set_xlim(0, s.mrr_at_risk_12m.max() / 1e6 * 1.35)
    fig.tight_layout()
    fig.savefig(FIGURES / "segment_revenue_at_risk.png", dpi=150)
    plt.close(fig)


def main():
    heatmap()
    curves_by_plan()
    churn_and_arpu()
    segment_risk()
    print("Charts written to", FIGURES)


if __name__ == "__main__":
    main()
