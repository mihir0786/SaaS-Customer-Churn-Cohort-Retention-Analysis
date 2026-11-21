"""
A/B testing framework for a targeted retention campaign.

Campaign idea: high-risk customers (top churn-risk decile) get proactive Customer
Success outreach plus a 15% discount for 3 months. Control gets business as usual.

This module does the four jobs of an experiment:
  1. design   - sample size from baseline churn, minimum detectable effect, alpha, power
  2. assign   - deterministic, hash-based 50/50 randomisation (reproducible, auditable)
  3. check    - covariate balance between arms before launch
  4. analyse  - sample-ratio-mismatch check, two-proportion z-test, 95% CI, revenue impact

Step 4 is demonstrated on SIMULATED outcomes (an assumed true effect), because the
campaign has not actually been run. The simulation only proves the analysis code works;
its numbers are not findings.

Outputs:
  outputs/tables/ab_test_design.json
  outputs/tables/ab_test_assignments.csv
  outputs/tables/ab_test_balance.csv
  outputs/tables/ab_test_simulated_readout.json

Usage:  python -m src.ab_test
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from src.config import RANDOM_STATE, TABLEAU, TABLES

EXPERIMENT_SALT = "retention-offer-2026q1"
ALPHA = 0.05
POWER = 0.80
RELATIVE_MDE = 0.15            # smallest churn reduction worth detecting (15% relative)
DISCOUNT = 0.15
DISCOUNT_MONTHS = 3
CSM_COST_PER_ACCOUNT = 40.0    # assumed cost of one outreach, USD


# ---------------------------------------------------------------- 1. design
def required_sample_size(p_control, relative_mde=RELATIVE_MDE, alpha=ALPHA, power=POWER):
    p_treat = p_control * (1 - relative_mde)
    effect = proportion_effectsize(p_control, p_treat)          # Cohen's h
    n = NormalIndPower().solve_power(effect_size=effect, alpha=alpha, power=power,
                                     ratio=1.0, alternative="two-sided")
    return int(np.ceil(n)), p_treat


def detectable_effect(p_control, n_per_arm, alpha=ALPHA, power=POWER):
    """Smallest relative reduction detectable with the customers we actually have."""
    h = NormalIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power,
                                     ratio=1.0, alternative="two-sided")
    # invert Cohen's h: h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))
    p_treat = np.sin(np.arcsin(np.sqrt(p_control)) - h / 2) ** 2
    return float(1 - p_treat / p_control)


# ---------------------------------------------------------------- 2. assign
def assign_arm(customer_id, salt=EXPERIMENT_SALT):
    digest = hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()
    return "treatment" if int(digest[:8], 16) % 100 < 50 else "control"


# ---------------------------------------------------------------- 3. check
def balance_table(df, covariates):
    rows = []
    for col in covariates:
        t, c = df.loc[df.arm == "treatment", col], df.loc[df.arm == "control", col]
        pooled_sd = np.sqrt((t.var() + c.var()) / 2)
        rows.append({"covariate": col, "treatment_mean": round(t.mean(), 4),
                     "control_mean": round(c.mean(), 4),
                     "std_mean_diff": round((t.mean() - c.mean()) / pooled_sd, 4) if pooled_sd else 0.0})
    out = pd.DataFrame(rows)
    out["balanced"] = out["std_mean_diff"].abs() < 0.1           # common rule of thumb
    return out


# ---------------------------------------------------------------- 4. analyse
def analyse(results, expected_split=0.5):
    """results: one row per customer with columns arm, churned (0/1), mrr."""
    n_t = int((results.arm == "treatment").sum())
    n_c = int((results.arm == "control").sum())
    srm_p = stats.chisquare([n_t, n_c], [(n_t + n_c) * expected_split] * 2).pvalue

    churn_t = results.loc[results.arm == "treatment", "churned"]
    churn_c = results.loc[results.arm == "control", "churned"]
    p_t, p_c = churn_t.mean(), churn_c.mean()
    z, p_value = proportions_ztest([churn_t.sum(), churn_c.sum()], [n_t, n_c])
    diff = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    ci = (diff - 1.96 * se, diff + 1.96 * se)

    # Revenue view: MRR kept per treated customer vs control, minus campaign cost
    mrr_t = results.loc[results.arm == "treatment", "mrr"]
    retained_mrr_per_customer = (1 - p_t) * mrr_t.mean() - (1 - p_c) * mrr_t.mean()
    cost_per_customer = CSM_COST_PER_ACCOUNT + (1 - p_t) * mrr_t.mean() * DISCOUNT * DISCOUNT_MONTHS
    annual_value = retained_mrr_per_customer * 12

    return {
        "sample_ratio_mismatch_p": round(float(srm_p), 4),
        "srm_ok": bool(srm_p > 0.001),
        "n_treatment": n_t, "n_control": n_c,
        "churn_treatment": round(float(p_t), 4), "churn_control": round(float(p_c), 4),
        "absolute_difference": round(float(diff), 4),
        "ci95_absolute_difference": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "relative_change": round(float(diff / p_c), 4),
        "z": round(float(z), 3), "p_value": round(float(p_value), 5),
        "significant": bool(p_value < ALPHA),
        "retained_mrr_per_treated_customer": round(float(retained_mrr_per_customer), 2),
        "campaign_cost_per_treated_customer": round(float(cost_per_customer), 2),
        "annualised_value_per_treated_customer": round(float(annual_value), 2),
        "roi": round(float((annual_value - cost_per_customer) / cost_per_customer), 2),
    }


def main():
    scores = pd.read_csv(TABLEAU / "customer_churn_scores.csv")
    metrics = json.loads((TABLES / "model_metrics.json").read_text())

    # Baseline: observed 90-day churn of the riskiest decile in the held-out test
    t = metrics["test"]
    baseline = round(t["churn_rate"] * t["top_decile_lift"], 4)
    target = scores[scores.risk_tier == "High"].copy()

    n_needed, p_treat = required_sample_size(baseline)
    n_available = len(target) // 2
    design = {
        "population": "High risk tier (top churn-risk decile) at the Dec-2025 scoring snapshot",
        "eligible_customers": int(len(target)),
        "primary_metric": "90-day logo churn",
        "secondary_metrics": ["retained MRR", "net revenue retention"],
        "guardrails": ["discount cost", "support tickets per account", "downgrades"],
        "baseline_churn_control": baseline,
        "relative_mde": RELATIVE_MDE,
        "expected_treatment_churn_at_mde": round(float(p_treat), 4),
        "alpha": ALPHA, "power": POWER,
        "required_per_arm": n_needed,
        "available_per_arm": n_available,
        "detectable_relative_effect_with_available_sample": round(detectable_effect(baseline, n_available), 4),
        "duration": "fixed 90-day horizon from each customer's assignment; no early stopping on peeks",
        "randomisation": f"sha256('{EXPERIMENT_SALT}:' + customer_id), bucket < 50 -> treatment",
    }
    (TABLES / "ab_test_design.json").write_text(json.dumps(design, indent=2))

    target["arm"] = target["customer_id"].map(assign_arm)
    target[["customer_id", "arm", "plan", "company_size", "churn_probability", "mrr"]] \
        .to_csv(TABLES / "ab_test_assignments.csv", index=False)

    target["is_basic"] = (target.plan == "Basic").astype(int)
    target["is_monthly"] = (target.billing_cycle == "monthly").astype(int)
    bal = balance_table(target, ["churn_probability", "mrr", "tenure_months", "is_basic", "is_monthly"])
    bal.to_csv(TABLES / "ab_test_balance.csv", index=False)

    # ---- SIMULATED outcome, only to exercise the analysis code ----------------
    rng = np.random.default_rng(RANDOM_STATE)
    assumed_effect = 0.20                         # pretend the offer cuts churn by 20%
    p = target["churn_probability"].to_numpy()
    p = np.where(target.arm == "treatment", p * (1 - assumed_effect), p)
    sim = target[["customer_id", "arm", "mrr"]].assign(churned=(rng.random(len(p)) < p).astype(int))
    readout = {"NOTE": "SIMULATED outcomes with an assumed 20% relative effect; not a real result",
               **analyse(sim)}
    (TABLES / "ab_test_simulated_readout.json").write_text(json.dumps(readout, indent=2))

    print(f"Baseline churn in High tier: {baseline:.1%}; to detect a {RELATIVE_MDE:.0%} relative drop "
          f"need {n_needed:,}/arm, have {n_available:,}/arm "
          f"(detectable: {design['detectable_relative_effect_with_available_sample']:.1%})")
    print("Balance (|SMD| < 0.1 = balanced):")
    print(bal.to_string(index=False))
    print("Simulated readout:", json.dumps(readout, indent=1))


if __name__ == "__main__":
    main()
