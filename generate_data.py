"""
Generate a synthetic B2B SaaS subscription dataset.

Outputs (data/raw/):
  customers.csv       one row per customer (50,000 by default)
  monthly_usage.csv   one row per customer per month they were a paying customer
  cancellations.csv   one row per churned customer, with cancellation date and reason

How churn is simulated
  Each customer has a hidden "account health" score that drifts month to month.
  Health drives the usage you can observe (active users, logins, features used,
  support tickets, NPS), so usage drops BEFORE a customer cancels, as in real products.
  Plan, company size, acquisition channel, region, billing cycle, failed payments,
  early tenure and a July-2024 price increase on the Basic plan also move churn.
  Annual-billing customers can only cancel at their renewal month.

Usage
  python scripts/generate_data.py                       # 50,000 customers, seed 42
  python scripts/generate_data.py --customers 5000 --seed 7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2023-01-01")
N_SIGNUP_MONTHS = 36          # signups Jan-2023 .. Dec-2025
N_MONTHS = 36                 # observed Jan-2023 .. Dec-2025
PRICE_INCREASE_MONTH = 18     # Jul-2024 (month index counted from Jan-2023)

PLANS = ["Basic", "Pro", "Enterprise"]
SIZES = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
CHANNELS = ["Organic", "Paid Search", "Referral", "Partner", "Outbound Sales"]
PRICE_PER_SEAT = {"Basic": 10.0, "Pro": 25.0, "Enterprise": 50.0}
BASIC_PRICE_AFTER_INCREASE = 12.0
ANNUAL_DISCOUNT = 0.85
N_FEATURES = 20

# Effects on the monthly churn log-odds (positive = more churn)
PLAN_EFFECT = {"Basic": 0.35, "Pro": 0.0, "Enterprise": -0.6}
SIZE_EFFECT = {"1-10": 0.4, "11-50": 0.15, "51-200": 0.0, "201-1000": -0.3, "1000+": -0.5}
CHANNEL_EFFECT = {"Organic": 0.0, "Paid Search": 0.3, "Referral": -0.3,
                  "Partner": -0.15, "Outbound Sales": 0.05}
REGION_EFFECT = {"North America": 0.0, "EMEA": 0.0, "APAC": 0.05, "LATAM": 0.15}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def month_start(i):
    return START + pd.DateOffset(months=int(i))


def lookup(mapping, values):
    return np.array([mapping[v] for v in values], dtype=float)


def build_customers(n, rng):
    # Signup volume grows about 2% a month
    weights = 1.02 ** np.arange(N_SIGNUP_MONTHS)
    signup_idx = np.sort(rng.choice(N_SIGNUP_MONTHS, size=n, p=weights / weights.sum()))

    size = rng.choice(SIZES, size=n, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    plan_probs = {  # bigger companies buy bigger plans
        "1-10": [0.70, 0.27, 0.03], "11-50": [0.55, 0.38, 0.07],
        "51-200": [0.35, 0.45, 0.20], "201-1000": [0.15, 0.45, 0.40],
        "1000+": [0.05, 0.35, 0.60],
    }
    plan = np.empty(n, dtype=object)
    for s, probs in plan_probs.items():
        m = size == s
        plan[m] = rng.choice(PLANS, size=m.sum(), p=probs)

    channel_probs = {"Basic": [0.35, 0.33, 0.17, 0.08, 0.07],
                     "Pro": [0.28, 0.22, 0.15, 0.12, 0.23],
                     "Enterprise": [0.12, 0.08, 0.12, 0.18, 0.50]}
    channel = np.empty(n, dtype=object)
    for p, probs in channel_probs.items():
        m = plan == p
        channel[m] = rng.choice(CHANNELS, size=m.sum(), p=probs)

    region = rng.choice(REGIONS, size=n, p=[0.45, 0.30, 0.15, 0.10])
    annual_p = np.select([plan == "Basic", plan == "Pro"], [0.20, 0.35], 0.60)
    billing = np.where(rng.random(n) < annual_p, "annual", "monthly")

    seat_ranges = {"1-10": (1, 10), "11-50": (5, 30), "51-200": (15, 80),
                   "201-1000": (40, 200), "1000+": (100, 400)}
    seats = np.zeros(n, dtype=int)
    for s, (lo, hi) in seat_ranges.items():
        m = size == s
        seats[m] = rng.integers(lo, hi, size=m.sum(), endpoint=True)

    signup_dates = (pd.to_datetime([month_start(i) for i in signup_idx])
                    + pd.to_timedelta(rng.integers(0, 28, size=n), unit="D"))

    customers = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "signup_date": signup_dates.date,
        "plan": plan,
        "company_size": size,
        "region": region,
        "acquisition_channel": channel,
        "billing_cycle": billing,
        "seats_at_signup": seats,
    })
    return customers, signup_idx


def pick_reasons(rng, pay_failed, h, price_shock):
    reasons = rng.choice(
        ["Switched to competitor", "Budget cuts", "Missing features",
         "Low usage", "Company closed", "Price"],
        size=len(h), p=[0.25, 0.20, 0.15, 0.20, 0.10, 0.10]).astype(object)
    reasons[(h < -1.0) & (rng.random(len(h)) < 0.6)] = "Low usage"
    reasons[price_shock & (rng.random(len(h)) < 0.6)] = "Price"
    reasons[pay_failed] = "Payment failure"
    return reasons


def simulate(customers, signup_idx, rng):
    n = len(customers)
    ids = customers["customer_id"].to_numpy()
    plan = customers["plan"].to_numpy()
    annual = customers["billing_cycle"].to_numpy() == "annual"
    channel = customers["acquisition_channel"].to_numpy()
    seats = customers["seats_at_signup"].to_numpy().astype(float)

    static_logit = (-3.75
                    + lookup(PLAN_EFFECT, plan)
                    + lookup(SIZE_EFFECT, customers["company_size"])
                    + lookup(CHANNEL_EFFECT, channel)
                    + lookup(REGION_EFFECT, customers["region"]))
    base_price = lookup(PRICE_PER_SEAT, plan)
    feature_bias = np.select([plan == "Enterprise", plan == "Pro"], [0.4, 0.2], 0.0)

    # Starting health: referrals start healthier, paid-search signups weaker
    health = (rng.normal(0, 1, n)
              + np.where(channel == "Referral", 0.25, 0.0)
              + np.where(channel == "Paid Search", -0.15, 0.0))

    alive = np.ones(n, dtype=bool)
    churn_month = np.full(n, -1)
    churn_reason = np.empty(n, dtype=object)
    frames = []

    for t in range(N_MONTHS):
        idx = np.flatnonzero(alive & (signup_idx <= t))
        if idx.size == 0:
            continue
        tenure = t - signup_idx[idx]
        h = health[idx]
        s = seats[idx]

        # ---- observable usage this month --------------------------------
        price = base_price[idx].copy()
        if t >= PRICE_INCREASE_MONTH:
            price[plan[idx] == "Basic"] = BASIC_PRICE_AFTER_INCREASE
        mrr = np.round(s * price * np.where(annual[idx], ANNUAL_DISCOUNT, 1.0), 2)

        active_users = rng.binomial(s.astype(int), sigmoid(0.6 + 0.9 * h))
        logins = rng.poisson(active_users * np.exp(2.3 + 0.35 * h) * rng.gamma(4, 0.25, idx.size))
        features_used = rng.binomial(N_FEATURES, sigmoid(-0.3 + 0.8 * h + feature_bias[idx]))
        tickets = rng.poisson(np.exp(-1.2 - 0.5 * h + 0.15 * np.log1p(s)))
        payment_failed = rng.random(idx.size) < np.where(annual[idx], 0.004, 0.018)
        surveyed = (tenure % 3 == 2) & (rng.random(idx.size) < 0.25)
        nps = np.where(surveyed,
                       np.clip(np.round(7 + 1.8 * h + rng.normal(0, 1.5, idx.size)), 0, 10),
                       np.nan)

        frames.append(pd.DataFrame({
            "customer_id": ids[idx],
            "month": month_start(t).date(),
            "mrr": mrr,
            "seats": s.astype(int),
            "active_users": active_users,
            "logins": logins,
            "features_used": features_used,
            "support_tickets": tickets,
            "payment_failed": payment_failed.astype(int),
            "nps_score": nps,
        }))

        # ---- churn decision at month end ----------------------------------
        price_shock = ((t >= PRICE_INCREASE_MONTH) & (t < PRICE_INCREASE_MONTH + 3)
                       & (plan[idx] == "Basic"))
        logit = (static_logit[idx] - 0.9 * h
                 + np.where(tenure <= 2, 0.5, 0.0)          # onboarding risk
                 + np.where(payment_failed, 1.4, 0.0)       # involuntary churn
                 + np.where(price_shock, 0.7, 0.0))         # price increase reaction
        renewal = (tenure + 1) % 12 == 0
        p_annual = 1 - (1 - sigmoid(logit - 0.35)) ** 12    # a year of risk at renewal
        p = np.where(annual[idx], np.where(renewal, p_annual, 0.0), sigmoid(logit))
        if t == N_MONTHS - 1:
            p[:] = 0.0                                      # data ends Dec-2025
        churned = rng.random(idx.size) < p

        c = idx[churned]
        alive[c] = False
        churn_month[c] = t
        churn_reason[c] = pick_reasons(rng, payment_failed[churned], h[churned], price_shock[churned])

        # ---- evolve health and seats for next month -------------------------
        health[idx] = h - 0.02 + rng.normal(0, 0.3, idx.size)
        change = rng.random(idx.size) < 0.06
        grow, shrink = change & (h > 0.3), change & (h < -0.5)
        seats[idx[grow]] = np.ceil(s[grow] * rng.uniform(1.05, 1.25, grow.sum()))
        seats[idx[shrink]] = np.maximum(1, np.floor(s[shrink] * rng.uniform(0.75, 0.95, shrink.sum())))

    usage = pd.concat(frames, ignore_index=True)
    gone = np.flatnonzero(churn_month >= 0)
    cancellations = pd.DataFrame({          # effective the 1st of the month after the last paid month
        "customer_id": ids[gone],
        "cancel_date": [month_start(m + 1).date() for m in churn_month[gone]],
        "cancel_reason": churn_reason[gone],
    })
    return usage, cancellations


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--customers", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    customers, signup_idx = build_customers(args.customers, rng)
    usage, cancellations = simulate(customers, signup_idx, rng)

    customers.to_csv(out / "customers.csv", index=False)
    usage.to_csv(out / "monthly_usage.csv", index=False)
    cancellations.to_csv(out / "cancellations.csv", index=False)
    print(f"customers.csv      {len(customers):>9,} rows")
    print(f"monthly_usage.csv  {len(usage):>9,} rows")
    print(f"cancellations.csv  {len(cancellations):>9,} rows "
          f"({len(cancellations) / len(customers):.1%} of customers churned by Dec-2025)")


if __name__ == "__main__":
    main()
