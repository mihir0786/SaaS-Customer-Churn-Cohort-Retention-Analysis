"""
Logistic-regression churn model.

Question: which customers paying today will stop paying within 90 days?
Data:     outputs/tables/churn_features.csv (built by sql/05_churn_features.sql)
Split:    out-of-time, never random, so the test mimics real use
            train  = snapshots Mar-2024 .. Mar-2025
            valid  = Jun-2025  (used only to choose the alert threshold)
            test   = Sep-2025  (touched once, for the reported metrics)
            score  = Dec-2025  (latest month, no label yet -> risk list for the business)

Outputs:
  outputs/tables/model_metrics.json
  outputs/tables/model_coefficients.csv
  outputs/tableau/customer_churn_scores.csv
  outputs/figures/roc_curve.png, cumulative_gains.png, model_coefficients.png

Usage:  python -m src.churn_model
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (FIGURES, RANDOM_STATE, SCORE_SNAPSHOT, TABLEAU, TABLES,
                        TEST_SNAPSHOT, TRAIN_SNAPSHOTS, VALID_SNAPSHOT)

CATEGORICAL = ["plan", "company_size", "region", "acquisition_channel", "billing_cycle"]
NUMERIC = [
    "tenure_months", "log_mrr", "log_seats", "can_cancel_in_window", "months_in_window",
    "seat_utilization_3m", "seat_utilization_change",
    "logins_per_user_3m", "logins_per_user_change",
    "features_used_3m", "features_used_change",
    "support_tickets_3m", "payment_failures_3m", "mrr_change_pct_3m",
    "last_nps", "has_nps",
]
TARGET = "churned_next_90d"


def load_features():
    df = pd.read_csv(TABLES / "churn_features.csv", parse_dates=["snapshot_month"])
    df["log_mrr"] = np.log1p(df["mrr"])
    df["log_seats"] = np.log1p(df["seats"])
    df["has_nps"] = df["last_nps"].notna().astype(int)
    # Missing trend values mean "no change measurable" (e.g. zero active users)
    for col in ["logins_per_user_change", "seat_utilization_change", "features_used_change"]:
        df[col] = df[col].fillna(0)
    return df


def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
    ])
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)
    return Pipeline([("prep", pre), ("model", model)])


def best_f1_threshold(y, p):
    grid = np.arange(0.05, 0.80, 0.01)
    scores = [f1_score(y, p >= t) for t in grid]
    return round(float(grid[int(np.argmax(scores))]), 2)


def lift_at(y, p, top=0.10):
    order = np.argsort(-p)
    k = int(len(p) * top)
    return float(np.asarray(y)[order[:k]].mean() / np.mean(y))


def evaluate(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "rows": int(len(y)),
        "churn_rate": round(float(np.mean(y)), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "accuracy_at_0.5": round(accuracy_score(y, p >= 0.5), 4),
        "baseline_accuracy_predict_no_churn": round(1 - float(np.mean(y)), 4),
        "threshold": threshold,
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "top_decile_lift": round(lift_at(y, p, 0.10), 2),
        "top_decile_capture_rate": round(float(np.asarray(y)[np.argsort(-p)[: len(p) // 10]].sum() / np.sum(y)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def coefficient_table(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    tbl = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coefficient": coefs})
    tbl["odds_ratio"] = np.exp(tbl["coefficient"])
    tbl["direction"] = np.where(tbl["coefficient"] > 0, "raises churn risk", "lowers churn risk")
    return tbl.reindex(tbl["coefficient"].abs().sort_values(ascending=False).index).round(4)


def plot_roc(y, p, auc):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"Logistic regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="Random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curve, out-of-time test (Sep-2025)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_curve.png", dpi=150)
    plt.close(fig)


def plot_gains(y, p):
    order = np.argsort(-p)
    captured = np.cumsum(np.asarray(y)[order]) / np.sum(y)
    share = np.arange(1, len(p) + 1) / len(p)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(share, captured, lw=2, label="Model")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="Random targeting")
    ax.set(xlabel="Share of customers contacted (highest risk first)",
           ylabel="Share of churners reached", title="Cumulative gains, Sep-2025 test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "cumulative_gains.png", dpi=150)
    plt.close(fig)


def plot_coefficients(tbl):
    top = tbl.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["coefficient"],
            color=np.where(top["coefficient"] > 0, "#c0392b", "#2e86c1"))
    ax.axvline(0, c="black", lw=0.8)
    ax.set(xlabel="Coefficient (log-odds per 1 SD or vs. reference category)",
           title="Strongest churn drivers")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_coefficients.png", dpi=150)
    plt.close(fig)


def main():
    df = load_features()
    snap = df["snapshot_month"].dt.strftime("%Y-%m-%d")
    train = df[snap.isin(TRAIN_SNAPSHOTS)]
    valid = df[snap == VALID_SNAPSHOT]
    test = df[snap == TEST_SNAPSHOT]
    score = df[snap == SCORE_SNAPSHOT].copy()
    X = CATEGORICAL + NUMERIC

    pipe = build_pipeline().fit(train[X], train[TARGET])
    threshold = best_f1_threshold(valid[TARGET], pipe.predict_proba(valid[X])[:, 1])

    p_test = pipe.predict_proba(test[X])[:, 1]
    metrics = {
        "train_rows": int(len(train)),
        "train_snapshots": TRAIN_SNAPSHOTS,
        "validation": evaluate(valid[TARGET], pipe.predict_proba(valid[X])[:, 1], threshold),
        "test": evaluate(test[TARGET], p_test, threshold),
    }
    (TABLES / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    coefs = coefficient_table(pipe)
    coefs.to_csv(TABLES / "model_coefficients.csv", index=False)

    plot_roc(test[TARGET], p_test, metrics["test"]["roc_auc"])
    plot_gains(test[TARGET], p_test)
    plot_coefficients(coefs)

    # Refit on every labelled snapshot before scoring the latest month
    labelled = df[df[TARGET].notna()]
    final = build_pipeline().fit(labelled[X], labelled[TARGET])
    score["churn_probability"] = final.predict_proba(score[X])[:, 1].round(4)
    score["risk_decile"] = pd.qcut(score["churn_probability"].rank(method="first"), 10,
                                   labels=range(1, 11)).astype(int)   # 10 = riskiest
    score["risk_tier"] = pd.cut(score["risk_decile"], [0, 7, 9, 10],
                                labels=["Low", "Medium", "High"]).astype(str)
    score["expected_mrr_at_risk_90d"] = (score["churn_probability"] * score["mrr"]).round(2)
    cols = ["customer_id", "snapshot_month", "plan", "company_size", "region",
            "acquisition_channel", "billing_cycle", "tenure_months", "mrr", "seats",
            "seat_utilization_3m", "logins_per_user_3m", "features_used_3m",
            "support_tickets_3m", "payment_failures_3m", "last_nps",
            "churn_probability", "risk_decile", "risk_tier", "expected_mrr_at_risk_90d"]
    score[cols].sort_values("churn_probability", ascending=False) \
        .to_csv(TABLEAU / "customer_churn_scores.csv", index=False)

    t = metrics["test"]
    print(f"Train rows {len(train):,} | test rows {t['rows']:,} | test churn rate {t['churn_rate']:.1%}")
    print(f"ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | top-decile lift {t['top_decile_lift']}x")
    print(f"Accuracy @0.5 {t['accuracy_at_0.5']:.1%} (baseline {t['baseline_accuracy_predict_no_churn']:.1%})")
    print(f"@threshold {t['threshold']:.2f}: precision {t['precision']:.1%}, recall {t['recall']:.1%}, "
          f"F1 {t['f1']:.3f}, accuracy {t['accuracy']:.1%}")
    print("Top drivers:")
    print(coefs.head(8)[["feature", "coefficient", "odds_ratio"]].to_string(index=False))
    print(f"Scored {len(score):,} active customers for {SCORE_SNAPSHOT}; "
          f"High tier expected 90-day MRR at risk ${score.loc[score.risk_tier == 'High', 'expected_mrr_at_risk_90d'].sum():,.0f}")


if __name__ == "__main__":
    main()
