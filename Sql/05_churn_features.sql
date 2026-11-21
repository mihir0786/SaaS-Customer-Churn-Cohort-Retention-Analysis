-- 05_churn_features.sql
-- Model-ready feature table for churn prediction.
-- One row per (snapshot month, customer paying in that month).
--   Features: account attributes + behaviour over the 3 months ending at the snapshot.
--   Label   : churned_next_90d = 1 if the customer is NOT paying 3 months later.
--             NULL when the future is not observed yet (the scoring snapshot).
-- Using several quarterly snapshots gives the model more examples and lets us
-- test it "out of time": train on older snapshots, test on the newest labelled one.

CREATE OR REPLACE TABLE churn_features AS
WITH bounds AS (
    SELECT MAX(month) AS last_month FROM customer_months
),
snapshots AS (
    SELECT UNNEST([
        DATE '2024-03-01', DATE '2024-06-01', DATE '2024-09-01', DATE '2024-12-01',
        DATE '2025-03-01', DATE '2025-06-01', DATE '2025-09-01', DATE '2025-12-01'
    ]) AS snapshot_month
),
anchor AS (                                  -- customers paying in the snapshot month
    SELECT s.snapshot_month, cm.*
    FROM snapshots s
    JOIN customer_months cm ON cm.month = s.snapshot_month
),
window_rows AS (                             -- the snapshot month and the 2 months before it
    SELECT
        a.snapshot_month,
        a.customer_id,
        w.month,
        w.mrr,
        w.active_users,
        w.seats,
        w.logins,
        w.features_used,
        w.support_tickets,
        w.payment_failed,
        w.active_users / w.seats                       AS seat_utilization,
        w.logins / NULLIF(w.active_users, 0)           AS logins_per_user
    FROM anchor a
    JOIN customer_months w
      ON w.customer_id = a.customer_id
     AND w.month BETWEEN CAST(a.snapshot_month - INTERVAL 2 MONTH AS DATE) AND a.snapshot_month
),
behaviour AS (
    SELECT
        snapshot_month,
        customer_id,
        COUNT(*)                                       AS months_in_window,
        AVG(seat_utilization)                          AS seat_utilization_3m,
        SUM(logins) / NULLIF(SUM(active_users), 0)     AS logins_per_user_3m,
        AVG(features_used)                             AS features_used_3m,
        SUM(support_tickets)                           AS support_tickets_3m,
        SUM(payment_failed)                            AS payment_failures_3m,
        ARG_MIN(mrr, month)                            AS mrr_window_start,
        ARG_MIN(seat_utilization, month)               AS utilization_window_start,
        ARG_MIN(logins_per_user, month)                AS logins_per_user_window_start,
        ARG_MIN(features_used, month)                  AS features_window_start
    FROM window_rows
    GROUP BY snapshot_month, customer_id
),
latest_nps AS (                              -- most recent survey answer up to the snapshot
    SELECT
        a.snapshot_month,
        a.customer_id,
        ARG_MAX(h.nps_score, h.month)                  AS last_nps
    FROM anchor a
    JOIN customer_months h
      ON h.customer_id = a.customer_id
     AND h.month <= a.snapshot_month
     AND h.nps_score IS NOT NULL
    GROUP BY a.snapshot_month, a.customer_id
),
labels AS (
    SELECT
        a.snapshot_month,
        a.customer_id,
        CASE
            WHEN a.snapshot_month + INTERVAL 3 MONTH > b.last_month THEN NULL
            WHEN f.customer_id IS NULL THEN 1
            ELSE 0
        END                                            AS churned_next_90d
    FROM anchor a
    CROSS JOIN bounds b
    LEFT JOIN customer_months f
           ON f.customer_id = a.customer_id
          AND f.month = CAST(a.snapshot_month + INTERVAL 3 MONTH AS DATE)
)
SELECT
    a.snapshot_month,
    a.customer_id,
    a.plan,
    a.company_size,
    a.region,
    a.acquisition_channel,
    a.billing_cycle,
    a.months_since_signup                                           AS tenure_months,
    a.mrr,
    a.seats,
    CASE WHEN a.billing_cycle = 'monthly' OR a.months_since_signup % 12 >= 9
         THEN 1 ELSE 0 END                                          AS can_cancel_in_window,
    b.months_in_window,
    ROUND(b.seat_utilization_3m, 4)                                 AS seat_utilization_3m,
    ROUND(a.active_users / a.seats - b.utilization_window_start, 4) AS seat_utilization_change,
    ROUND(b.logins_per_user_3m, 3)                                  AS logins_per_user_3m,
    ROUND(a.logins / NULLIF(a.active_users, 0)
          - b.logins_per_user_window_start, 3)                      AS logins_per_user_change,
    ROUND(b.features_used_3m, 3)                                    AS features_used_3m,
    a.features_used - b.features_window_start                       AS features_used_change,
    b.support_tickets_3m,
    b.payment_failures_3m,
    ROUND((a.mrr - b.mrr_window_start) / b.mrr_window_start, 4)     AS mrr_change_pct_3m,
    n.last_nps,
    l.churned_next_90d
FROM anchor a
JOIN behaviour   b USING (snapshot_month, customer_id)
JOIN labels      l USING (snapshot_month, customer_id)
LEFT JOIN latest_nps n USING (snapshot_month, customer_id)
ORDER BY a.snapshot_month, a.customer_id;
