-- 01_customer_months.sql
-- One row per customer per paid month, enriched with cohort, tenure and the
-- neighbouring months. Every later query builds on this table.
-- Dialect: DuckDB (Postgres-like). See README for BigQuery / Postgres notes.

CREATE OR REPLACE TABLE customer_months AS
WITH base AS (
    SELECT
        u.customer_id,
        u.month,
        u.mrr,
        u.seats,
        u.active_users,
        u.logins,
        u.features_used,
        u.support_tickets,
        u.payment_failed,
        u.nps_score,
        c.plan,
        c.company_size,
        c.region,
        c.acquisition_channel,
        c.billing_cycle,
        CAST(DATE_TRUNC('month', c.signup_date) AS DATE) AS cohort_month
    FROM monthly_usage AS u
    JOIN customers     AS c USING (customer_id)
)
SELECT
    *,
    DATE_DIFF('month', cohort_month, month)      AS months_since_signup,
    LAG(month)  OVER w                           AS prev_month,
    LEAD(month) OVER w                           AS next_month,
    LAG(mrr)    OVER w                           AS prev_mrr,
    ROW_NUMBER() OVER w                          AS paid_month_number
FROM base
WINDOW w AS (PARTITION BY customer_id ORDER BY month);
