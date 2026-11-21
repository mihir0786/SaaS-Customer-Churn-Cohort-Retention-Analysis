-- 02_cohort_retention.sql
-- (a) cohort_retention: monthly acquisition cohort x months since signup (0..12)
--     with logo retention and revenue retention, the classic triangle.
-- (b) retention_curves: one 12-month curve per segment (overall, plan, channel,
--     company size, billing cycle), averaged only over cohorts old enough to be
--     observed at that age, so young cohorts never drag a curve down.

CREATE OR REPLACE TABLE cohort_retention AS
WITH cohort_activity AS (
    SELECT
        cohort_month,
        months_since_signup,
        COUNT(DISTINCT customer_id) AS active_customers,
        SUM(mrr)                    AS mrr
    FROM customer_months
    GROUP BY cohort_month, months_since_signup
),
with_base AS (
    SELECT
        *,
        FIRST_VALUE(active_customers) OVER c AS cohort_size,
        FIRST_VALUE(mrr)              OVER c AS starting_mrr,
        LAG(active_customers)         OVER c AS prev_active
    FROM cohort_activity
    WINDOW c AS (PARTITION BY cohort_month ORDER BY months_since_signup)
)
SELECT
    cohort_month,
    months_since_signup,
    cohort_size,
    active_customers,
    COALESCE(prev_active - active_customers, 0)         AS customers_lost,
    ROUND(active_customers / cohort_size, 4)            AS logo_retention,
    ROUND(mrr / starting_mrr, 4)                        AS revenue_retention,
    ROUND(mrr, 2)                                       AS mrr
FROM with_base
WHERE months_since_signup <= 12
ORDER BY cohort_month, months_since_signup;


CREATE OR REPLACE TABLE retention_curves AS
WITH last_month AS (
    SELECT MAX(month) AS max_month FROM customer_months
),
cohorts AS (       -- every customer's cohort and segments, even if they churned in month 0
    SELECT
        customer_id,
        CAST(DATE_TRUNC('month', signup_date) AS DATE) AS cohort_month,
        plan, acquisition_channel, company_size, billing_cycle
    FROM customers
),
ages AS (SELECT UNNEST(RANGE(0, 13)) AS months_since_signup),
eligible AS (      -- customer x age, only where that age has been observed
    SELECT c.*, a.months_since_signup
    FROM cohorts c
    CROSS JOIN ages a
    CROSS JOIN last_month l
    WHERE c.cohort_month + TO_MONTHS(a.months_since_signup) <= l.max_month
),
status AS (
    SELECT
        e.*,
        (cm.customer_id IS NOT NULL)::INT AS is_active
    FROM eligible e
    LEFT JOIN customer_months cm
           ON cm.customer_id = e.customer_id
          AND cm.months_since_signup = e.months_since_signup
),
unpivoted AS (
    SELECT 'Overall' AS dimension, 'All customers' AS segment, months_since_signup, is_active FROM status
    UNION ALL SELECT 'Plan',                plan,                months_since_signup, is_active FROM status
    UNION ALL SELECT 'Acquisition channel', acquisition_channel, months_since_signup, is_active FROM status
    UNION ALL SELECT 'Company size',        company_size,        months_since_signup, is_active FROM status
    UNION ALL SELECT 'Billing cycle',       billing_cycle,       months_since_signup, is_active FROM status
)
SELECT
    dimension,
    segment,
    months_since_signup,
    COUNT(*)                         AS customers_observed,
    SUM(is_active)                   AS customers_active,
    ROUND(AVG(is_active), 4)         AS logo_retention
FROM unpivoted
GROUP BY dimension, segment, months_since_signup
ORDER BY dimension, segment, months_since_signup;
