-- 04_segment_revenue_at_risk.sql
-- Segment = plan x company size (15 segments).
-- For each segment: current customers and MRR, ARPU, trailing-6-month logo and
-- MRR churn rates, and the MRR expected to be lost over the next 12 months if
-- that churn rate holds:  mrr_at_risk_12m = current_mrr * (1 - (1 - monthly_mrr_churn)^12)
-- "High-value" = segment ARPU at or above the company-wide ARPU.

CREATE OR REPLACE TABLE segment_revenue_at_risk AS
WITH bounds AS (
    SELECT MAX(month) AS last_month FROM customer_months
),
current_book AS (                       -- the book of business in the latest month
    SELECT
        plan,
        company_size,
        COUNT(*)  AS active_customers,
        SUM(mrr)  AS current_mrr
    FROM customer_months, bounds
    WHERE month = last_month
    GROUP BY plan, company_size
),
recent_months AS (                          -- months M-6 .. M-1 as starting points
    SELECT
        cm.plan,
        cm.company_size,
        cm.month,
        cm.mrr,
        (cm.next_month IS NULL OR cm.next_month > cm.month + INTERVAL 1 MONTH)::INT AS churned_next_month
    FROM customer_months cm, bounds b
    WHERE cm.month >= CAST(b.last_month - INTERVAL 6 MONTH AS DATE)
      AND cm.month <  b.last_month
),
churn_rates AS (
    SELECT
        plan,
        company_size,
        SUM(churned_next_month) / COUNT(*)                                   AS monthly_logo_churn,
        SUM(CASE WHEN churned_next_month = 1 THEN mrr ELSE 0 END) / SUM(mrr) AS monthly_mrr_churn
    FROM recent_months
    GROUP BY plan, company_size
),
scored AS (
    SELECT
        b.plan,
        b.company_size,
        b.active_customers,
        b.current_mrr,
        b.current_mrr / b.active_customers                                   AS arpu,
        r.monthly_logo_churn,
        r.monthly_mrr_churn,
        b.current_mrr * (1 - POWER(1 - r.monthly_mrr_churn, 12))            AS mrr_at_risk_12m
    FROM current_book b
    JOIN churn_rates  r USING (plan, company_size)
),
company AS (
    SELECT SUM(current_mrr) / SUM(active_customers) AS company_arpu,
           SUM(mrr_at_risk_12m)                     AS total_at_risk
    FROM scored
)
SELECT
    s.plan,
    s.company_size,
    s.plan || ' / ' || s.company_size                                        AS segment,
    s.active_customers,
    ROUND(s.current_mrr, 2)                                                  AS current_mrr,
    ROUND(s.arpu, 2)                                                         AS arpu,
    (s.arpu >= c.company_arpu)                                               AS is_high_value,
    ROUND(s.monthly_logo_churn, 4)                                           AS monthly_logo_churn,
    ROUND(s.monthly_mrr_churn, 4)                                            AS monthly_mrr_churn,
    ROUND(s.mrr_at_risk_12m, 2)                                              AS mrr_at_risk_12m,
    ROUND(s.mrr_at_risk_12m / c.total_at_risk, 4)                            AS share_of_total_risk,
    RANK() OVER (ORDER BY s.mrr_at_risk_12m DESC)                            AS risk_rank_all,
    CASE WHEN s.arpu >= c.company_arpu
         THEN RANK() OVER (PARTITION BY (s.arpu >= c.company_arpu)
                           ORDER BY s.mrr_at_risk_12m DESC) END              AS risk_rank_high_value,
    NTILE(3) OVER (ORDER BY s.monthly_mrr_churn DESC)                        AS churn_tercile   -- 1 = highest churn
FROM scored s
CROSS JOIN company c
ORDER BY s.mrr_at_risk_12m DESC;
