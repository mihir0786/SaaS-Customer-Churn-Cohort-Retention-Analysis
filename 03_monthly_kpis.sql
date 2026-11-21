-- 03_monthly_kpis.sql
-- Company-level KPIs per calendar month:
--   active customers, MRR, ARPU, new / churned customers,
--   logo churn rate, gross MRR churn, net revenue retention,
--   and the MRR bridge (new + expansion - contraction - churned).
-- A customer churns in month M when they paid in M-1 and did not pay in M.

CREATE OR REPLACE TABLE monthly_kpis AS
WITH bounds AS (
    SELECT MIN(month) AS first_month, MAX(month) AS last_month FROM customer_months
),
activity AS (
    SELECT
        month,
        COUNT(*)                                                         AS active_customers,
        SUM(mrr)                                                         AS mrr,
        SUM(CASE WHEN prev_month IS NULL THEN 1 ELSE 0 END)              AS new_customers,
        SUM(CASE WHEN prev_month IS NULL THEN mrr ELSE 0 END)            AS new_mrr,
        SUM(CASE WHEN prev_month IS NOT NULL AND mrr > prev_mrr
                 THEN mrr - prev_mrr ELSE 0 END)                         AS expansion_mrr,
        SUM(CASE WHEN prev_month IS NOT NULL AND mrr < prev_mrr
                 THEN prev_mrr - mrr ELSE 0 END)                         AS contraction_mrr
    FROM customer_months
    GROUP BY month
),
churn AS (
    SELECT
        CAST(cm.month + INTERVAL 1 MONTH AS DATE)                        AS month,
        COUNT(*)                                                         AS churned_customers,
        SUM(cm.mrr)                                                      AS churned_mrr
    FROM customer_months cm, bounds b
    WHERE cm.month < b.last_month
      AND (cm.next_month IS NULL OR cm.next_month > cm.month + INTERVAL 1 MONTH)
    GROUP BY 1
),
joined AS (
    SELECT
        a.*,
        COALESCE(c.churned_customers, 0)                                 AS churned_customers,
        COALESCE(c.churned_mrr, 0)                                       AS churned_mrr,
        LAG(a.active_customers) OVER (ORDER BY a.month)                  AS start_customers,
        LAG(a.mrr)              OVER (ORDER BY a.month)                  AS start_mrr
    FROM activity a
    LEFT JOIN churn c USING (month)
)
SELECT
    month,
    active_customers,
    ROUND(mrr, 2)                                                        AS mrr,
    ROUND(mrr / active_customers, 2)                                     AS arpu,
    new_customers,
    churned_customers,
    ROUND(churned_customers / start_customers, 4)                        AS logo_churn_rate,
    ROUND(AVG(churned_customers / start_customers)
          OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 4)
                                                                         AS logo_churn_rate_3m_avg,
    ROUND(new_mrr, 2)                                                    AS new_mrr,
    ROUND(expansion_mrr, 2)                                              AS expansion_mrr,
    ROUND(contraction_mrr, 2)                                            AS contraction_mrr,
    ROUND(churned_mrr, 2)                                                AS churned_mrr,
    ROUND(churned_mrr / start_mrr, 4)                                    AS gross_mrr_churn_rate,
    ROUND((start_mrr + expansion_mrr - contraction_mrr - churned_mrr) / start_mrr, 4)
                                                                         AS net_mrr_retention,
    ROUND(SUM(new_mrr - churned_mrr + expansion_mrr - contraction_mrr)
          OVER (ORDER BY month), 2)                                      AS cumulative_net_new_mrr
FROM joined
ORDER BY month;
