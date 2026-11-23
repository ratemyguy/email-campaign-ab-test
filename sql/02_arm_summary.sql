-- Arm-level summary on the CLEAN file, run by src/sql_checks.py (DuckDB).
-- Window functions put each arm's share of customers and its lift over the
-- no-email control on the same row. src/sql_checks.py cross-checks every
-- value against the pandas pipeline.

WITH by_arm AS (
    SELECT
        arm,
        CASE arm WHEN 'Control' THEN 1 WHEN 'Mens' THEN 2 ELSE 3 END AS arm_order,
        COUNT(*)                                  AS customers,
        SUM(visit)                                AS visits,
        SUM(conversion)                           AS buyers,
        SUM(spend)                                AS revenue,
        AVG(visit)                                AS visit_rate,
        AVG(conversion)                           AS conversion_rate,
        AVG(spend)                                AS spend_per_customer,
        SUM(spend) / NULLIF(SUM(conversion), 0)   AS spend_per_buyer
    FROM clean
    GROUP BY arm
)
SELECT
    arm,
    customers,
    customers / SUM(customers) OVER ()                                              AS share_of_customers,
    visits,
    buyers,
    ROUND(revenue, 2)                                                               AS revenue,
    visit_rate,
    conversion_rate,
    spend_per_customer,
    spend_per_buyer,
    visit_rate         - MAX(visit_rate)         FILTER (WHERE arm = 'Control') OVER () AS visit_rate_lift_vs_control,
    conversion_rate    - MAX(conversion_rate)    FILTER (WHERE arm = 'Control') OVER () AS conversion_rate_lift_vs_control,
    spend_per_customer - MAX(spend_per_customer) FILTER (WHERE arm = 'Control') OVER () AS spend_lift_vs_control
FROM by_arm
ORDER BY arm_order;
