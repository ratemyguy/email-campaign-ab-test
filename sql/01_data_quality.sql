-- Data-quality checks on the RAW file, run by src/sql_checks.py (DuckDB).
-- The view `raw` is the downloaded CSV exactly as published (64,000 rows).
-- One output row per check: what was checked, the count/value, and a note.

WITH
dupes AS (                      -- rows that exactly repeat another row (all 12 columns)
    SELECT *, COUNT(*) OVER (PARTITION BY recency, history_segment, history, mens, womens,
                                          zip_code, newbie, channel, segment, visit,
                                          conversion, spend) AS copies,
           ROW_NUMBER() OVER (PARTITION BY recency, history_segment, history, mens, womens,
                                          zip_code, newbie, channel, segment, visit,
                                          conversion, spend) AS copy_no
    FROM raw
),
buyers AS (SELECT spend FROM raw WHERE conversion = 1)
SELECT 'rows' AS check_name, COUNT(*)::DOUBLE AS value,
       'one row per customer; the file has no customer ID' AS note
FROM raw
UNION ALL
SELECT 'rows_with_any_missing_value',
       COUNT(*) FILTER (WHERE recency IS NULL OR history_segment IS NULL OR history IS NULL
                         OR mens IS NULL OR womens IS NULL OR zip_code IS NULL OR newbie IS NULL
                         OR channel IS NULL OR segment IS NULL OR visit IS NULL
                         OR conversion IS NULL OR spend IS NULL)::DOUBLE,
       'no imputation needed'
FROM raw
UNION ALL
SELECT 'exact_duplicate_rows', COUNT(*) FILTER (WHERE copy_no > 1)::DOUBLE,
       'rows identical to an earlier row in all 12 columns; kept (no ID to tell two look-alike customers apart)'
FROM dupes
UNION ALL
SELECT 'exact_duplicate_rows_with_history_29.99', COUNT(*) FILTER (WHERE copy_no > 1 AND history = 29.99)::DOUBLE,
       'most duplicates sit at the $29.99 past-year-spend floor, where identical profiles are expected'
FROM dupes
UNION ALL
SELECT 'customers_with_history_exactly_29.99', COUNT(*) FILTER (WHERE history = 29.99)::DOUBLE,
       'the minimum past-year spend in the file'
FROM raw
UNION ALL
SELECT 'history_min_dollars', MIN(history), 'past-year spend, dollars' FROM raw
UNION ALL
SELECT 'history_max_dollars', MAX(history), 'past-year spend, dollars' FROM raw
UNION ALL
SELECT 'history_outside_its_history_segment_band',
       COUNT(*) FILTER (WHERE NOT (
             (history_segment = '1) $0 - $100'     AND history <  100)
          OR (history_segment = '2) $100 - $200'   AND history >= 100  AND history < 200)
          OR (history_segment = '3) $200 - $350'   AND history >= 200  AND history < 350)
          OR (history_segment = '4) $350 - $500'   AND history >= 350  AND history < 500)
          OR (history_segment = '5) $500 - $750'   AND history >= 500  AND history < 750)
          OR (history_segment = '6) $750 - $1,000' AND history >= 750  AND history < 1000)
          OR (history_segment = '7) $1,000 +'      AND history >= 1000)))::DOUBLE,
       'history_segment is consistent with history'
FROM raw
UNION ALL
SELECT 'customers_buying_neither_mens_nor_womens', COUNT(*) FILTER (WHERE mens = 0 AND womens = 0)::DOUBLE,
       'every customer bought men''s or women''s merchandise (or both) last year'
FROM raw
UNION ALL
SELECT 'customers_buying_both_mens_and_womens', COUNT(*) FILTER (WHERE mens = 1 AND womens = 1)::DOUBLE, ''
FROM raw
UNION ALL
SELECT 'zip_code_spelled_Surburban', COUNT(*) FILTER (WHERE zip_code = 'Surburban')::DOUBLE,
       'typo in the source; recoded to Suburban (no row says Suburban)'
FROM raw
UNION ALL
SELECT 'buyers_conversion_1', COUNT(*)::DOUBLE, 'customers who bought in the two weeks after the send' FROM buyers
UNION ALL
SELECT 'spend_positive_but_conversion_0', COUNT(*) FILTER (WHERE spend > 0 AND conversion = 0)::DOUBLE, 'spend and conversion agree' FROM raw
UNION ALL
SELECT 'conversion_1_but_spend_0', COUNT(*) FILTER (WHERE spend = 0 AND conversion = 1)::DOUBLE, 'spend and conversion agree' FROM raw
UNION ALL
SELECT 'conversion_1_but_no_visit', COUNT(*) FILTER (WHERE conversion = 1 AND visit = 0)::DOUBLE, 'every buyer visited' FROM raw
UNION ALL
SELECT 'buyer_spend_min_dollars', MIN(spend), '' FROM buyers
UNION ALL
SELECT 'buyer_spend_median_dollars', MEDIAN(spend), '' FROM buyers
UNION ALL
SELECT 'buyer_spend_max_dollars', MAX(spend), 'no buyer spent more than this' FROM buyers
UNION ALL
SELECT 'buyers_at_max_spend', COUNT(*) FILTER (WHERE spend = (SELECT MAX(spend) FROM buyers))::DOUBLE,
       'several buyers sit exactly at the maximum: spend looks top-coded at $499'
FROM buyers
UNION ALL
SELECT 'buyers_at_29.99', COUNT(*) FILTER (WHERE spend = 29.99)::DOUBLE,
       'the most common order value'
FROM buyers
UNION ALL
SELECT 'highest_buyer_spend_below_max', MAX(spend) FILTER (WHERE spend < (SELECT MAX(spend) FROM buyers)),
       'gap between the 2nd-highest value and the maximum'
FROM buyers;
