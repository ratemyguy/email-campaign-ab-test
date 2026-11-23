"""Shared paths, constants and chart style for the email A/B/n test project.

Every script imports from here, so a constant (seed, number of bootstrap
resamples, colour) is defined exactly once.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SQL_DIR = ROOT / "sql"
RAW_DIR = ROOT / "data" / "raw" / "hillstrom"
RAW_CSV = RAW_DIR / "hillstrom_2008.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_CSV = PROCESSED_DIR / "hillstrom_clean.csv"
RESULTS = ROOT / "results"
IMAGES = ROOT / "images"
NOTEBOOK = ROOT / "notebooks" / "ab_test_analysis.ipynb"
README = ROOT / "README.md"

# Source: Kevin Hillstrom, "The MineThatData E-Mail Analytics And Data Mining
# Challenge", MineThatData blog, 2008-03-20.
DATA_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
POST_URL = "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html"
# SHA-256 of the file downloaded on 2026-11-23 (3,964,977 bytes, Last-Modified 2008-03-21).
DATA_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"

# One seed for everything random (bootstrap, train/holdout split): the date the
# challenge was published.
SEED = 20080320
ALPHA = 0.05
POWER = 0.80
N_BOOT = 10_000          # bootstrap resamples for the main spend effects
N_BOOT_POLICY = 5_000    # bootstrap resamples for the holdout policy comparison

# Arms. The raw file calls them "Mens E-Mail", "Womens E-Mail", "No E-Mail".
ARM_FROM_SEGMENT = {
    "No E-Mail": "Control",
    "Mens E-Mail": "Mens",
    "Womens E-Mail": "Womens",
}
ARMS = ["Control", "Mens", "Womens"]
TREATMENTS = ["Mens", "Womens"]
ARM_LABEL = {"Control": "No email (control)", "Mens": "Mens email", "Womens": "Womens email"}

# Outcomes measured in the two weeks after the send.
METRICS = ["visit", "conversion", "spend"]
METRIC_LABEL = {
    "visit": "Visited the website",
    "conversion": "Bought (conversion)",
    "spend": "Spend per customer ($)",
}

# Comparisons tested in the main-effects family (Holm-corrected together).
COMPARISONS = [("Mens", "Control"), ("Womens", "Control"), ("Mens", "Womens")]

# Pre-period covariates (all measured before the email was sent).
COVARIATE_FORMULA = (
    "recency + log_history + C(purchase_history) + newbie + C(zip_code) + C(channel)"
)

# Subgroup dimensions for the heterogeneity analysis: the customer attributes the
# challenge describes, with bucket cut-points set by group size, not by outcomes.
SUBGROUP_DIMS = {
    "purchase_history": "Past purchases",
    "recency_bucket": "Months since last purchase",
    "history_bucket": "Past-year spend",
    "channel": "Purchase channel",
    "customer_type": "New vs existing",
    "zip_code": "ZIP code type",
}
LEVEL_ORDER = {
    "purchase_history": ["Mens only", "Womens only", "Both"],
    "recency_bucket": ["1-3", "4-6", "7-9", "10-12"],
    "history_bucket": ["Under $100", "$100-$199", "$200-$349", "$350+"],
    "channel": ["Phone", "Web", "Multichannel"],
    "customer_type": ["Existing customer", "New customer"],
    "zip_code": ["Urban", "Suburban", "Rural"],
}

# Chart style (STYLE.md).
NAVY, TEAL, YELLOW, ORANGE, GRAY = "#1F3A5F", "#2A9D8F", "#E9C46A", "#E76F51", "#8D99AE"
ARM_COLOR = {"Control": GRAY, "Mens": NAVY, "Womens": TEAL}
DPI = 150
FIGSIZE = (8.0, 4.67)  # 1200 x 700 px at 150 dpi
SOURCE_NOTE = (
    "Source: Kevin Hillstrom, MineThatData E-Mail Analytics Challenge (2008), "
    "64,000 customers randomly split into 3 arms."
)
