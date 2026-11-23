"""Validate the raw CSV, clean it, add analysis columns and save a processed copy.

Cleaning is deliberately minimal (the file has no missing values):
  * "Surburban" (a typo in the source file) becomes "Suburban";
  * the treatment label "segment" becomes arm = Control / Mens / Womens;
  * pre-period subgroup columns are derived from existing columns with fixed,
    pre-stated cut-points (no outcome data is used to define them).

    python -I src/prepare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402

RAW_DTYPES = {
    "recency": "int64",
    "history_segment": "str",
    "history": "float64",
    "mens": "int64",
    "womens": "int64",
    "zip_code": "str",
    "newbie": "int64",
    "channel": "str",
    "segment": "str",
    "visit": "int64",
    "conversion": "int64",
    "spend": "float64",
}

# The dataset's own past-year spend bands; the four bands above $350 are merged
# so each bucket holds at least 12,000 customers.
HISTORY_BUCKET = {
    "1) $0 - $100": "Under $100",
    "2) $100 - $200": "$100-$199",
    "3) $200 - $350": "$200-$349",
    "4) $350 - $500": "$350+",
    "5) $500 - $750": "$350+",
    "6) $750 - $1,000": "$350+",
    "7) $1,000 +": "$350+",
}
SEGMENT_BOUNDS = {  # history_segment label -> [low, high) in dollars
    "1) $0 - $100": (0, 100),
    "2) $100 - $200": (100, 200),
    "3) $200 - $350": (200, 350),
    "4) $350 - $500": (350, 500),
    "5) $500 - $750": (500, 750),
    "6) $750 - $1,000": (750, 1000),
    "7) $1,000 +": (1000, np.inf),
}

CATEGORICALS = {
    "arm": cfg.ARMS,
    **{dim: levels for dim, levels in cfg.LEVEL_ORDER.items()},
}


def read_raw(path: Path = cfg.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=RAW_DTYPES)
    expected = list(RAW_DTYPES)
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if the raw file is not what the analysis assumes."""
    assert len(df) == 64_000, len(df)
    assert not df.isna().any().any(), "missing values"
    assert df["recency"].between(1, 12).all()
    assert (df["history"] > 0).all()
    for col in ["mens", "womens", "newbie", "visit", "conversion"]:
        assert set(df[col].unique()) <= {0, 1}, col
    assert set(df["zip_code"]) == {"Urban", "Surburban", "Rural"}
    assert set(df["channel"]) == {"Phone", "Web", "Multichannel"}
    assert set(df["segment"]) == set(cfg.ARM_FROM_SEGMENT)
    assert set(df["history_segment"]) == set(SEGMENT_BOUNDS)
    # spend > 0 exactly when the customer converted; a buyer must have visited
    assert ((df["spend"] > 0) == (df["conversion"] == 1)).all()
    assert not ((df["conversion"] == 1) & (df["visit"] == 0)).any()
    # every customer bought men's or women's merchandise (or both) last year
    assert ((df["mens"] == 1) | (df["womens"] == 1)).all()
    lo = df["history_segment"].map(lambda s: SEGMENT_BOUNDS[s][0])
    hi = df["history_segment"].map(lambda s: SEGMENT_BOUNDS[s][1])
    assert ((df["history"] >= lo) & (df["history"] < hi)).all(), "history outside its band"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"row_id": np.arange(1, len(df) + 1)})
    out["arm"] = df["segment"].map(cfg.ARM_FROM_SEGMENT)
    out["segment"] = df["segment"]
    out["recency"] = df["recency"]
    out["recency_bucket"] = pd.cut(
        df["recency"], bins=[0, 3, 6, 9, 12], labels=cfg.LEVEL_ORDER["recency_bucket"]
    ).astype(str)
    out["history"] = df["history"]
    out["history_segment"] = df["history_segment"]
    out["history_bucket"] = df["history_segment"].map(HISTORY_BUCKET)
    out["mens"] = df["mens"]
    out["womens"] = df["womens"]
    out["purchase_history"] = np.select(
        [(df["mens"] == 1) & (df["womens"] == 0), (df["mens"] == 0) & (df["womens"] == 1)],
        ["Mens only", "Womens only"],
        default="Both",
    )
    out["newbie"] = df["newbie"]
    out["customer_type"] = np.where(df["newbie"] == 1, "New customer", "Existing customer")
    out["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    out["channel"] = df["channel"]
    for col in ["visit", "conversion", "spend"]:
        out[col] = df[col]
    assert out.notna().all().all()
    return out


def load_clean(path: Path = cfg.PROCESSED_CSV) -> pd.DataFrame:
    """Load the processed file with ordered categoricals (used by every step)."""
    df = pd.read_csv(path)
    df["log_history"] = np.log(df["history"])
    for col, levels in CATEGORICALS.items():
        df[col] = pd.Categorical(df[col].astype(str), categories=levels, ordered=True)
        assert df[col].notna().all(), col
    return df


def main() -> pd.DataFrame:
    raw = read_raw()
    validate(raw)
    df = clean(raw)
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.PROCESSED_CSV, index=False)
    print(
        f"Processed data: {cfg.PROCESSED_CSV.relative_to(cfg.ROOT)} "
        f"({len(df):,} rows, {df.shape[1]} columns, {cfg.PROCESSED_CSV.stat().st_size / 1e6:.1f} MB)"
    )
    return df


if __name__ == "__main__":
    main()
