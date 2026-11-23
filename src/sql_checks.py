"""Run the SQL data-quality checks and the arm summary in DuckDB.

Writes results/data_quality.csv and results/arm_summary.csv, then checks that
the SQL arm summary matches the pandas pipeline to floating-point precision.

    python -I src/sql_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402
import prepare  # noqa: E402


def _csv_view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    literal = str(path).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv('{literal}', header = true)")


def run_sql(name: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute((cfg.SQL_DIR / name).read_text()).df()


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg.RESULTS.mkdir(exist_ok=True)
    con = duckdb.connect()
    _csv_view(con, "raw", cfg.RAW_CSV)
    _csv_view(con, "clean", cfg.PROCESSED_CSV)

    dq = run_sql("01_data_quality.sql", con)
    dq.to_csv(cfg.RESULTS / "data_quality.csv", index=False)

    arm = run_sql("02_arm_summary.sql", con)
    arm.to_csv(cfg.RESULTS / "arm_summary.csv", index=False)
    con.close()

    # Cross-check the SQL summary against pandas.
    df = prepare.load_clean()
    g = df.groupby("arm", observed=True)
    pd_summary = pd.DataFrame(
        {
            "customers": g.size(),
            "visits": g["visit"].sum(),
            "buyers": g["conversion"].sum(),
            "visit_rate": g["visit"].mean(),
            "conversion_rate": g["conversion"].mean(),
            "spend_per_customer": g["spend"].mean(),
        }
    )
    sql_summary = arm.set_index("arm").loc[cfg.ARMS]
    for col in pd_summary.columns:
        assert np.allclose(sql_summary[col].to_numpy(float), pd_summary.loc[cfg.ARMS, col].to_numpy(float)), col
    dqv = dq.set_index("check_name")["value"]
    assert dqv["rows"] == 64_000 and dqv["rows_with_any_missing_value"] == 0
    print("SQL checks OK: data_quality.csv, arm_summary.csv (SQL and pandas agree)")
    return dq, arm


if __name__ == "__main__":
    main()
