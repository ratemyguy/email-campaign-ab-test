"""Step 4 - Regression adjustment and CUPED: same answer, smaller standard errors?

For each outcome and each email vs. no email, four estimators of the same effect:
  1. Difference in means         OLS of outcome on arm dummies (HC2 robust SE; equals Welch's SE)
  2. CUPED                       outcome minus theta x (past-year spend - its mean), theta = cov/var
  3. OLS + covariates (ANCOVA)   arm dummies + every pre-email covariate, HC2 robust SE
  4. Lin (2013)                  as 3, plus arm x centered-covariate interactions (robustness)
Randomization makes all four unbiased; covariates only help by soaking up outcome
variance, so the SE shrinks by sqrt(1 - R^2) at best.

Outputs: results/regression_adjustment.csv, results/regression_coefficients.csv

    python -I src/s04_regression_adjustment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

import config as cfg  # noqa: E402
import prepare  # noqa: E402
import stats_utils as su  # noqa: E402

COV_TYPE = "HC2"


def covariate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The pre-email covariates as numeric columns (reference levels dropped)."""
    X = pd.DataFrame(
        {
            "recency": df["recency"].astype(float),
            "log_history": df["log_history"].astype(float),
            "newbie": df["newbie"].astype(float),
        }
    )
    for col, ref in [("purchase_history", "Mens only"), ("zip_code", "Urban"), ("channel", "Phone")]:
        for level in cfg.LEVEL_ORDER[col]:
            if level != ref:
                X[f"{col}={level}"] = (df[col] == level).astype(float)
    return X


def lin_estimator(df: pd.DataFrame, y: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = covariate_matrix(df)
    Xc = X - X.mean()
    design = pd.DataFrame({"const": 1.0}, index=df.index)
    for t in cfg.TREATMENTS:
        design[f"arm={t}"] = (df["arm"] == t).astype(float)
    design = pd.concat([design, Xc], axis=1)
    for t in cfg.TREATMENTS:
        for c in Xc.columns:
            design[f"arm={t}:{c}"] = design[f"arm={t}"] * Xc[c]
    return sm.OLS(df[y].astype(float), design).fit(cov_type=COV_TYPE)


def cuped(df: pd.DataFrame, y: str) -> tuple[pd.Series, float, float]:
    """CUPED with past-year spend ('history') as the pre-period covariate."""
    x = df["history"].astype(float)
    yy = df[y].astype(float)
    theta = np.cov(yy, x, ddof=1)[0, 1] / x.var(ddof=1)
    rho = np.corrcoef(yy, x)[0, 1]
    return yy - theta * (x - x.mean()), theta, rho


def main() -> pd.DataFrame:
    df = prepare.load_clean()
    rows, coef_rows = [], []
    for y in cfg.METRICS:
        unadj = smf.ols(f"{y} ~ C(arm)", data=df).fit(cov_type=COV_TYPE)
        adj = smf.ols(f"{y} ~ C(arm) + {cfg.COVARIATE_FORMULA}", data=df).fit(cov_type=COV_TYPE)
        lin = lin_estimator(df, y)
        y_cuped, theta, rho = cuped(df, y)
        for name, res in [("ols_covariates", adj)]:
            coefs = pd.DataFrame(
                {"outcome": y, "model": name, "term": res.params.index, "coef": res.params.to_numpy(),
                 "robust_se": res.bse.to_numpy(), "p_value": res.pvalues.to_numpy()}
            )
            coef_rows.append(coefs)
        for t in cfg.TREATMENTS:
            term = f"C(arm)[T.{t}]"
            # 1. difference in means: HC2 robust SE on arm dummies == Welch SE
            w = su.welch_ttest(df.loc[df["arm"] == t, y], df.loc[df["arm"] == "Control", y])
            assert np.isclose(unadj.params[term], w["abs_lift"]) and np.isclose(unadj.bse[term], w["se"], rtol=1e-6)
            c = su.diff_in_means(y_cuped[df["arm"] == t], y_cuped[df["arm"] == "Control"])
            # model_r2: R^2 of the fitted regression; cuped_corr: corr(outcome, past-year spend)
            # share of the outcome variance left after the arm dummies that the covariates explain
            partial = (adj.rsquared - unadj.rsquared) / (1 - unadj.rsquared)
            estimates = [
                ("1. Difference in means", unadj.params[term], unadj.bse[term], unadj.rsquared, np.nan, ""),
                ("2. CUPED (past-year spend)", c["abs_lift"], c["se"], np.nan, rho,
                 f"theta = {theta:.6f}; variance reduction is about corr^2"),
                ("3. OLS + pre-email covariates", adj.params[term], adj.bse[term], adj.rsquared, np.nan,
                 "covariates: " + cfg.COVARIATE_FORMULA),
                ("4. Lin (2013) interacted OLS", lin.params[f"arm={t}"], lin.bse[f"arm={t}"], lin.rsquared, np.nan,
                 "OLS with arm x centered-covariate interactions"),
            ]
            base_se = unadj.bse[term]
            for method, est, se, r2, corr, note in estimates:
                rows.append(
                    {
                        "outcome": y, "comparison": f"{t} vs Control", "method": method, "estimate": est,
                        "robust_se": se, "ci_low": est - su.Z * se, "ci_high": est + su.Z * se,
                        "model_r2": r2, "cuped_corr": corr,
                        "covariate_partial_r2": partial if method.startswith("3.") else np.nan,
                        "se_change_vs_unadjusted": se / base_se - 1,
                        "variance_reduction": 1 - (se / base_se) ** 2,
                        "equivalent_extra_sample": (base_se / se) ** 2 - 1,
                        "estimate_change_vs_unadjusted": est - unadj.params[term], "note": note,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(cfg.RESULTS / "regression_adjustment.csv", index=False)
    pd.concat(coef_rows).to_csv(cfg.RESULTS / "regression_coefficients.csv", index=False)
    a = out[out["method"].str.startswith("3.")].set_index(["outcome", "comparison"])
    print(
        "Step 4 regression adjustment (OLS+covariates, SE change vs unadjusted): "
        + "; ".join(f"{o} {c}: {r.se_change_vs_unadjusted:+.2%}" for (o, c), r in a.iterrows())
    )
    return out


if __name__ == "__main__":
    main()
