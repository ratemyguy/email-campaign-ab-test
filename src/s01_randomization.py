"""Step 1 - Randomization check.

* Sample-ratio-mismatch (SRM) test: are the group sizes consistent with 1/3 each?
* Covariate balance: means by arm, standardized mean differences (SMD), a per-
  covariate test across the three arms, and one joint test (multinomial logit,
  likelihood-ratio test) that pre-email traits do not predict the arm.

Outputs: results/srm_check.csv, results/balance_table.csv,
         results/balance_joint_test.csv, images/balance_smd.png

    python -I src/s01_randomization.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

import config as cfg  # noqa: E402
import plots  # noqa: E402
import prepare  # noqa: E402
import stats_utils as su  # noqa: E402

# (label, column or (column, level), kind)
BALANCE_ROWS = [
    ("Months since last purchase", "recency", "numeric"),
    ("Past-year spend ($)", "history", "numeric"),
    ("Bought men's only", ("purchase_history", "Mens only"), "share"),
    ("Bought women's only", ("purchase_history", "Womens only"), "share"),
    ("Bought both", ("purchase_history", "Both"), "share"),
    ("New customer (last 12 months)", "newbie", "share"),
    ("Urban ZIP", ("zip_code", "Urban"), "share"),
    ("Suburban ZIP", ("zip_code", "Suburban"), "share"),
    ("Rural ZIP", ("zip_code", "Rural"), "share"),
    ("Channel: phone", ("channel", "Phone"), "share"),
    ("Channel: web", ("channel", "Web"), "share"),
    ("Channel: multichannel", ("channel", "Multichannel"), "share"),
]


def _column(df: pd.DataFrame, spec) -> pd.Series:
    if isinstance(spec, tuple):
        col, level = spec
        return (df[col] == level).astype(float)
    return df[spec].astype(float)


def srm_check(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["arm"].value_counts().reindex(cfg.ARMS)
    expected = np.full(3, len(df) / 3)
    chi = stats.chisquare(counts.to_numpy(), f_exp=expected)
    return pd.DataFrame(
        {
            "arm": cfg.ARMS,
            "customers": counts.to_numpy(),
            "expected_customers": expected,
            "share": counts.to_numpy() / len(df),
            "chi2_stat": chi.statistic,
            "chi2_df": 2,
            "p_value": chi.pvalue,
        }
    )


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, spec, kind in BALANCE_ROWS:
        x = _column(df, spec)
        by_arm = {arm: x[df["arm"] == arm] for arm in cfg.ARMS}
        if kind == "numeric":
            test = "one-way ANOVA"
            p = stats.f_oneway(*by_arm.values()).pvalue
        else:
            test = "chi-square (2x3)"
            table = pd.crosstab(x, df["arm"])
            p = stats.chi2_contingency(table.to_numpy(), correction=False).pvalue
        rows.append(
            {
                "covariate": label,
                "kind": kind,
                "mean_control": by_arm["Control"].mean(),
                "mean_mens": by_arm["Mens"].mean(),
                "mean_womens": by_arm["Womens"].mean(),
                "smd_mens_vs_control": su.smd(by_arm["Mens"], by_arm["Control"]),
                "smd_womens_vs_control": su.smd(by_arm["Womens"], by_arm["Control"]),
                "smd_mens_vs_womens": su.smd(by_arm["Mens"], by_arm["Womens"]),
                "test_across_3_arms": test,
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    out["max_abs_smd"] = out[["smd_mens_vs_control", "smd_womens_vs_control", "smd_mens_vs_womens"]].abs().max(axis=1)
    return out


def joint_test(df: pd.DataFrame) -> pd.DataFrame:
    """Multinomial logit of arm on every pre-email covariate; LR test vs. intercept only."""
    d = df.assign(arm_code=df["arm"].cat.codes)
    model = smf.mnlogit("arm_code ~ " + cfg.COVARIATE_FORMULA, data=d).fit(disp=False, maxiter=200)
    return pd.DataFrame(
        [
            {
                "model": "multinomial logit: arm ~ " + cfg.COVARIATE_FORMULA,
                "lr_chi2": float(model.llr),
                "df": int(model.df_model),
                "p_value": float(model.llr_pvalue),
                "pseudo_r2": float(model.prsquared),
                "converged": bool(model.mle_retvals["converged"]),
            }
        ]
    )


def chart(bal: pd.DataFrame, srm: pd.DataFrame, path: Path) -> Path:
    fig, ax = plots.new_figure()
    y = np.arange(len(bal))[::-1]
    ax.axvspan(-0.1, 0.1, color="#F2F2F2", zorder=0)
    ax.axvline(0, color="#444444", lw=0.8)
    ax.scatter(bal["smd_mens_vs_control"], y + 0.12, color=cfg.NAVY, s=22, label="Mens email vs. no email", zorder=3)
    ax.scatter(bal["smd_womens_vs_control"], y - 0.12, color=cfg.TEAL, s=22, marker="D", label="Womens email vs. no email", zorder=3)
    ax.set_yticks(y, bal["covariate"])
    ax.set_xlim(-0.12, 0.12)
    ax.set_xlabel("Standardized mean difference (treatment minus control, in standard deviations)")
    ax.text(0.098, y.max() + 0.55, "|SMD| < 0.1 = balanced", ha="right", va="bottom", fontsize=7, color="#555555")
    ax.legend(loc="upper left")
    p_srm = srm["p_value"].iloc[0]
    bound = math.ceil(bal["max_abs_smd"].max() * 100) / 100  # round up: a true upper bound
    title = f"Randomization held: the groups looked alike before the email (every |SMD| < {bound:.2f})"
    subtitle = (
        f"Group sizes match 1/3 each (chi-square p = {p_srm:.2f}); 12 pre-email traits, all within the "
        "usual |SMD| < 0.1 balance threshold"
    )
    fig.subplots_adjust(left=0.27, right=0.98)
    return plots.finish(fig, title, path, subtitle=subtitle, top=0.86, bottom=0.15)


def main() -> dict:
    df = prepare.load_clean()
    srm = srm_check(df)
    bal = balance_table(df)
    joint = joint_test(df)
    srm.to_csv(cfg.RESULTS / "srm_check.csv", index=False)
    bal.to_csv(cfg.RESULTS / "balance_table.csv", index=False)
    joint.to_csv(cfg.RESULTS / "balance_joint_test.csv", index=False)
    chart(bal, srm, cfg.IMAGES / "balance_smd.png")
    print(
        f"Step 1 randomization: SRM chi2 p = {srm['p_value'].iloc[0]:.3f}; max |SMD| = {bal['max_abs_smd'].max():.4f}; "
        f"joint LR test p = {joint['p_value'].iloc[0]:.3f}"
    )
    return {"srm": srm, "balance": bal, "joint": joint}


if __name__ == "__main__":
    main()
