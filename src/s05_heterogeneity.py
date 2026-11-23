"""Step 5 - Heterogeneous effects: did the emails work differently for different customers?

Six subgroup dimensions come from the customer attributes the challenge describes
(config.SUBGROUP_DIMS: past purchases, recency, past-year spend, channel, new vs existing,
ZIP type), with cut-points set by group size, not by outcomes. For every level:
lift of each email vs. no email, and Mens vs. Womens, on all three metrics,
with 95% CIs and the realized MDE (2.80 x SE) so a null can be read as
"small effect" or "too few customers".

"Does the effect differ by subgroup?" is tested once per dimension x comparison x
metric with Cochran's Q (a chi-square test that the subgroup lifts are equal),
Holm-corrected across all 54 such tests. The strongest pattern is then re-estimated
as an interaction regression (OLS, HC2 robust SEs); its Wald test reproduces Q exactly.

Outputs: results/subgroup_effects.csv, results/heterogeneity_tests.csv,
         results/interaction_model.csv, results/interaction_test.csv,
         images/subgroup_forest.png

    python -I src/s05_heterogeneity.py
"""
from __future__ import annotations

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

COMPARISONS = [("Mens", "Control"), ("Womens", "Control"), ("Mens", "Womens")]
SHORT_DIM_LABEL = {
    "purchase_history": "Past purchases",
    "recency_bucket": "Months since last buy",
    "history_bucket": "Past-year spend",
    "channel": "Channel",
    "customer_type": "New vs. existing",
    "zip_code": "ZIP type",
}


def subgroup_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim, dim_label in cfg.SUBGROUP_DIMS.items():
        for level in cfg.LEVEL_ORDER[dim]:
            cell = df[df[dim] == level]
            for t, c in COMPARISONS:
                for m in cfg.METRICS:
                    r = su.diff_in_means(cell.loc[cell["arm"] == t, m], cell.loc[cell["arm"] == c, m])
                    rows.append(
                        {
                            "dimension": dim, "dimension_label": dim_label, "level": level,
                            "customers_in_level": len(cell), "share_of_customers": len(cell) / len(df),
                            "comparison": f"{t} vs {c}", "metric": m,
                            "n_treatment": r["n_t"], "n_baseline": r["n_c"],
                            "mean_treatment": r["mean_t"], "mean_baseline": r["mean_c"],
                            "abs_lift": r["abs_lift"], "se": r["se"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                            "p_value": r["p_value"],
                            "rel_lift": r["mean_t"] / r["mean_c"] - 1 if r["mean_c"] > 0 else np.nan,
                            "mde_realized_abs": su.mde(r["se"]),
                            "buyers_treatment": int(cell.loc[cell["arm"] == t, "conversion"].sum()),
                            "buyers_baseline": int(cell.loc[cell["arm"] == c, "conversion"].sum()),
                        }
                    )
    out = pd.DataFrame(rows)
    out["ci_excludes_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out


def heterogeneity_tests(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dim, comp, m), g in sub.groupby(["dimension", "comparison", "metric"], sort=False):
        q = su.cochran_q(g["abs_lift"], g["se"])
        lo = g.loc[g["abs_lift"].idxmin()]
        hi = g.loc[g["abs_lift"].idxmax()]
        rows.append(
            {
                "dimension": dim, "comparison": comp, "metric": m, "levels": len(g),
                "q_stat": q["q_stat"], "df": q["q_df"], "p_value": q["q_p_value"],
                "lowest_level": lo["level"], "lowest_lift": lo["abs_lift"],
                "highest_level": hi["level"], "highest_lift": hi["abs_lift"],
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = su.holm(out["p_value"])
    out["significant_raw"] = out["p_value"] < cfg.ALPHA
    out["significant_holm"] = out["p_holm"] < cfg.ALPHA
    return out.sort_values("p_value").reset_index(drop=True)


def interaction_model(df: pd.DataFrame, het: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-estimate the strongest pattern as an interaction regression."""
    top = het.iloc[0]
    dim, m = top["dimension"], top["metric"]
    t, c = top["comparison"].split(" vs ")
    ref_level = cfg.LEVEL_ORDER[dim][0]
    formula = (
        f"{m} ~ C(arm, Treatment(reference='{c}')) * C({dim}, Treatment(reference='{ref_level}'))"
    )
    res = smf.ols(formula, data=df).fit(cov_type="HC2")
    coefs = pd.DataFrame(
        {
            "term": res.params.index, "coef": res.params.to_numpy(), "robust_se": res.bse.to_numpy(),
            "ci_low": res.conf_int()[0].to_numpy(), "ci_high": res.conf_int()[1].to_numpy(),
            "p_value": res.pvalues.to_numpy(),
        }
    )
    arm_term = f"C(arm, Treatment(reference='{c}'))[T.{t}]"
    inter = [p for p in res.params.index if p.startswith(arm_term + ":")]
    wald = res.wald_test(", ".join(f"{p} = 0" for p in inter), use_f=False, scalar=True)
    stat = float(np.asarray(wald.statistic).squeeze())
    wald_p = float(stats.chi2.sf(stat, len(inter)))  # sf keeps precision for tiny p-values
    assert np.isclose(stat, top["q_stat"], rtol=1e-6), (stat, top["q_stat"])
    test = pd.DataFrame(
        [
            {
                "pattern": f"{top['comparison']} lift in {m} differs by {dim}",
                "formula": formula, "reference_level": ref_level, "interaction_terms": len(inter),
                "wald_chi2_hc2": stat, "df": len(inter), "p_value": wald_p,
                "cochran_q": top["q_stat"], "q_matches_wald": True, "p_holm_54_tests": top["p_holm"],
            }
        ]
    )
    return coefs, test


def chart(sub: pd.DataFrame, het: pd.DataFrame, eff: pd.DataFrame, path: Path) -> Path:
    fig, axes = plots.new_figure(1, 2, figsize=(8.0, 6.3), sharey=True)
    rows, labels, headers = [], [], []
    ypos = 0
    for dim in cfg.SUBGROUP_DIMS:
        headers.append((ypos, SHORT_DIM_LABEL[dim]))
        ypos += 1
        for level in cfg.LEVEL_ORDER[dim]:
            rows.append((ypos, dim, level))
            labels.append(level)
            ypos += 1
    total = ypos
    panels = [("visit", 100, "Visit-rate lift (percentage points)"), ("spend", 1, "Spend lift ($ per customer)")]
    for ax, (m, scale, xlabel) in zip(axes, panels):
        for i, (y0, label) in enumerate(headers):
            nxt = headers[i + 1][0] if i + 1 < len(headers) else total
            if i % 2 == 0:
                ax.axhspan(total - nxt + 0.5, total - y0 + 0.5, color="#F4F4F4", zorder=0)
        for t, color, marker, dy in [("Mens", cfg.NAVY, "o", 0.16), ("Womens", cfg.TEAL, "D", -0.16)]:
            overall = eff[(eff["comparison"] == f"{t} vs Control") & (eff["metric"] == m)]["abs_lift"].iloc[0]
            ax.axvline(overall * scale, color=color, ls="--", lw=0.9, alpha=0.8)
            for y, dim, level in rows:
                r = sub[(sub["dimension"] == dim) & (sub["level"] == level) & (sub["comparison"] == f"{t} vs Control")
                        & (sub["metric"] == m)].iloc[0]
                yy = total - y + dy
                ax.plot([r["ci_low"] * scale, r["ci_high"] * scale], [yy, yy], color=color, lw=1.1)
                ax.scatter(r["abs_lift"] * scale, yy, color=color, marker=marker, s=16, zorder=3,
                           label=f"{t} email vs. no email" if (y == rows[0][0]) else None)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel(xlabel)
    ytick_pos = [total - y for y, _, _ in rows] + [total - y for y, _ in headers]
    ytick_lab = labels + [h for _, h in headers]
    axes[0].set_yticks(ytick_pos, ytick_lab)
    for tick, lab in zip(axes[0].get_yticklabels(), ytick_lab):
        if lab in dict(headers).values():
            tick.set_fontweight("bold")
    axes[0].set_ylim(0.3, total + 0.7)
    axes[0].set_title("Visits", fontsize=9)
    axes[1].set_title("Spend per customer", fontsize=9)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    lo = sub.loc[(sub["metric"] == "spend") & (sub["comparison"] != "Mens vs Womens"), "ci_low"].min()
    hi = sub.loc[(sub["metric"] == "spend") & (sub["comparison"] != "Mens vs Womens"), "ci_high"].max()
    axes[1].set_xlim(lo - 0.2, hi + 1.4)  # room on the right for the legend
    axes[1].legend(handles, leg_labels, loc="lower right", fontsize=7)
    ph = sub[(sub["dimension"] == "purchase_history") & (sub["comparison"] == "Womens vs Control") & (sub["metric"] == "visit")]
    ph = ph.set_index("level")["abs_lift"]
    title = "The Womens email barely moved men's-only buyers; the Mens email worked for all"
    n_sig = int(het["significant_holm"].sum())
    n_sig_spend = int(het.loc[het["metric"] == "spend", "significant_holm"].sum())
    subtitle = (
        f"Womens email visit lift: {ph['Mens only'] * 100:+.1f} pts for men's-only buyers vs. "
        f"{ph['Womens only'] * 100:+.1f} for women's-only. 95% CIs; dashed lines = overall lift.\n"
        f"Only {n_sig} of {len(het)} subgroup-difference tests survive Holm correction ({n_sig_spend} on spend): "
        "most spend differences here are noise."
    )
    fig.subplots_adjust(left=0.2, right=0.98, wspace=0.08)
    return plots.finish(fig, title, path, subtitle=subtitle, top=0.85, bottom=0.1)


def main() -> dict:
    df = prepare.load_clean()
    sub = subgroup_effects(df)
    het = heterogeneity_tests(sub)
    coefs, test = interaction_model(df, het)
    eff = pd.read_csv(cfg.RESULTS / "main_effects.csv")
    sub.to_csv(cfg.RESULTS / "subgroup_effects.csv", index=False)
    het.to_csv(cfg.RESULTS / "heterogeneity_tests.csv", index=False)
    coefs.to_csv(cfg.RESULTS / "interaction_model.csv", index=False)
    test.to_csv(cfg.RESULTS / "interaction_test.csv", index=False)
    chart(sub, het, eff, cfg.IMAGES / "subgroup_forest.png")
    print(
        f"Step 5 heterogeneity: {int(het['significant_holm'].sum())} of {len(het)} tests significant after Holm; "
        f"strongest: {test['pattern'].iloc[0]} (Wald chi2 = {test['wald_chi2_hc2'].iloc[0]:.1f})"
    )
    return {"subgroups": sub, "tests": het, "interaction": coefs}


if __name__ == "__main__":
    main()
