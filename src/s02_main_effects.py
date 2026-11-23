"""Step 2 - Main effects of each email vs. no email, and Mens vs. Womens.

Metrics (two weeks after the send): visit rate, conversion rate, spend per customer.
* visit / conversion: two-proportion z-test; unpooled 95% CI; relative-lift CI on the log ratio.
* spend: Welch t-test and t-based CI, plus a bootstrap (10,000 resamples) because spend
  is zero for 99% of customers and very skewed for the rest.
* Holm correction over the family of 9 tests (3 comparisons x 3 metrics).

Outputs: results/arm_metrics.csv, results/main_effects.csv, results/metric_precision.csv,
         results/spend_per_buyer.csv, results/spend_lift_decomposition.csv,
         results/bootstrap_check.csv, images/lift_by_metric.png, images/spend_distribution.png

    python -I src/s02_main_effects.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402
import plots  # noqa: E402
import prepare  # noqa: E402
import stats_utils as su  # noqa: E402


def arm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("arm", observed=True)
    out = pd.DataFrame(
        {
            "customers": g.size(),
            "visits": g["visit"].sum(),
            "buyers": g["conversion"].sum(),
            "revenue": g["spend"].sum(),
            "visit_rate": g["visit"].mean(),
            "conversion_rate": g["conversion"].mean(),
            "spend_per_customer": g["spend"].mean(),
            "spend_sd": g["spend"].std(),
        }
    ).reindex(cfg.ARMS)
    out["spend_per_buyer"] = out["revenue"] / out["buyers"]
    return out.reset_index()


def compare(df: pd.DataFrame, t_arm: str, c_arm: str, metric: str) -> dict:
    y_t = df.loc[df["arm"] == t_arm, metric].to_numpy(float)
    y_c = df.loc[df["arm"] == c_arm, metric].to_numpy(float)
    if metric == "spend":
        r = su.welch_ttest(y_t, y_c)
    else:
        r = su.two_proportion_ztest(int(y_t.sum()), len(y_t), int(y_c.sum()), len(y_c))
    rel, rel_lo, rel_hi = su.relative_lift_ci(r["mean_t"], r["se_mean_t"], r["mean_c"], r["se_mean_c"])
    return {
        "comparison": f"{t_arm} vs {c_arm}",
        "treatment": t_arm,
        "baseline": c_arm,
        "metric": metric,
        "test": r["test"],
        "n_treatment": r["n_t"],
        "n_baseline": r["n_c"],
        "mean_treatment": r["mean_t"],
        "mean_baseline": r["mean_c"],
        "abs_lift": r["abs_lift"],
        "se": r["se"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"],
        "rel_lift": rel,
        "rel_ci_low": rel_lo,
        "rel_ci_high": rel_hi,
        "stat": r["stat"],
        "p_value": r["p_value"],
    }


def bootstrap_spend(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Bootstrap draws of mean spend for each arm (exact zero-inflated shortcut)."""
    return {
        arm: su.bootstrap_mean_zero_inflated(df.loc[df["arm"] == arm, "spend"].to_numpy(float), cfg.N_BOOT, rng)
        for arm in cfg.ARMS
    }


def bootstrap_check(df: pd.DataFrame, boot: dict[str, np.ndarray], rng: np.random.Generator) -> pd.DataFrame:
    """Show the shortcut matches the textbook row bootstrap (and the analytic SE)."""
    rows = []
    for arm in cfg.ARMS:
        y = df.loc[df["arm"] == arm, "spend"].to_numpy(float)
        rowwise = su.bootstrap_mean_rows(y, 2_000, rng)
        rows.append(
            {
                "arm": arm,
                "analytic_se_of_mean": y.std(ddof=1) / np.sqrt(len(y)),
                "shortcut_bootstrap_se_10000": boot[arm].std(ddof=1),
                "row_bootstrap_se_2000": rowwise.std(ddof=1),
            }
        )
    out = pd.DataFrame(rows)
    out["shortcut_vs_row_ratio"] = out["shortcut_bootstrap_se_10000"] / out["row_bootstrap_se_2000"]
    assert (out["shortcut_vs_row_ratio"].sub(1).abs() < 0.05).all(), out
    return out


def main_effects(df: pd.DataFrame, boot: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = [compare(df, t, c, m) for (t, c) in cfg.COMPARISONS for m in cfg.METRICS]
    out = pd.DataFrame(rows)
    out["p_holm"] = su.holm(out["p_value"])
    out["significant_raw"] = out["p_value"] < cfg.ALPHA
    out["significant_holm"] = out["p_holm"] < cfg.ALPHA
    # Bootstrap CIs for spend.
    for col in ["boot_se", "boot_ci_low", "boot_ci_high", "boot_rel_ci_low", "boot_rel_ci_high"]:
        out[col] = np.nan
    for i, r in out[out["metric"] == "spend"].iterrows():
        diff = boot[r["treatment"]] - boot[r["baseline"]]
        ratio = boot[r["treatment"]] / boot[r["baseline"]] - 1
        out.loc[i, "boot_se"] = diff.std(ddof=1)
        out.loc[i, ["boot_ci_low", "boot_ci_high"]] = su.percentile_ci(diff)
        out.loc[i, ["boot_rel_ci_low", "boot_rel_ci_high"]] = su.percentile_ci(ratio)
    # Per 1,000 customers emailed: visits, buyers or dollars.
    out["lift_per_1000"] = out["abs_lift"] * 1000
    lo = np.where(out["metric"] == "spend", out["boot_ci_low"], out["ci_low"])
    hi = np.where(out["metric"] == "spend", out["boot_ci_high"], out["ci_high"])
    out["lift_per_1000_ci_low"] = lo * 1000
    out["lift_per_1000_ci_high"] = hi * 1000
    return out


def metric_precision(df: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    """How precisely each metric is measured, and why spend is so much noisier."""
    ctrl = df[df["arm"] == "Control"]
    rows = []
    for m in cfg.METRICS:
        mean, sd = ctrl[m].mean(), ctrl[m].std()
        row = {"metric": m, "control_mean": mean, "control_sd": sd, "coefficient_of_variation": sd / mean}
        for t in cfg.TREATMENTS:
            e = eff[(eff["comparison"] == f"{t} vs Control") & (eff["metric"] == m)].iloc[0]
            lo, hi = (e["boot_ci_low"], e["boot_ci_high"]) if m == "spend" else (e["ci_low"], e["ci_high"])
            row[f"{t.lower()}_ci_halfwidth_pct_of_lift"] = (hi - lo) / 2 / abs(e["abs_lift"])
        rows.append(row)
    out = pd.DataFrame(rows)
    cv_visit = out.loc[out["metric"] == "visit", "coefficient_of_variation"].iloc[0]
    # Sample size needed for the same relative precision scales with CV^2.
    out["sample_size_multiplier_vs_visit"] = (out["coefficient_of_variation"] / cv_visit) ** 2
    return out


def spend_per_buyer(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive only: buyers are a post-treatment subset, so this is not a randomized comparison."""
    buyers = df[df["conversion"] == 1]
    rows = []
    for arm in cfg.ARMS:
        y = buyers.loc[buyers["arm"] == arm, "spend"].to_numpy(float)
        se = y.std(ddof=1) / np.sqrt(len(y))
        rows.append(
            {
                "arm": arm, "buyers": len(y), "mean_spend_per_buyer": y.mean(), "median_spend_per_buyer": np.median(y),
                "sd": y.std(ddof=1), "ci_low": y.mean() - su.Z * se, "ci_high": y.mean() + su.Z * se,
            }
        )
    out = pd.DataFrame(rows)
    ctrl = buyers.loc[buyers["arm"] == "Control", "spend"].to_numpy(float)
    for t in cfg.TREATMENTS:
        r = su.welch_ttest(buyers.loc[buyers["arm"] == t, "spend"].to_numpy(float), ctrl)
        out.loc[out["arm"] == t, "diff_vs_control"] = r["abs_lift"]
        out.loc[out["arm"] == t, "p_value_vs_control"] = r["p_value"]
    return out


def spend_decomposition(arms: pd.DataFrame) -> pd.DataFrame:
    """Spend per customer = conversion rate x spend per buyer. Split each lift into
    'more buyers' (at control's order value) and 'different order value'."""
    a = arms.set_index("arm")
    rows = []
    for t in cfg.TREATMENTS:
        d_conv = a.loc[t, "conversion_rate"] - a.loc["Control", "conversion_rate"]
        d_aov = a.loc[t, "spend_per_buyer"] - a.loc["Control", "spend_per_buyer"]
        total = a.loc[t, "spend_per_customer"] - a.loc["Control", "spend_per_customer"]
        more_buyers = d_conv * a.loc["Control", "spend_per_buyer"]
        order_value = a.loc[t, "conversion_rate"] * d_aov
        assert np.isclose(more_buyers + order_value, total)
        rows.append(
            {
                "arm": t, "spend_lift": total, "from_more_buyers": more_buyers, "from_order_value": order_value,
                "share_from_more_buyers": more_buyers / total,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts
def _fmt_abs(metric: str, x: float) -> str:
    if metric == "spend":
        return f"{'+' if x >= 0 else '-'}${abs(x):.2f}"
    return f"{x * 100:+.1f} pts" if metric == "visit" else f"{x * 100:+.2f} pts"


def _fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def chart_lift(eff: pd.DataFrame, path: Path) -> Path:
    fig, axes = plots.new_figure(1, 3)
    titles = {"visit": "Visit rate", "conversion": "Conversion rate", "spend": "Spend per customer"}
    for ax, m in zip(axes, cfg.METRICS):
        sub = eff[(eff["metric"] == m) & (eff["baseline"] == "Control")].set_index("treatment").loc[cfg.TREATMENTS]
        lo_col, hi_col = ("boot_rel_ci_low", "boot_rel_ci_high") if m == "spend" else ("rel_ci_low", "rel_ci_high")
        x = np.arange(2)
        vals = sub["rel_lift"].to_numpy() * 100
        err = np.vstack([vals - sub[lo_col].to_numpy() * 100, sub[hi_col].to_numpy() * 100 - vals])
        ax.bar(x, vals, color=[cfg.ARM_COLOR[t] for t in cfg.TREATMENTS], width=0.6)
        ax.errorbar(x, vals, yerr=err, fmt="none", ecolor="#222222", capsize=4, lw=1)
        for xi, (t, r) in zip(x, sub.iterrows()):
            ax.text(
                xi, r[hi_col] * 100 + 4, f"{r['rel_lift'] * 100:+.0f}%\n({_fmt_abs(m, r['abs_lift'])})",
                ha="center", va="bottom", fontsize=7.5,
            )
        ax.set_xticks(x, ["Mens email", "Womens email"])
        base = sub["mean_baseline"].iloc[0]
        base_txt = f"${base:.2f}" if m == "spend" else f"{base * 100:.2f}%"
        ax.set_title(f"{titles[m]}\n(no email: {base_txt})", fontsize=9)
        ax.axhline(0, color="#444444", lw=0.8)
        ax.set_ylim(0, 230)
    axes[0].set_ylabel("Lift vs. no email (%), 95% CI")
    mw = eff[eff["comparison"] == "Mens vs Womens"].set_index("metric")
    m_spend = eff[(eff["comparison"] == "Mens vs Control") & (eff["metric"] == "spend")].iloc[0]
    w_spend = eff[(eff["comparison"] == "Womens vs Control") & (eff["metric"] == "spend")].iloc[0]
    title = (
        f"Both emails worked: spend per customer rose {m_spend['rel_lift'] * 100:.0f}% (Mens) "
        f"and {w_spend['rel_lift'] * 100:.0f}% (Womens)"
    )
    vs_ctrl = eff[eff["baseline"] == "Control"]
    n_sig = int(vs_ctrl["significant_holm"].sum())
    mw_sig = "Mens beat Womens on all three metrics" if mw["significant_holm"].all() else "Mens vs. Womens"
    subtitle = (
        f"Two-week lift vs. the no-email group with 95% CIs (spend: bootstrap); {n_sig} of 6 lifts significant after "
        "Holm correction.\n"
        f"{mw_sig} (Holm {_fmt_p(mw.loc['visit', 'p_holm'])} visits, {_fmt_p(mw.loc['conversion', 'p_holm'])} "
        f"purchases, {_fmt_p(mw.loc['spend', 'p_holm'])} spend), but the spend gap's CI is wide."
    )
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.98)
    return plots.finish(fig, title, path, subtitle=subtitle, top=0.75, bottom=0.11)


def chart_spend(df: pd.DataFrame, spb: pd.DataFrame, dq: pd.Series, path: Path) -> Path:
    buyers = df[df["conversion"] == 1]
    fig, (ax1, ax2) = plots.new_figure(1, 2, gridspec_kw={"width_ratios": [1.7, 1]})
    bins = np.arange(0, 520, 20)
    ax1.hist(buyers["spend"], bins=bins, color=cfg.NAVY, edgecolor="white", linewidth=0.6)
    med = float(buyers["spend"].median())
    ax1.axvline(med, color=cfg.ORANGE, lw=1.2, ls="--")
    ax1.text(med + 6, ax1.get_ylim()[1] * 0.92, f"median ${med:.2f}", color=cfg.ORANGE, fontsize=7.5)
    n29 = int(dq["buyers_at_29.99"])
    n499 = int(dq["buyers_at_max_spend"])
    ax1.annotate(f"{n29} orders of exactly $29.99", xy=(35, ax1.get_ylim()[1] * 0.75), xytext=(120, ax1.get_ylim()[1] * 0.75),
                 fontsize=7.5, arrowprops={"arrowstyle": "->", "color": "#555555"}, va="center")
    ax1.annotate(f"{n499} orders of exactly ${dq['buyer_spend_max_dollars']:.0f}\n(nothing above: looks capped)",
                 xy=(495, 9), xytext=(330, ax1.get_ylim()[1] * 0.45), fontsize=7.5, ha="center",
                 arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax1.set_xlabel("Spend per buyer in the two weeks after the email ($)")
    ax1.set_ylabel("Buyers")
    ax1.set_title(f"Spend of the {len(buyers)} buyers (all groups)", fontsize=9)

    y = np.arange(len(spb))[::-1]
    for yi, (_, r) in zip(y, spb.iterrows()):
        c = cfg.ARM_COLOR[r["arm"]]
        ax2.errorbar(r["mean_spend_per_buyer"], yi, xerr=[[r["mean_spend_per_buyer"] - r["ci_low"]], [r["ci_high"] - r["mean_spend_per_buyer"]]],
                     fmt="o", color=c, capsize=3)
        ax2.text(r["ci_high"] + 2, yi, f"${r['mean_spend_per_buyer']:.0f} ({int(r['buyers'])} buyers)", va="center", fontsize=7.5)
    ax2.set_yticks(y, [cfg.ARM_LABEL[a] for a in spb["arm"]])
    ax2.set_xlim(80, 175)
    ax2.set_xlabel("Mean spend per buyer ($), 95% CI")
    ax2.set_title("Mean spend per buyer, by group", fontsize=9)
    total = len(df)
    title = f"Only {len(buyers)} of {total:,} customers ({len(buyers) / total:.1%}) bought anything in the two weeks"
    p_m = spb.loc[spb["arm"] == "Mens", "p_value_vs_control"].iloc[0]
    p_w = spb.loc[spb["arm"] == "Womens", "p_value_vs_control"].iloc[0]
    subtitle = (
        "So spend per customer rests on a few hundred orders and is the noisiest metric.\n"
        f"Buyers spent about the same in every group (vs. control: Mens {_fmt_p(p_m)}, Womens {_fmt_p(p_w)}): "
        "the emails added buyers, not bigger orders."
    )
    fig.subplots_adjust(wspace=0.55, left=0.07, right=0.97)
    return plots.finish(fig, title, path, subtitle=subtitle, top=0.77, bottom=0.14)


def main() -> dict:
    df = prepare.load_clean()
    rng = np.random.default_rng(cfg.SEED)
    arms = arm_metrics(df)
    boot = bootstrap_spend(df, rng)
    eff = main_effects(df, boot)
    check = bootstrap_check(df, boot, rng)
    prec = metric_precision(df, eff)
    spb = spend_per_buyer(df)
    decomp = spend_decomposition(arms)

    arms.to_csv(cfg.RESULTS / "arm_metrics.csv", index=False)
    eff.to_csv(cfg.RESULTS / "main_effects.csv", index=False)
    check.to_csv(cfg.RESULTS / "bootstrap_check.csv", index=False)
    prec.to_csv(cfg.RESULTS / "metric_precision.csv", index=False)
    spb.to_csv(cfg.RESULTS / "spend_per_buyer.csv", index=False)
    decomp.to_csv(cfg.RESULTS / "spend_lift_decomposition.csv", index=False)

    dq = pd.read_csv(cfg.RESULTS / "data_quality.csv", keep_default_na=False).set_index("check_name")["value"]
    chart_lift(eff, cfg.IMAGES / "lift_by_metric.png")
    chart_spend(df, spb, dq, cfg.IMAGES / "spend_distribution.png")
    print(
        "Step 2 main effects: "
        + "; ".join(f"{r.comparison} {r.metric} {r.abs_lift:+.4f} (Holm p={r.p_holm:.3g})" for r in eff.itertuples())
    )
    return {"arms": arms, "effects": eff, "precision": prec}


if __name__ == "__main__":
    main()
