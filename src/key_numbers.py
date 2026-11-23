"""Collect every number the README cites, straight from results/*.csv.

Each row: key, the exact text used in the README ("display"), the raw value, and the
results file (+ row/column) it comes from. src/verify_readme.py then checks that every
number in README.md is one of these (or a listed design constant / allowlisted token).

Output: results/key_numbers.csv

    python -I src/key_numbers.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402

R = cfg.RESULTS
ROWS: list[dict] = []


# ------------------------------------------------------------------ formatters
def n0(x) -> str:
    return f"{x:,.0f}"


MINUS = "\u2212"  # typographic minus sign used for every negative number in the README


def _m(text: str) -> str:
    return text.replace("-", MINUS)


def pct(x, d=0, sign=False) -> str:
    return _m(f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")


def pts(x, d=2, sign=True) -> str:
    return _m(f"{x * 100:+.{d}f} pts" if sign else f"{x * 100:.{d}f}")


def signed(x, d=0) -> str:
    return _m(f"{x:+,.{d}f}")


def usd(x, d=2, sign=False) -> str:
    s = f"${abs(x):,.{d}f}"
    if sign:
        return ("+" if x >= 0 else "−") + s
    return ("−" if x < 0 else "") + s


def pval(p) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def add(key: str, value, display: str, source: str, detail: str = "") -> None:
    ROWS.append({"key": key, "display": display, "value": value, "source": source, "detail": detail})


# ------------------------------------------------------------------ builders
def data_numbers() -> None:
    dq = pd.read_csv(R / "data_quality.csv", keep_default_na=False).set_index("check_name")["value"]
    src = "results/data_quality.csv"
    add("rows", dq["rows"], n0(dq["rows"]), src, "rows")
    add("buyers_total", dq["buyers_conversion_1"], n0(dq["buyers_conversion_1"]), src, "buyers_conversion_1")
    add("buyers_share", dq["buyers_conversion_1"] / dq["rows"], pct(dq["buyers_conversion_1"] / dq["rows"], 1), src,
        "buyers_conversion_1 / rows")
    add("zero_spend_share", 1 - dq["buyers_conversion_1"] / dq["rows"], pct(1 - dq["buyers_conversion_1"] / dq["rows"], 1),
        src, "1 - buyers_conversion_1 / rows")
    add("surburban_rows", dq["zip_code_spelled_Surburban"], n0(dq["zip_code_spelled_Surburban"]), src, "zip_code_spelled_Surburban")
    add("duplicate_rows", dq["exact_duplicate_rows"], n0(dq["exact_duplicate_rows"]), src, "exact_duplicate_rows")
    add("duplicate_rows_at_floor", dq["exact_duplicate_rows_with_history_29.99"],
        n0(dq["exact_duplicate_rows_with_history_29.99"]), src, "exact_duplicate_rows_with_history_29.99")
    add("history_floor", dq["history_min_dollars"], usd(dq["history_min_dollars"]), src, "history_min_dollars")
    add("customers_at_history_floor", dq["customers_with_history_exactly_29.99"],
        n0(dq["customers_with_history_exactly_29.99"]), src, "customers_with_history_exactly_29.99")
    add("spend_max", dq["buyer_spend_max_dollars"], usd(dq["buyer_spend_max_dollars"], 0), src, "buyer_spend_max_dollars")
    add("buyers_at_spend_max", dq["buyers_at_max_spend"], n0(dq["buyers_at_max_spend"]), src, "buyers_at_max_spend")
    add("spend_second_highest", dq["highest_buyer_spend_below_max"], usd(dq["highest_buyer_spend_below_max"]), src,
        "highest_buyer_spend_below_max")
    add("orders_at_2999", dq["buyers_at_29.99"], n0(dq["buyers_at_29.99"]), src, "buyers_at_29.99")
    add("bought_both", dq["customers_buying_both_mens_and_womens"], n0(dq["customers_buying_both_mens_and_womens"]),
        src, "customers_buying_both_mens_and_womens")


def randomization_numbers() -> None:
    srm = pd.read_csv(R / "srm_check.csv").set_index("arm")
    for arm in cfg.ARMS:
        add(f"n_{arm.lower()}", srm.loc[arm, "customers"], n0(srm.loc[arm, "customers"]), "results/srm_check.csv",
            f"customers, arm={arm}")
    add("srm_p", srm["p_value"].iloc[0], f"{srm['p_value'].iloc[0]:.2f}", "results/srm_check.csv", "p_value")
    add("srm_chi2", srm["chi2_stat"].iloc[0], f"{srm['chi2_stat'].iloc[0]:.2f}", "results/srm_check.csv", "chi2_stat")
    bal = pd.read_csv(R / "balance_table.csv")
    mx = bal["max_abs_smd"].max()
    add("max_abs_smd", mx, f"{mx:.3f}", "results/balance_table.csv", "max of max_abs_smd")
    add("smd_bound", math.ceil(mx * 100) / 100, f"{math.ceil(mx * 100) / 100:.2f}", "results/balance_table.csv",
        "max_abs_smd rounded up to 2 decimals")
    add("balance_covariates", len(bal), str(len(bal)), "results/balance_table.csv", "rows")
    jt = pd.read_csv(R / "balance_joint_test.csv").iloc[0]
    add("joint_lr_p", jt["p_value"], f"{jt['p_value']:.2f}", "results/balance_joint_test.csv", "p_value")
    add("joint_lr_chi2", jt["lr_chi2"], f"{jt['lr_chi2']:.1f}", "results/balance_joint_test.csv", "lr_chi2")
    add("joint_lr_df", jt["df"], str(int(jt["df"])), "results/balance_joint_test.csv", "df")


def main_effect_numbers() -> None:
    arms = pd.read_csv(R / "arm_metrics.csv").set_index("arm")
    src = "results/arm_metrics.csv"
    for arm in cfg.ARMS:
        a = arm.lower()
        add(f"visit_rate_{a}", arms.loc[arm, "visit_rate"], pct(arms.loc[arm, "visit_rate"], 2), src, f"visit_rate, {arm}")
        add(f"conv_rate_{a}", arms.loc[arm, "conversion_rate"], pct(arms.loc[arm, "conversion_rate"], 2), src,
            f"conversion_rate, {arm}")
        add(f"spend_{a}", arms.loc[arm, "spend_per_customer"], usd(arms.loc[arm, "spend_per_customer"]), src,
            f"spend_per_customer, {arm}")
        add(f"buyers_{a}", arms.loc[arm, "buyers"], n0(arms.loc[arm, "buyers"]), src, f"buyers, {arm}")

    eff = pd.read_csv(R / "main_effects.csv")
    src = "results/main_effects.csv"
    short = {"Mens vs Control": "m", "Womens vs Control": "w", "Mens vs Womens": "mw"}
    for _, e in eff.iterrows():
        k = f"{short[e['comparison']]}_{e['metric']}"
        detail = f"comparison={e['comparison']}, metric={e['metric']}"
        if e["metric"] == "spend":
            add(f"{k}_lift", e["abs_lift"], usd(e["abs_lift"], 2, sign=True), src, detail + ", abs_lift")
            add(f"{k}_ci", (e["boot_ci_low"], e["boot_ci_high"]),
                f"{usd(e['boot_ci_low'])} to {usd(e['boot_ci_high'])}", src, detail + ", boot_ci_low/high")
            add(f"{k}_rel_ci", (e["boot_rel_ci_low"], e["boot_rel_ci_high"]),
                f"{pct(e['boot_rel_ci_low'])} to {pct(e['boot_rel_ci_high'])}", src, detail + ", boot_rel_ci_low/high")
            add(f"{k}_welch_ci", (e["ci_low"], e["ci_high"]), f"{usd(e['ci_low'])} to {usd(e['ci_high'])}", src,
                detail + ", ci_low/high (Welch)")
            add(f"{k}_per1000", e["lift_per_1000"], usd(e["lift_per_1000"], 0, sign=True), src, detail + ", lift_per_1000")
        else:
            add(f"{k}_lift", e["abs_lift"], pts(e["abs_lift"]), src, detail + ", abs_lift")
            add(f"{k}_ci", (e["ci_low"], e["ci_high"]), f"{pts(e['ci_low'], sign=False)} to {pts(e['ci_high'], sign=False)}",
                src, detail + ", ci_low/high (x100)")
            add(f"{k}_rel_ci", (e["rel_ci_low"], e["rel_ci_high"]), f"{pct(e['rel_ci_low'])} to {pct(e['rel_ci_high'])}",
                src, detail + ", rel_ci_low/high")
            fmt = 0 if e["metric"] == "visit" else 1
            add(f"{k}_per1000", e["lift_per_1000"], signed(e["lift_per_1000"], fmt), src, detail + ", lift_per_1000")
        add(f"{k}_rel", e["rel_lift"], pct(e["rel_lift"], 0, sign=True), src, detail + ", rel_lift")
        add(f"{k}_holm_p", e["p_holm"], pval(e["p_holm"]), src, detail + ", p_holm")
    add("holm_significant", int(eff["significant_holm"].sum()), str(int(eff["significant_holm"].sum())), src,
        "count of significant_holm")
    add("holm_family_size", len(eff), str(len(eff)), src, "rows")

    prec = pd.read_csv(R / "metric_precision.csv").set_index("metric")
    src = "results/metric_precision.csv"
    add("halfwidth_visit_mens", prec.loc["visit", "mens_ci_halfwidth_pct_of_lift"],
        f"±{prec.loc['visit', 'mens_ci_halfwidth_pct_of_lift'] * 100:.0f}%", src, "visit, mens_ci_halfwidth_pct_of_lift")
    add("halfwidth_spend_mens", prec.loc["spend", "mens_ci_halfwidth_pct_of_lift"],
        f"±{prec.loc['spend', 'mens_ci_halfwidth_pct_of_lift'] * 100:.0f}%", src, "spend, mens_ci_halfwidth_pct_of_lift")
    add("cv_multiplier_spend", prec.loc["spend", "sample_size_multiplier_vs_visit"],
        f"{prec.loc['spend', 'sample_size_multiplier_vs_visit']:.0f}×", src, "spend, sample_size_multiplier_vs_visit")

    spb = pd.read_csv(R / "spend_per_buyer.csv").set_index("arm")
    src = "results/spend_per_buyer.csv"
    for arm in cfg.ARMS:
        add(f"spend_per_buyer_{arm.lower()}", spb.loc[arm, "mean_spend_per_buyer"],
            usd(spb.loc[arm, "mean_spend_per_buyer"], 0), src, f"mean_spend_per_buyer, {arm}")
    for arm in cfg.TREATMENTS:
        add(f"spend_per_buyer_p_{arm.lower()}", spb.loc[arm, "p_value_vs_control"], pval(spb.loc[arm, "p_value_vs_control"]),
            src, f"p_value_vs_control, {arm}")
    dec = pd.read_csv(R / "spend_lift_decomposition.csv").set_index("arm")
    for arm in cfg.TREATMENTS:
        add(f"share_more_buyers_{arm.lower()}", dec.loc[arm, "share_from_more_buyers"],
            pct(dec.loc[arm, "share_from_more_buyers"]), "results/spend_lift_decomposition.csv",
            f"share_from_more_buyers, {arm}")


def power_numbers() -> None:
    pw = pd.read_csv(R / "power_mde.csv")
    src = "results/power_mde.csv"
    mc = pw[pw["comparison"] == "Mens vs Control"].set_index("metric")
    for m in cfg.METRICS:
        add(f"mde_design_{m}", mc.loc[m, "mde_design_rel"], pct(mc.loc[m, "mde_design_rel"]), src,
            f"Mens vs Control, {m}, mde_design_rel")
        add(f"obs_over_mde_{m}", mc.loc[m, "observed_over_design_mde"], f"{mc.loc[m, 'observed_over_design_mde']:.1f}×",
            src, f"Mens vs Control, {m}, observed_over_design_mde")
    ss = pd.read_csv(R / "sample_size_planning.csv")
    src = "results/sample_size_planning.csv"
    r = ss[(ss["metric"] == "spend") & (ss["question"] == "detect a 10% relative lift vs. no email")].iloc[0]
    add("n_for_10pct_spend", r["customers_per_group"], n0(r["customers_per_group"]), src, "spend, 10% lift, customers_per_group")
    add("mult_for_10pct_spend", r["multiple_of_this_test"], f"{r['multiple_of_this_test']:.0f}×", src,
        "spend, 10% lift, multiple_of_this_test")
    r = ss[(ss["metric"] == "visit") & (ss["question"] == "detect a 10% relative lift vs. no email")].iloc[0]
    add("n_for_10pct_visit", r["customers_per_group"], n0(r["customers_per_group"]), src, "visit, 10% lift, customers_per_group")
    r = ss[ss["question"].str.startswith("confirm")].iloc[0]
    add("n_confirm_mw_spend", r["customers_per_group"], n0(r["customers_per_group"]), src, "confirm M-vs-W gap, customers_per_group")
    add("mult_confirm_mw_spend", r["multiple_of_this_test"], f"{r['multiple_of_this_test']:.1f}×", src,
        "confirm M-vs-W gap, multiple_of_this_test")


def regression_numbers() -> None:
    ra = pd.read_csv(R / "regression_adjustment.csv")
    src = "results/regression_adjustment.csv"

    def row(outcome, comp, method_prefix):
        return ra[(ra["outcome"] == outcome) & (ra["comparison"] == comp) & ra["method"].str.startswith(method_prefix)].iloc[0]

    for outcome in cfg.METRICS:
        for comp, k in [("Mens vs Control", "m"), ("Womens vs Control", "w")]:
            r = row(outcome, comp, "3.")
            add(f"ols_se_change_{outcome}_{k}", r["se_change_vs_unadjusted"], pct(r["se_change_vs_unadjusted"], 1, sign=True),
                src, f"{outcome}, {comp}, OLS + covariates, se_change_vs_unadjusted")
            add(f"ols_extra_sample_{outcome}_{k}", r["equivalent_extra_sample"], pct(r["equivalent_extra_sample"], 1),
                src, f"{outcome}, {comp}, OLS + covariates, equivalent_extra_sample")
            add(f"ols_estimate_{outcome}_{k}", r["estimate"],
                usd(r["estimate"], 2, sign=True) if outcome == "spend" else pts(r["estimate"]),
                src, f"{outcome}, {comp}, OLS + covariates, estimate")
            add(f"ols_se_{outcome}_{k}", r["robust_se"],
                f"{r['robust_se']:.4f}" if outcome == "spend" else f"{r['robust_se'] * 100:.3f}",
                src, f"{outcome}, {comp}, OLS + covariates, robust_se ({'$' if outcome == 'spend' else 'pts'})")
            u = row(outcome, comp, "1.")
            add(f"unadj_se_{outcome}_{k}", u["robust_se"],
                f"{u['robust_se']:.4f}" if outcome == "spend" else f"{u['robust_se'] * 100:.3f}",
                src, f"{outcome}, {comp}, difference in means, robust_se ({'$' if outcome == 'spend' else 'pts'})")
            add(f"unadj_estimate_{outcome}_{k}", u["estimate"],
                usd(u["estimate"], 2, sign=True) if outcome == "spend" else pts(u["estimate"]),
                src, f"{outcome}, {comp}, difference in means, estimate")
            c = row(outcome, comp, "2.")
            add(f"cuped_se_change_{outcome}_{k}", c["se_change_vs_unadjusted"], pct(c["se_change_vs_unadjusted"], 1, sign=True),
                src, f"{outcome}, {comp}, CUPED, se_change_vs_unadjusted")
        r = row(outcome, "Mens vs Control", "3.")
        add(f"ols_r2_{outcome}", r["model_r2"], pct(r["model_r2"], 1), src, f"{outcome}, OLS + covariates, model_r2")
        add(f"partial_r2_{outcome}", r["covariate_partial_r2"], pct(r["covariate_partial_r2"], 1), src,
            f"{outcome}, OLS + covariates, covariate_partial_r2")
        c = row(outcome, "Mens vs Control", "2.")
        add(f"cuped_corr_{outcome}", c["cuped_corr"], f"{c['cuped_corr']:.3f}", src, f"{outcome}, CUPED, cuped_corr")
    # largest movement of any adjusted estimate (methods 2-4) vs. the unadjusted one
    adj = ra[~ra["method"].str.startswith("1.")]
    mv = adj[adj["outcome"] == "visit"]["estimate_change_vs_unadjusted"].abs().max()
    ms = adj[adj["outcome"] == "spend"]["estimate_change_vs_unadjusted"].abs().max()
    add("max_estimate_move_visit", mv, f"< {math.ceil(mv * 10000) / 100:.2f} pts", src,
        "visit, max |estimate_change_vs_unadjusted| (methods 2-4), rounded up")
    add("max_estimate_move_spend", ms, f"< {usd(math.ceil(ms * 100) / 100)}", src,
        "spend, max |estimate_change_vs_unadjusted| (methods 2-4), rounded up")
    ols = ra[ra["method"].str.startswith("3.")]
    mx = ols.loc[ols["outcome"] != "visit", "se_change_vs_unadjusted"].abs().max()
    add("max_se_change_conv_spend", mx, f"{math.ceil(mx * 1000) / 10:.1f}%", src,
        "conversion & spend, max |se_change_vs_unadjusted| (OLS + covariates), rounded up")


def heterogeneity_numbers() -> None:
    sub = pd.read_csv(R / "subgroup_effects.csv")
    src = "results/subgroup_effects.csv"
    ph = sub[sub["dimension"] == "purchase_history"]
    lv_key = {"Mens only": "mo", "Womens only": "wo", "Both": "both"}
    for _, r in ph.iterrows():
        comp = {"Mens vs Control": "m", "Womens vs Control": "w", "Mens vs Womens": "mw"}[r["comparison"]]
        k = f"ph_{lv_key[r['level']]}_{comp}_{r['metric']}"
        detail = f"purchase_history={r['level']}, {r['comparison']}, {r['metric']}"
        if r["metric"] == "spend":
            add(k, r["abs_lift"], usd(r["abs_lift"], 2, sign=True), src, detail + ", abs_lift")
            add(k + "_ci", (r["ci_low"], r["ci_high"]), f"{usd(r['ci_low'], 2, sign=True)} to {usd(r['ci_high'], 2, sign=True)}",
                src, detail + ", ci_low/high")
        else:
            add(k, r["abs_lift"], pts(r["abs_lift"], 1), src, detail + ", abs_lift")
            add(k + "_ci", (r["ci_low"], r["ci_high"]), f"{pts(r['ci_low'], 1, sign=True)} to {pts(r['ci_high'], 1, sign=True)}",
                src, detail + ", ci_low/high")
    for comp, k in [("Mens vs Control", "m"), ("Womens vs Control", "w")]:
        for m in cfg.METRICS:
            g = sub[(sub["comparison"] == comp) & (sub["metric"] == m)]
            add(f"subgroups_ci_excl0_{k}_{m}", int(g["ci_excludes_zero"].sum()), str(int(g["ci_excludes_zero"].sum())), src,
                f"{comp}, {m}, count of ci_excludes_zero")
    neg = sub[(sub["comparison"] != "Mens vs Womens") & (sub["metric"] == "spend") & (sub["ci_high"] < 0)]
    add("subgroups_sig_negative_spend", len(neg), str(len(neg)), src,
        "email vs control, spend, count of subgroups with ci_high < 0")
    add("subgroup_levels", sub[["dimension", "level"]].drop_duplicates().shape[0],
        str(sub[["dimension", "level"]].drop_duplicates().shape[0]), src, "distinct dimension x level")
    add("subgroup_dims", sub["dimension"].nunique(), str(sub["dimension"].nunique()), src, "distinct dimension")
    hb = sub[(sub["dimension"] == "history_bucket") & (sub["comparison"] == "Womens vs Control") & (sub["metric"] == "spend")]
    lo, hi = hb.loc[hb["abs_lift"].idxmin()], hb.loc[hb["abs_lift"].idxmax()]
    add("w_spend_hist_low", lo["abs_lift"], usd(lo["abs_lift"], 2, sign=True), src, f"history_bucket={lo['level']}, Womens vs Control, spend")
    add("w_spend_hist_high", hi["abs_lift"], usd(hi["abs_lift"], 2, sign=True), src, f"history_bucket={hi['level']}, Womens vs Control, spend")

    het = pd.read_csv(R / "heterogeneity_tests.csv")
    src = "results/heterogeneity_tests.csv"
    sig = het[het["significant_holm"]]
    # README claim: every Holm-significant heterogeneity test is about past purchases and visits
    assert (sig["dimension"] == "purchase_history").all() and (sig["metric"] == "visit").all(), sig
    add("het_tests", len(het), str(len(het)), src, "rows")
    add("het_sig_holm", int(het["significant_holm"].sum()), str(int(het["significant_holm"].sum())), src, "count significant_holm")
    add("het_sig_holm_spend", int(het.loc[het["metric"] == "spend", "significant_holm"].sum()),
        str(int(het.loc[het["metric"] == "spend", "significant_holm"].sum())), src, "count significant_holm, spend")
    add("het_sig_raw", int(het["significant_raw"].sum()), str(int(het["significant_raw"].sum())), src, "count significant_raw")
    r = het[(het["dimension"] == "history_bucket") & (het["comparison"] == "Womens vs Control") & (het["metric"] == "spend")].iloc[0]
    add("het_hist_w_spend_p", r["p_value"], f"{r['p_value']:.2f}", src, "history_bucket, Womens vs Control, spend, p_value")
    add("het_hist_w_spend_p_holm", r["p_holm"], f"{r['p_holm']:.2f}", src, "history_bucket, Womens vs Control, spend, p_holm")
    r = het[(het["dimension"] == "purchase_history") & (het["comparison"] == "Mens vs Control") & (het["metric"] == "visit")].iloc[0]
    add("het_ph_m_visit_q", r["q_stat"], f"{r['q_stat']:.1f}", src, "purchase_history, Mens vs Control, visit, q_stat")

    it = pd.read_csv(R / "interaction_test.csv").iloc[0]
    add("interaction_chi2", it["wald_chi2_hc2"], f"{it['wald_chi2_hc2']:.1f}", "results/interaction_test.csv", "wald_chi2_hc2")
    im = pd.read_csv(R / "interaction_model.csv")
    t = im[im["term"].str.contains(r"\[T.Womens\]:") & im["term"].str.contains("Womens only")].iloc[0]
    add("interaction_w_wo_coef", t["coef"], pts(t["coef"], 1), "results/interaction_model.csv",
        "Womens x Womens-only interaction, coef")
    add("interaction_w_wo_ci", (t["ci_low"], t["ci_high"]), f"{pts(t['ci_low'], 1, sign=False)} to {pts(t['ci_high'], 1, sign=False)}",
        "results/interaction_model.csv", "Womens x Womens-only interaction, ci_low/high (x100)")


def policy_numbers() -> None:
    sp = pd.read_csv(R / "policy_split.csv").set_index("half")
    add("train_size", sp.loc["train", "total"], n0(sp.loc["train", "total"]), "results/policy_split.csv", "train total")
    add("holdout_size", sp.loc["holdout", "total"], n0(sp.loc["holdout", "total"]), "results/policy_split.csv", "holdout total")
    rule = pd.read_csv(R / "policy_rule.csv").set_index("level")
    src = "results/policy_rule.csv"
    lv_key = {"Mens only": "mo", "Womens only": "wo", "Both": "both"}
    for lv, k in lv_key.items():
        for t in cfg.TREATMENTS:
            add(f"train_{k}_{t.lower()}_lift", rule.loc[lv, f"train_{t.lower()}_lift"],
                usd(rule.loc[lv, f"train_{t.lower()}_lift"], 2, sign=True), src, f"{lv}, train_{t.lower()}_lift")
    comps = pd.read_csv(R / "policy_holdout_comparisons.csv")
    src = "results/policy_holdout_comparisons.csv"
    key = {"Targeted rule": "rule", "Mens email to everyone": "mall", "Womens email to everyone": "wall", "No email": "none"}
    for _, c in comps.iterrows():
        k = f"ho_{key[c['policy']]}_vs_{key[c['versus']]}_{c['metric']}"
        detail = f"policy={c['policy']}, versus={c['versus']}, metric={c['metric']}"
        if c["metric"] == "spend":
            add(k, c["diff_per_1000"], usd(c["diff_per_1000"], 0, sign=True), src, detail + ", diff_per_1000")
            add(k + "_ci", (c["ci_low_per_1000"], c["ci_high_per_1000"]),
                f"{usd(c['ci_low_per_1000'], 0, sign=True)} to {usd(c['ci_high_per_1000'], 0, sign=True)}", src,
                detail + ", ci_low/high_per_1000")
            add(k + "_mde", c["mde80_per_1000"], usd(c["mde80_per_1000"], 0), src, detail + ", mde80_per_1000")
        else:
            fmt = 0 if c["metric"] == "visit" else 1
            add(k, c["diff_per_1000"], signed(c["diff_per_1000"], fmt), src, detail + ", diff_per_1000")
            add(k + "_ci", (c["ci_low_per_1000"], c["ci_high_per_1000"]),
                f"{signed(c['ci_low_per_1000'], fmt)} to {signed(c['ci_high_per_1000'], fmt)}", src,
                detail + ", ci_low/high_per_1000")
    vals = pd.read_csv(R / "policy_holdout_values.csv").set_index(["policy", "metric"])
    v = vals.loc[("No email", "spend"), "value_per_customer"]
    add("holdout_none_spend", v, usd(v), "results/policy_holdout_values.csv", "No email, spend, value_per_customer")
    tvh = pd.read_csv(R / "policy_train_vs_holdout.csv").set_index("policy")
    src = "results/policy_train_vs_holdout.csv"
    for pol, k in [("Targeted rule", "rule"), ("Mens email to everyone", "mall"), ("Womens email to everyone", "wall")]:
        add(f"train_value_{k}", tvh.loc[pol, "train_incremental_per_1000"], usd(tvh.loc[pol, "train_incremental_per_1000"], 0),
            src, f"{pol}, train_incremental_per_1000")
        add(f"holdout_value_{k}", tvh.loc[pol, "holdout_incremental_per_1000"],
            usd(tvh.loc[pol, "holdout_incremental_per_1000"], 0), src, f"{pol}, holdout_incremental_per_1000")
    shrink = 1 - tvh.loc["Targeted rule", "holdout_incremental_per_1000"] / tvh.loc["Targeted rule", "train_incremental_per_1000"]
    add("rule_shrinkage", shrink, pct(shrink), src, "1 - holdout / train, Targeted rule")
    diag = pd.read_csv(R / "policy_rule_search_diagnostic.csv")
    add("search_shrink_min", diag["shrinkage"].min(), pct(diag["shrinkage"].min()), "results/policy_rule_search_diagnostic.csv",
        "min shrinkage over the six one-variable rules")
    add("search_shrink_max", diag["shrinkage"].max(), pct(diag["shrinkage"].max()), "results/policy_rule_search_diagnostic.csv",
        "max shrinkage over the six one-variable rules")

    rob = pd.read_csv(R / "policy_split_robustness_summary.csv").set_index("comparison")
    src = "results/policy_split_robustness_summary.csv"
    add("robust_splits", rob["splits"].iloc[0], n0(rob["splits"].iloc[0]), src, "splits")
    add("robust_equal_mens", rob["share_rule_equals_mens_to_all"].iloc[0], pct(rob["share_rule_equals_mens_to_all"].iloc[0]),
        src, "share_rule_equals_mens_to_all")
    r = rob.loc["rule_vs_womens_to_all"]
    add("robust_vs_w_pos", r["share_of_splits_positive"], pct(r["share_of_splits_positive"]), src,
        "rule_vs_womens_to_all, share_of_splits_positive")
    add("robust_vs_w_mean", r["mean_per_1000"], usd(r["mean_per_1000"], 0, sign=True), src, "rule_vs_womens_to_all, mean_per_1000")
    r = rob.loc["rule_vs_mens_to_all"]
    add("robust_vs_m_pos", r["share_of_splits_positive"], pct(r["share_of_splits_positive"]), src,
        "rule_vs_mens_to_all, share_of_splits_positive")
    add("robust_vs_m_mean", r["mean_per_1000"], usd(r["mean_per_1000"], 0, sign=True), src, "rule_vs_mens_to_all, mean_per_1000")
    add("robust_vs_m_p95", r["p95_per_1000"], usd(r["p95_per_1000"], 0, sign=True), src, "rule_vs_mens_to_all, p95_per_1000")
    add("robust_vs_m_p05", r["p05_per_1000"], usd(r["p05_per_1000"], 0, sign=True), src, "rule_vs_mens_to_all, p05_per_1000")

    roi = pd.read_csv(R / "roi_sensitivity.csv")
    src = "results/roi_sensitivity.csv"
    for margin in sorted(roi["assumed_gross_margin"].unique()):
        add(f"margin_{margin * 100:.0f}", margin, f"{margin * 100:.0f}%", src, "assumed_gross_margin (ASSUMPTION)")
    for cost in sorted(roi["assumed_cost_per_email"].unique()):
        add(f"cost_{cost}", cost, usd(cost, 3) if cost < 0.01 else usd(cost), src, "assumed_cost_per_email (ASSUMPTION)")
    for pol, k in [("Mens email to everyone", "m"), ("Womens email to everyone", "w")]:
        for margin in sorted(roi["assumed_gross_margin"].unique()):
            r = roi[(roi["policy"] == pol) & (roi["assumed_gross_margin"] == margin)].iloc[0]
            mk = f"{margin * 100:.0f}"
            add(f"breakeven_{k}_{mk}", r["breakeven_cost_per_email"], usd(r["breakeven_cost_per_email"]), src,
                f"{pol}, margin={margin}, breakeven_cost_per_email")
            add(f"breakeven_{k}_{mk}_cilow", r["breakeven_cost_per_email_at_ci_low"],
                usd(r["breakeven_cost_per_email_at_ci_low"]), src, f"{pol}, margin={margin}, breakeven_cost_per_email_at_ci_low")
        for cost in sorted(roi["assumed_cost_per_email"].unique()):
            r = roi[(roi["policy"] == pol) & (roi["assumed_cost_per_email"] == cost) & (roi["assumed_gross_margin"] == 0.3)].iloc[0]
            add(f"profit_{k}_30_{cost}_cilow", r["incremental_gross_profit_per_1000_at_ci_low"],
                usd(r["incremental_gross_profit_per_1000_at_ci_low"], 0, sign=True), src,
                f"{pol}, margin=0.3, cost={cost}, incremental_gross_profit_per_1000_at_ci_low")


def design_constants() -> None:
    """Design choices stated in the README (they come from the code, not the data)."""
    src = "src/config.py"
    add("n_boot", cfg.N_BOOT, n0(cfg.N_BOOT), src, "N_BOOT")
    add("n_boot_policy", cfg.N_BOOT_POLICY, n0(cfg.N_BOOT_POLICY), src, "N_BOOT_POLICY")
    add("seed", cfg.SEED, str(cfg.SEED), src, "SEED")
    add("alpha", cfg.ALPHA, f"{cfg.ALPHA:.2f}", src, "ALPHA")
    add("power", cfg.POWER, f"{cfg.POWER * 100:.0f}%", src, "POWER")
    add("ci_level", 1 - cfg.ALPHA, f"{(1 - cfg.ALPHA) * 100:.0f}%", src, "1 - ALPHA")
    add("mde_multiplier", 2.80, "2.80", "src/stats_utils.py", "MDE_MULTIPLIER = 1.96 + 0.84")
    add("planning_lift_10", 0.10, "10%", "src/s03_power.py", "PLANNING_LIFTS (a 10% relative lift)")
    add("per_1000", 1000, "1,000", "src/s02_main_effects.py, src/s06_targeting_policy.py",
        "unit: effects are also reported per 1,000 customers (lift_per_1000, diff_per_1000)")
    add("challenge_best_customers", 10_000, "10,000", cfg.POST_URL,
        "the author's challenge question: which 10,000 customers would you email")


def main() -> pd.DataFrame:
    for f in [data_numbers, randomization_numbers, main_effect_numbers, power_numbers, regression_numbers,
              heterogeneity_numbers, policy_numbers, design_constants]:
        f()
    out = pd.DataFrame(ROWS)
    assert out["key"].is_unique, out[out["key"].duplicated()]
    out["value"] = out["value"].astype(str)
    out.to_csv(R / "key_numbers.csv", index=False)
    print(f"Key numbers: {len(out)} entries -> results/key_numbers.csv")
    return out


if __name__ == "__main__":
    main()
