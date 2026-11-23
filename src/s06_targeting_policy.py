"""Step 6 - Who should get which email next time? A simple targeting rule, tested on fresh customers.

1. Split customers 50/50 at random (fixed seed) into a training half and a holdout half.
2. Training half only: within each past-purchase group (men's only / women's only / both -
   chosen for a business reason: the emails feature men's vs. women's merchandise), estimate the
   spend lift of each email vs. no email and pick the email with the larger lift; pick
   "no email" if neither lift is positive.
3. Holdout half only: estimate what that rule - and three blanket policies (no email,
   Womens email to everyone, Mens email to everyone) - would have produced, per 1,000
   customers. Because the email was assigned at random within every group, the value of
   any rule is a weighted average of the matching arm's group means (a post-stratified
   difference in means). 95% CIs from 5,000 bootstrap resamples of the holdout
   (resampling within each arm).

A diagnostic table also scores all six one-variable rules on both halves; it is not used
to choose the rule (choosing on the holdout would make it a second training set).
A robustness check repeats the whole split / learn / score procedure on 500 random splits.
An ROI table converts each email's full-sample spend lift to gross profit under labelled
ASSUMPTIONS (gross margin, cost per email); these numbers are not in the data.

Outputs: results/policy_split.csv, results/policy_rule.csv, results/policy_holdout_values.csv,
         results/policy_holdout_comparisons.csv, results/policy_train_vs_holdout.csv,
         results/policy_rule_search_diagnostic.csv, results/roi_sensitivity.csv,
         results/policy_split_robustness.csv, results/policy_split_robustness_summary.csv,
         images/policy_holdout.png

    python -I src/s06_targeting_policy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import config as cfg  # noqa: E402
import plots  # noqa: E402
import prepare  # noqa: E402
import stats_utils as su  # noqa: E402

RULE_DIM = "purchase_history"
ARM_INDEX = {a: i for i, a in enumerate(cfg.ARMS)}          # Control 0, Mens 1, Womens 2
POLICY_NAMES = ["No email", "Womens email to everyone", "Mens email to everyone", "Targeted rule"]
# ASSUMPTIONS for the ROI table (not in the data): gross margin on merchandise and the
# all-in cost of sending one email.
MARGINS = [0.30, 0.40, 0.50]
COSTS_PER_EMAIL = [0.001, 0.01, 0.10]
N_SPLITS = 500  # repeated random splits for the robustness check


def split_halves(df: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(cfg.SEED)
    order = rng.permutation(len(df))
    half = np.empty(len(df), dtype=object)
    half[order[: len(df) // 2]] = "train"
    half[order[len(df) // 2:]] = "holdout"
    return pd.Series(half, index=df.index, name="half")


def learn_rule(train: pd.DataFrame, dim: str = RULE_DIM) -> pd.DataFrame:
    rows = []
    for level in cfg.LEVEL_ORDER[dim]:
        cell = train[train[dim] == level]
        ctrl = cell.loc[cell["arm"] == "Control", "spend"]
        lifts = {t: su.diff_in_means(cell.loc[cell["arm"] == t, "spend"], ctrl) for t in cfg.TREATMENTS}
        best = max(cfg.TREATMENTS, key=lambda t: lifts[t]["abs_lift"])
        action = best if lifts[best]["abs_lift"] > 0 else "Control"
        row = {"dimension": dim, "level": level, "train_customers": len(cell),
               "train_share": len(cell) / len(train), "train_mean_spend_control": ctrl.mean()}
        for t in cfg.TREATMENTS:
            r = lifts[t]
            row.update({f"train_{t.lower()}_lift": r["abs_lift"], f"train_{t.lower()}_ci_low": r["ci_low"],
                        f"train_{t.lower()}_ci_high": r["ci_high"]})
        row["action"] = action
        row["action_label"] = cfg.ARM_LABEL[action] if action != "Control" else "No email"
        row["reason"] = (
            f"larger training spend lift ({lifts[best]['abs_lift']:+.2f} $/customer vs. "
            f"{lifts['Womens' if best == 'Mens' else 'Mens']['abs_lift']:+.2f})"
            if action != "Control" else "neither email raised training spend"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def policy_actions(rule: pd.DataFrame) -> dict[str, dict[str, str]]:
    levels = list(rule["level"])
    return {
        "No email": {lv: "Control" for lv in levels},
        "Womens email to everyone": {lv: "Womens" for lv in levels},
        "Mens email to everyone": {lv: "Mens" for lv in levels},
        "Targeted rule": dict(zip(rule["level"], rule["action"])),
    }


def _arrays(part: pd.DataFrame, dim: str):
    levels = cfg.LEVEL_ORDER[dim]
    cell = part[dim].map({lv: i for i, lv in enumerate(levels)}).to_numpy(int)
    arm = part["arm"].map(ARM_INDEX).to_numpy(int)
    y = part[cfg.METRICS].to_numpy(float)
    return levels, cell, arm, y


def _policy_values(keys: np.ndarray, y: np.ndarray, idx: np.ndarray, n_cells: int,
                   act_idx: dict[str, np.ndarray]) -> dict:
    """Policy values for a batch of (re)samples.

    keys: cell * 3 + arm for every holdout row; y: (n, 3) outcomes; idx: (b, n) row indices,
    one row of indices per resample. Returns {(policy, metric): array of b values}.
    """
    b = idx.shape[0]
    k = 3 * n_cells
    flat = (keys[idx] + (np.arange(b) * k)[:, None]).ravel()
    counts = np.bincount(flat, minlength=b * k).reshape(b, n_cells, 3)
    share = counts.sum(axis=2) / counts.sum(axis=(1, 2))[:, None]
    out = {}
    for j, m in enumerate(cfg.METRICS):
        sums = np.bincount(flat, weights=y[idx, j].ravel(), minlength=b * k).reshape(b, n_cells, 3)
        means = sums / counts
        for name, act in act_idx.items():
            out[(name, m)] = (share * means[:, np.arange(n_cells), act]).sum(axis=1)
    return out


def evaluate(part: pd.DataFrame, actions: dict[str, dict[str, str]], dim: str, n_boot: int = 0,
             rng: np.random.Generator | None = None) -> tuple[dict, dict]:
    """Post-stratified value of each policy on `part`, plus bootstrap draws if n_boot > 0."""
    levels, cell, arm, y = _arrays(part, dim)
    act_idx = {name: np.array([ARM_INDEX[a[lv]] for lv in levels]) for name, a in actions.items()}
    keys = cell * 3 + arm
    point = {key: float(v[0]) for key, v in
             _policy_values(keys, y, np.arange(len(part))[None, :], len(levels), act_idx).items()}
    draws: dict = {}
    if n_boot:
        by_arm = [np.flatnonzero(arm == a) for a in range(3)]
        chunks = []
        for start in range(0, n_boot, 100):
            b = min(100, n_boot - start)
            idx = np.concatenate([rng.choice(ix, size=(b, len(ix)), replace=True) for ix in by_arm], axis=1)
            chunks.append(_policy_values(keys, y, idx, len(levels), act_idx))
        draws = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    return point, draws


def holdout_tables(point: dict, draws: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    vals = []
    for name in POLICY_NAMES:
        for m in cfg.METRICS:
            lo, hi = su.percentile_ci(draws[(name, m)])
            vals.append({"policy": name, "metric": m, "value_per_customer": point[(name, m)],
                         "value_per_1000": point[(name, m)] * 1000, "ci_low_per_1000": lo * 1000,
                         "ci_high_per_1000": hi * 1000})
    comps = []
    for name in POLICY_NAMES[1:]:
        for base in POLICY_NAMES[:3]:
            if base == name:
                continue
            for m in cfg.METRICS:
                d = draws[(name, m)] - draws[(base, m)]
                est = point[(name, m)] - point[(base, m)]
                se = d.std(ddof=1)
                lo, hi = su.percentile_ci(d)
                identical = se == 0
                comps.append(
                    {
                        "policy": name, "versus": base, "metric": m, "diff_per_1000": est * 1000,
                        "ci_low_per_1000": lo * 1000, "ci_high_per_1000": hi * 1000, "boot_se_per_1000": se * 1000,
                        "mde80_per_1000": su.mde(se) * 1000,
                        "p_value_normal": np.nan if identical else float(2 * stats.norm.sf(abs(est / se))),
                        "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    }
                )
    return pd.DataFrame(vals), pd.DataFrame(comps)


def rule_search_diagnostic(train: pd.DataFrame, hold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in cfg.SUBGROUP_DIMS:
        rule = learn_rule(train, dim)
        acts = policy_actions(rule)
        acts = {"rule": acts["Targeted rule"], "none": acts["No email"]}
        p_tr, _ = evaluate(train, acts, dim)
        p_ho, _ = evaluate(hold, acts, dim)
        rows.append(
            {
                "dimension": dim, "rule": "; ".join(f"{lv}: {a}" for lv, a in zip(rule["level"], rule["action"])),
                "train_incremental_spend_per_1000": (p_tr[("rule", "spend")] - p_tr[("none", "spend")]) * 1000,
                "holdout_incremental_spend_per_1000": (p_ho[("rule", "spend")] - p_ho[("none", "spend")]) * 1000,
                "used_for_decision": dim == RULE_DIM,
            }
        )
    out = pd.DataFrame(rows)
    out["shrinkage"] = 1 - out["holdout_incremental_spend_per_1000"] / out["train_incremental_spend_per_1000"]
    return out


def split_robustness(df: pd.DataFrame, n_splits: int = N_SPLITS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the whole learn-on-one-half / score-on-the-other procedure on many random splits.

    Guards against reading too much into one lucky (or unlucky) split. Point estimates only.
    """
    levels = cfg.LEVEL_ORDER[RULE_DIM]
    cell = df[RULE_DIM].map({lv: i for i, lv in enumerate(levels)}).to_numpy(int)
    arm = df["arm"].map(ARM_INDEX).to_numpy(int)
    spend = df["spend"].to_numpy(float)
    key = cell * 3 + arm
    rng = np.random.default_rng(cfg.SEED + 2)
    rows = []
    for i in range(n_splits):
        train = np.zeros(len(df), bool)
        train[rng.permutation(len(df))[: len(df) // 2]] = True
        stats_ = {}
        for name, mask in [("train", train), ("hold", ~train)]:
            cnt = np.bincount(key[mask], minlength=9).reshape(3, 3)
            tot = np.bincount(key[mask], weights=spend[mask], minlength=9).reshape(3, 3)
            stats_[name] = (tot / cnt, cnt.sum(axis=1) / cnt.sum())
        m_tr, _ = stats_["train"]
        lifts = m_tr[:, 1:] - m_tr[:, [0]]                      # (cell, Mens/Womens)
        act = np.where(lifts.max(axis=1) > 0, 1 + lifts.argmax(axis=1), 0)
        m_ho, w_ho = stats_["hold"]
        value = {
            "rule": (w_ho * m_ho[np.arange(3), act]).sum(),
            "mens": (w_ho * m_ho[:, 1]).sum(),
            "womens": (w_ho * m_ho[:, 2]).sum(),
            "none": (w_ho * m_ho[:, 0]).sum(),
        }
        rows.append(
            {
                "split": i,
                "rule": "; ".join(f"{lv}: {cfg.ARMS[a]}" for lv, a in zip(levels, act)),
                "rule_equals_mens_to_all": bool((act == 1).all()),
                "rule_vs_mens_to_all_per_1000": (value["rule"] - value["mens"]) * 1000,
                "rule_vs_womens_to_all_per_1000": (value["rule"] - value["womens"]) * 1000,
                "rule_vs_no_email_per_1000": (value["rule"] - value["none"]) * 1000,
            }
        )
    per = pd.DataFrame(rows)
    summ = []
    for col in ["rule_vs_mens_to_all_per_1000", "rule_vs_womens_to_all_per_1000", "rule_vs_no_email_per_1000"]:
        x = per[col]
        summ.append(
            {
                "comparison": col.replace("_per_1000", ""), "splits": n_splits,
                "share_rule_equals_mens_to_all": per["rule_equals_mens_to_all"].mean(),
                "mean_per_1000": x.mean(), "median_per_1000": x.median(),
                "p05_per_1000": x.quantile(0.05), "p95_per_1000": x.quantile(0.95),
                "share_of_splits_positive": (x > 0).mean(),
                "share_of_splits_negative": (x < 0).mean(),
            }
        )
    return per, pd.DataFrame(summ)


def roi_table(eff: pd.DataFrame) -> pd.DataFrame:
    """Gross profit per 1,000 customers emailed, under ASSUMED margins and email costs.

    Uses the full-sample spend lift of each email vs. no email (step 2, bootstrap CI): the
    most precise estimate of 'email everyone with X'. Spend is revenue, so it is turned into
    gross profit with an assumed gross margin; each email is charged an assumed send cost.
    """
    rows = []
    for t in cfg.TREATMENTS:
        e = eff[(eff["comparison"] == f"{t} vs Control") & (eff["metric"] == "spend")].iloc[0]
        lift, lo = e["lift_per_1000"], e["lift_per_1000_ci_low"]
        for margin in MARGINS:
            for cost in COSTS_PER_EMAIL:
                rows.append(
                    {
                        "policy": f"{t} email to everyone", "assumed_gross_margin": margin,
                        "assumed_cost_per_email": cost, "emails_per_1000_customers": 1000,
                        "incremental_spend_per_1000": lift, "incremental_spend_per_1000_ci_low": lo,
                        "incremental_gross_profit_per_1000": margin * lift - 1000 * cost,
                        "incremental_gross_profit_per_1000_at_ci_low": margin * lo - 1000 * cost,
                        "breakeven_cost_per_email": margin * lift / 1000,
                        "breakeven_cost_per_email_at_ci_low": margin * lo / 1000,
                    }
                )
    return pd.DataFrame(rows)


def _money(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


def chart(comps: pd.DataFrame, rule: pd.DataFrame, path: Path) -> Path:
    fig, ax = plots.new_figure()
    names = POLICY_NAMES[1:]
    sub = comps[(comps["versus"] == "No email")].set_index(["policy", "metric"])
    y = np.arange(len(names))[::-1]
    colors = {"Womens email to everyone": cfg.TEAL, "Mens email to everyone": cfg.NAVY, "Targeted rule": cfg.ORANGE}
    spend = sub.xs("spend", level="metric")
    for yi, name in zip(y, names):
        s_ = spend.loc[name]
        ax.barh(yi, s_["diff_per_1000"], color=colors[name], height=0.55)
        ax.errorbar(s_["diff_per_1000"], yi, xerr=[[s_["diff_per_1000"] - s_["ci_low_per_1000"]],
                                                   [s_["ci_high_per_1000"] - s_["diff_per_1000"]]],
                    fmt="none", ecolor="#222222", capsize=4, lw=1)
        v = sub.loc[(name, "visit")]["diff_per_1000"]
        b = sub.loc[(name, "conversion")]["diff_per_1000"]
        ax.text(s_["ci_high_per_1000"] + 30, yi, f"{_money(s_['diff_per_1000'])}  |  {v:+.0f} visits, {b:+.1f} buyers",
                va="center", fontsize=8)
    who = {lv: a for lv, a in zip(rule["level"], rule["action"])}
    mens_to = [lv for lv, a in who.items() if a == "Mens"]
    womens_to = [lv for lv, a in who.items() if a == "Womens"]
    none_to = [lv for lv, a in who.items() if a == "Control"]
    nice = {"Mens only": "men's-only", "Womens only": "women's-only", "Both": "both"}
    rule_txt = "Targeted rule"
    if mens_to:
        rule_txt += "\nMens email: " + " & ".join(nice[x] for x in mens_to)
    if womens_to:
        rule_txt += "\nWomens email: " + " & ".join(nice[x] for x in womens_to)
    if none_to:
        rule_txt += "\nNo email: " + " & ".join(nice[x] for x in none_to)
    labels = {"Womens email to everyone": "Womens email to everyone", "Mens email to everyone": "Mens email to everyone",
              "Targeted rule": rule_txt}
    ax.set_yticks(y, [labels[n] for n in names])
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlim(min(0, spend["ci_low_per_1000"].min()) * 1.15, spend["ci_high_per_1000"].max() * 1.78)
    ax.set_xlabel("Extra spend per 1,000 customers vs. sending no email, holdout half ($, 95% CI)")
    t = comps[(comps["policy"] == "Targeted rule")].set_index(["versus", "metric"])
    tw, tm = t.loc[("Womens email to everyone", "spend")], t.loc[("Mens email to everyone", "spend")]
    beat_w = "beat" if tw["ci_low_per_1000"] > 0 else "did not clearly beat"
    beat_m = "beat" if tm["ci_low_per_1000"] > 0 else "did not beat"
    title = f"On fresh customers, targeting {beat_w} Womens-to-all but {beat_m} Mens-to-all"
    subtitle = (
        f"Targeted rule vs. Womens-to-all: {_money(tw['diff_per_1000'])} per 1,000 customers (95% CI "
        f"{_money(tw['ci_low_per_1000'])} to {_money(tw['ci_high_per_1000'])}); vs. Mens-to-all: "
        f"{_money(tm['diff_per_1000'])} ({_money(tm['ci_low_per_1000'])} to {_money(tm['ci_high_per_1000'])}).\n"
        "Rule learned on a random half of customers, scored on the other half. Bars: spend; text: visits and buyers."
    )
    fig.subplots_adjust(left=0.29, right=0.98)
    return plots.finish(fig, title, path, subtitle=subtitle, top=0.8, bottom=0.14)


def main() -> dict:
    df = prepare.load_clean()
    df["half"] = split_halves(df)
    train, hold = df[df["half"] == "train"], df[df["half"] == "holdout"]
    split = pd.crosstab(df["half"], df["arm"]).reindex(["train", "holdout"])
    split["total"] = split.sum(axis=1)
    split.reset_index().to_csv(cfg.RESULTS / "policy_split.csv", index=False)

    rule = learn_rule(train)
    actions = policy_actions(rule)
    rng = np.random.default_rng(cfg.SEED + 1)
    point, draws = evaluate(hold, actions, RULE_DIM, cfg.N_BOOT_POLICY, rng)
    vals, comps = holdout_tables(point, draws)
    p_tr, _ = evaluate(train, actions, RULE_DIM)
    tvh = pd.DataFrame(
        [
            {
                "policy": name, "metric": "spend",
                "train_incremental_per_1000": (p_tr[(name, "spend")] - p_tr[("No email", "spend")]) * 1000,
                "holdout_incremental_per_1000": (point[(name, "spend")] - point[("No email", "spend")]) * 1000,
            }
            for name in POLICY_NAMES[1:]
        ]
    )
    diag = rule_search_diagnostic(train, hold)
    roi = roi_table(pd.read_csv(cfg.RESULTS / "main_effects.csv"))
    rob_per, rob_sum = split_robustness(df.drop(columns="half"))
    rob_per.to_csv(cfg.RESULTS / "policy_split_robustness.csv", index=False)
    rob_sum.to_csv(cfg.RESULTS / "policy_split_robustness_summary.csv", index=False)

    rule.to_csv(cfg.RESULTS / "policy_rule.csv", index=False)
    vals.to_csv(cfg.RESULTS / "policy_holdout_values.csv", index=False)
    comps.to_csv(cfg.RESULTS / "policy_holdout_comparisons.csv", index=False)
    tvh.to_csv(cfg.RESULTS / "policy_train_vs_holdout.csv", index=False)
    diag.to_csv(cfg.RESULTS / "policy_rule_search_diagnostic.csv", index=False)
    roi.to_csv(cfg.RESULTS / "roi_sensitivity.csv", index=False)
    chart(comps, rule, cfg.IMAGES / "policy_holdout.png")
    t = comps[(comps["policy"] == "Targeted rule") & (comps["metric"] == "spend")].set_index("versus")
    print(
        "Step 6 targeting: rule = " + ", ".join(f"{lv}: {a}" for lv, a in zip(rule["level"], rule["action"]))
        + "; holdout spend per 1,000 vs " + "; ".join(
            f"{v}: {r.diff_per_1000:+.0f} [{r.ci_low_per_1000:+.0f}, {r.ci_high_per_1000:+.0f}]" for v, r in t.iterrows())
    )
    return {"rule": rule, "values": vals, "comparisons": comps}


if __name__ == "__main__":
    main()
