"""Write notebooks/ab_test_analysis.ipynb (narrated walkthrough) and execute it.

The notebook re-derives the headline tests by hand, then reads the CSVs that the
pipeline wrote to results/. Markdown cells hold no result numbers: every number in
the narrative is printed by a code cell, so it always matches results/.

    python -I src/build_notebook.py          # build + execute
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import nbformat as nbf  # noqa: E402

import config as cfg  # noqa: E402

CELLS: list[tuple[str, str]] = [
    ("md", """# Email campaign A/B/n test: did it work, for whom, and who should get it next time?

A narrated walkthrough of the analysis in this repo. Data: Kevin Hillstrom's MineThatData E-Mail Analytics
Challenge (2008): 64,000 recent customers were randomly split into three equal groups. One group got an email
featuring men's merchandise, one an email featuring women's merchandise, one no email. Visits, purchases and
spend were tracked for the two weeks after the send.

How to read it: `src/run_all.py` rebuilds every table in `results/` and every chart in `images/` from the raw
CSV. This notebook re-derives the key tests by hand, then reads those files. The markdown cells contain no
result numbers; every number is printed by a code cell."""),
    ("code", """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from IPython.display import Image, display
from scipy import stats

import config as cfg
import prepare
import stats_utils as su

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
R = cfg.RESULTS


def show(path):
    display(Image(filename=str(ROOT / "images" / path), width=900))


df = prepare.load_clean()
print(f"{len(df):,} customers x {df.shape[1]} columns")
df.head()"""),
    ("md", """## 0. The data and its quirks

No missing values, but a few things to know before analysing: a typo in the ZIP-code labels, many exact
duplicate rows (there is no customer ID), and a spend variable that is zero for almost everyone and appears to
be capped at the top. The checks below run in SQL (DuckDB, `sql/01_data_quality.sql`)."""),
    ("code", """dq = pd.read_csv(R / "data_quality.csv", keep_default_na=False)
dq"""),
    ("md", """## 1. Was the randomization clean?

Two checks before trusting any comparison:
* **Sample ratio mismatch (SRM)**: are the group sizes consistent with a 1/3 split? A chi-square goodness-of-fit
  test; a tiny p-value would mean the assignment or the data pipeline is broken.
* **Covariate balance**: pre-email traits should look the same in every group. Standardized mean differences
  (SMD) under 0.1 are the usual threshold; a multinomial logit of group on all traits gives one joint test."""),
    ("code", """srm = pd.read_csv(R / "srm_check.csv")
bal = pd.read_csv(R / "balance_table.csv")
joint = pd.read_csv(R / "balance_joint_test.csv").iloc[0]
display(srm[["arm", "customers", "expected_customers", "share"]])
print(f"SRM chi-square = {srm['chi2_stat'].iloc[0]:.3f}, p = {srm['p_value'].iloc[0]:.3f}")
print(f"Largest |SMD| across all pairs of groups: {bal['max_abs_smd'].max():.4f}")
print(f"Joint test (multinomial logit LR): chi2 = {joint['lr_chi2']:.2f}, df = {joint['df']}, p = {joint['p_value']:.3f}")
bal[["covariate", "mean_control", "mean_mens", "mean_womens", "smd_mens_vs_control", "smd_womens_vs_control", "p_value"]]"""),
    ("code", """show("balance_smd.png")"""),
    ("md", """## 2. Main effects: each email vs. no email, and Mens vs. Womens

Visits and conversions are 0/1 outcomes, so each comparison is a **two-proportion z-test** (pooled variance for
the test, unpooled for the confidence interval). Below, the Mens-vs-control visit test is computed by hand and
checked against the pipeline's table."""),
    ("code", """eff = pd.read_csv(R / "main_effects.csv")
t, c = df[df["arm"] == "Mens"]["visit"], df[df["arm"] == "Control"]["visit"]
p_t, p_c = t.mean(), c.mean()
p_pool = (t.sum() + c.sum()) / (len(t) + len(c))
z = (p_t - p_c) / np.sqrt(p_pool * (1 - p_pool) * (1 / len(t) + 1 / len(c)))
se = np.sqrt(p_t * (1 - p_t) / len(t) + p_c * (1 - p_c) / len(c))
lo, hi = p_t - p_c - 1.96 * se, p_t - p_c + 1.96 * se
print(f"Visit rate: Mens {p_t:.2%} vs. no email {p_c:.2%}")
print(f"Lift {100 * (p_t - p_c):+.2f} pts, 95% CI [{100 * lo:.2f}, {100 * hi:.2f}], z = {z:.1f}, p = {2 * stats.norm.sf(abs(z)):.1e}")
row = eff[(eff["comparison"] == "Mens vs Control") & (eff["metric"] == "visit")].iloc[0]
assert np.isclose(row["abs_lift"], p_t - p_c) and np.isclose(row["stat"], z)
print("Matches results/main_effects.csv")"""),
    ("md", """Spend is different: about 99% of customers spent \\$0, and the rest spent anywhere up to a few hundred
dollars. The difference in means uses **Welch's t-test** (unequal variances), and the confidence interval is
also **bootstrapped** (10,000 resamples). The bootstrap here uses an exact shortcut: in each resample only the
non-zero rows matter, so it draws how many buyers land in the resample (Binomial) and which ones (uniform). The
check table shows it matches the textbook row bootstrap and the analytic standard error."""),
    ("code", """display(pd.read_csv(R / "bootstrap_check.csv"))
cols = ["comparison", "metric", "mean_treatment", "mean_baseline", "abs_lift", "ci_low", "ci_high",
        "boot_ci_low", "boot_ci_high", "rel_lift", "p_value", "p_holm", "significant_holm"]
eff[cols]"""),
    ("md", """**Holm correction.** Nine tests (3 comparisons x 3 metrics) are treated as one family; Holm's step-down
method keeps the chance of any false positive at 5%. The sentence below is generated from the table."""),
    ("code", """def fmt(r):
    if r["metric"] == "spend":
        return f"{r['abs_lift']:+.2f} $/customer (bootstrap 95% CI {r['boot_ci_low']:.2f} to {r['boot_ci_high']:.2f})"
    return f"{100 * r['abs_lift']:+.2f} pts (95% CI {100 * r['ci_low']:.2f} to {100 * r['ci_high']:.2f})"


for _, r in eff.iterrows():
    print(f"{r['comparison']:18s} {r['metric']:10s} {fmt(r):55s} relative {r['rel_lift']:+.0%}  Holm p = {r['p_holm']:.2g}")
print(f"Significant after Holm: {int(eff['significant_holm'].sum())} of {len(eff)}")"""),
    ("code", """show("lift_by_metric.png")"""),
    ("md", """**Why spend is the noisy metric.** Only a few hundred customers bought anything, so the spend average rests
on a few hundred orders. Spend per buyer was about the same in every group (a descriptive comparison: buyers
are selected after treatment, so it is not randomized), so the revenue lift comes from *more buyers*."""),
    ("code", """arms = pd.read_csv(R / "arm_metrics.csv")
display(arms)
display(pd.read_csv(R / "spend_per_buyer.csv"))
display(pd.read_csv(R / "spend_lift_decomposition.csv"))
prec = pd.read_csv(R / "metric_precision.csv")
display(prec)
p = prec.set_index("metric")
print(f"95% CI half-width as a share of the Mens lift: visits {p.loc['visit', 'mens_ci_halfwidth_pct_of_lift']:.0%}, "
      f"spend {p.loc['spend', 'mens_ci_halfwidth_pct_of_lift']:.0%}")"""),
    ("code", """show("spend_distribution.png")"""),
    ("md", """## 3. Power: what could this test detect?

Minimum detectable effect (MDE) at 80% power and a two-sided 5% test: **MDE = (1.96 + 0.84) x standard error**.
The design MDE plugs the control group's variance into both groups at this test's size. A non-significant
result only means "no effect" when the MDE is small relative to the effects that matter."""),
    ("code", """pw = pd.read_csv(R / "power_mde.csv")
display(pw[["comparison", "metric", "baseline_mean", "mde_design_rel", "mde_realized_rel", "observed_rel_lift",
            "observed_over_design_mde", "verdict"]])
pd.read_csv(R / "sample_size_planning.csv")"""),
    ("code", """show("power_mde.png")"""),
    ("md", """## 4. Regression adjustment and CUPED

Randomization makes the simple difference in means unbiased. Adding pre-email covariates to an OLS regression
(with HC2 heteroskedasticity-robust standard errors) should leave the estimate almost unchanged and can shrink
its standard error by soaking up outcome variance: the same idea as CUPED, which subtracts theta x (pre-period
metric - its mean). The gain is bounded by how much variance the covariates explain."""),
    ("code", """import statsmodels.formula.api as smf

m0 = smf.ols("visit ~ C(arm)", data=df).fit(cov_type="HC2")
m1 = smf.ols("visit ~ C(arm) + " + cfg.COVARIATE_FORMULA, data=df).fit(cov_type="HC2")
for name, m in [("unadjusted", m0), ("adjusted", m1)]:
    print(f"{name:10s} Mens effect {100 * m.params['C(arm)[T.Mens]']:+.3f} pts, robust SE {100 * m.bse['C(arm)[T.Mens]']:.4f}, R2 = {m.rsquared:.4f}")
ra = pd.read_csv(R / "regression_adjustment.csv")
ra[["outcome", "comparison", "method", "estimate", "robust_se", "model_r2", "cuped_corr", "se_change_vs_unadjusted",
    "equivalent_extra_sample"]]"""),
    ("md", """## 5. Heterogeneous effects: for whom did it work?

Six subgroup dimensions come from the customer attributes the challenge describes (past purchases, recency,
past-year spend, channel, new vs. existing, ZIP type), with cut-points set by group size, not by outcomes. With 19 subgroups x 3 comparisons x 3 metrics, some subgroup estimates will look large by
chance, so the question "does the effect differ by subgroup?" is tested once per dimension with **Cochran's Q**
(a chi-square test that the subgroup lifts are equal), and those 54 tests are Holm-corrected together."""),
    ("code", """het = pd.read_csv(R / "heterogeneity_tests.csv")
print(f"Significant after Holm: {int(het['significant_holm'].sum())} of {len(het)} "
      f"(raw p < 0.05: {int(het['significant_raw'].sum())})")
het.head(10)"""),
    ("code", """sub = pd.read_csv(R / "subgroup_effects.csv")
ph = sub[(sub["dimension"] == "purchase_history") & (sub["metric"] == "visit")]
ph.pivot(index="level", columns="comparison", values="abs_lift").mul(100).round(2)"""),
    ("code", """show("subgroup_forest.png")"""),
    ("md", """The strongest pattern, re-estimated as an interaction regression (OLS with HC2 robust SEs). Its Wald test on
the interaction terms reproduces Cochran's Q exactly, a useful cross-check that the two methods agree."""),
    ("code", """test = pd.read_csv(R / "interaction_test.csv").iloc[0]
print(test["pattern"])
print(f"Wald chi2 = {test['wald_chi2_hc2']:.2f} (df = {test['df']}), Cochran's Q = {test['cochran_q']:.2f}, p = {test['p_value']:.1e}")
pd.read_csv(R / "interaction_model.csv")"""),
    ("md", """## 6. Who should get which email next time?

A rule learned and scored on the same customers will always look good. So customers are split 50/50 at random:
the **training half** picks the email for each past-purchase group (a segmentation chosen for a business
reason: the emails feature men's vs. women's merchandise), and the **holdout half** estimates what that rule
would have produced,
next to three blanket policies. Email was randomized within every group, so a policy's value is a weighted
average of the matching group-arm means; CIs come from 5,000 bootstrap resamples of the holdout."""),
    ("code", """display(pd.read_csv(R / "policy_split.csv"))
rule = pd.read_csv(R / "policy_rule.csv")
rule[["level", "train_customers", "train_mens_lift", "train_mens_ci_low", "train_mens_ci_high",
      "train_womens_lift", "train_womens_ci_low", "train_womens_ci_high", "action_label", "reason"]]"""),
    ("code", """comps = pd.read_csv(R / "policy_holdout_comparisons.csv")
comps[comps["policy"] == "Targeted rule"]"""),
    ("code", """show("policy_holdout.png")"""),
    ("md", """**How much did the training-half estimate shrink on fresh customers?** And is the answer an accident of one
random split? The robustness check repeats the whole split / learn / score procedure 500 times."""),
    ("code", """display(pd.read_csv(R / "policy_train_vs_holdout.csv"))
display(pd.read_csv(R / "policy_rule_search_diagnostic.csv"))
pd.read_csv(R / "policy_split_robustness_summary.csv")"""),
    ("md", """**ROI under stated assumptions.** Spend is revenue, not profit, and the data has no cost per email. The table
converts each email's full-sample spend lift into gross profit per 1,000 customers emailed for *assumed* gross
margins (30-50%) and *assumed* costs per email (\\$0.001-\\$0.10). These are assumptions for illustration, not
data."""),
    ("code", """roi = pd.read_csv(R / "roi_sensitivity.csv")
roi.pivot_table(index=["policy", "assumed_gross_margin"], columns="assumed_cost_per_email",
                values="incremental_gross_profit_per_1000").round(0)"""),
    ("md", """## 7. Summary (generated from results/)"""),
    ("code", """def money(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


e = eff.set_index(["comparison", "metric"])
c = comps.set_index(["policy", "versus", "metric"])
rob = pd.read_csv(R / "policy_split_robustness_summary.csv").set_index("comparison")
m_s, w_s, mw_s = e.loc[("Mens vs Control", "spend")], e.loc[("Womens vs Control", "spend")], e.loc[("Mens vs Womens", "spend")]
print(f"* Mens email: spend +${m_s['abs_lift']:.2f} per customer ({m_s['rel_lift']:+.0%}), "
      f"95% CI ${m_s['boot_ci_low']:.2f} to ${m_s['boot_ci_high']:.2f}.")
print(f"* Womens email: spend +${w_s['abs_lift']:.2f} per customer ({w_s['rel_lift']:+.0%}), "
      f"95% CI ${w_s['boot_ci_low']:.2f} to ${w_s['boot_ci_high']:.2f}.")
print(f"* Mens minus Womens: +${mw_s['abs_lift']:.2f} per customer, Holm p = {mw_s['p_holm']:.3f}.")
w_mo = sub[(sub["dimension"] == "purchase_history") & (sub["level"] == "Mens only") & (sub["comparison"] == "Womens vs Control") & (sub["metric"] == "visit")].iloc[0]
print(f"* Womens email among men's-only buyers: visits {100 * w_mo['abs_lift']:+.1f} pts (95% CI {100 * w_mo['ci_low']:.1f} to {100 * w_mo['ci_high']:.1f}).")
for base in ["No email", "Womens email to everyone", "Mens email to everyone"]:
    r = c.loc[("Targeted rule", base, "spend")]
    print(f"* Holdout, targeted rule vs. {base}: {money(r['diff_per_1000'])} per 1,000 customers "
          f"(95% CI {money(r['ci_low_per_1000'])} to {money(r['ci_high_per_1000'])}).")
r = rob.loc["rule_vs_mens_to_all"]
print(f"* Over {int(r['splits'])} random splits the rule beat Mens-to-all in {r['share_of_splits_positive']:.0%} of splits "
      f"(mean {money(r['mean_per_1000'])} per 1,000).")"""),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    cfg.NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, cfg.NOTEBOOK)
    return nb


def execute() -> None:
    jupyter = Path(sys.executable).with_name("jupyter")
    cmd = [str(jupyter), "nbconvert", "--execute", "--to", "notebook", "--inplace",
           "--ExecutePreprocessor.timeout=600", str(cfg.NOTEBOOK)]
    subprocess.run(cmd, check=True, cwd=cfg.NOTEBOOK.parent)
    size = cfg.NOTEBOOK.stat().st_size
    assert size < 5_000_000, f"notebook is {size:,} bytes"
    print(f"Notebook executed: {cfg.NOTEBOOK.relative_to(cfg.ROOT)} ({size / 1e6:.2f} MB)")


def main() -> None:
    build()
    execute()


if __name__ == "__main__":
    main()
