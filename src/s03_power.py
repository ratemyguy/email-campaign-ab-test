"""Step 3 - Power and minimum detectable effect (MDE).

MDE at 80% power, two-sided alpha = 0.05:  MDE = (1.96 + 0.84) x standard error = 2.80 x SE.

* Design MDE: the usual pre-test calculation, plugging the control group's variance
  into both arms at this test's sample size (~21,300 customers per group).
* Realized MDE: 2.80 x the observed standard error of each comparison.
* Sample-size table: customers per group needed to detect a given relative lift
  (design formula), plus the size of a follow-up test that could confirm the
  Mens-vs-Womens spend gap if the true gap equals the observed one.

Outputs: results/power_mde.csv, results/sample_size_planning.csv, images/power_mde.png

    python -I src/s03_power.py
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

PLANNING_LIFTS = [0.05, 0.10, 0.20, 0.50]
COLOR = {"visit": cfg.NAVY, "conversion": cfg.ORANGE, "spend": cfg.TEAL}


def _a(x: float) -> str:
    """'a' or 'an' before a percentage written as digits (an 8%, an 11%, a 48%)."""
    digits = f"{x * 100:.0f}"
    return "an" if digits.startswith("8") or digits in {"11", "18"} else "a"


def power_table(df: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    ctrl = df[df["arm"] == "Control"]
    rows = []
    for _, e in eff.iterrows():
        m = e["metric"]
        var_c = ctrl[m].var(ddof=1)
        se_design = np.sqrt(var_c / e["n_treatment"] + var_c / e["n_baseline"])
        mde_design = su.mde(se_design)
        mde_real = su.mde(e["se"])
        rows.append(
            {
                "comparison": e["comparison"],
                "metric": m,
                "n_treatment": e["n_treatment"],
                "n_baseline": e["n_baseline"],
                "baseline_mean": e["mean_baseline"],
                "mde_design_abs": mde_design,
                "mde_design_rel": mde_design / e["mean_baseline"],
                "mde_realized_abs": mde_real,
                "mde_realized_rel": mde_real / e["mean_baseline"],
                "observed_abs_lift": e["abs_lift"],
                "observed_rel_lift": e["rel_lift"],
                "observed_over_design_mde": abs(e["abs_lift"]) / mde_design,
                "significant_holm": e["significant_holm"],
            }
        )
    out = pd.DataFrame(rows)
    out["verdict"] = np.where(
        out["significant_holm"],
        "detected (significant after Holm)",
        np.where(out["observed_over_design_mde"] < 1, "not detected; effect smaller than the MDE (underpowered)",
                 "not detected"),
    )
    return out


def sample_size_table(df: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    ctrl = df[df["arm"] == "Control"]
    rows = []
    for m in cfg.METRICS:
        mu, var_c = ctrl[m].mean(), ctrl[m].var(ddof=1)
        for lift in PLANNING_LIFTS:
            rows.append(
                {
                    "question": f"detect a {lift:.0%} relative lift vs. no email",
                    "metric": m,
                    "true_difference": lift * mu,
                    "customers_per_group": su.n_per_arm_needed(lift * mu, var_c, var_c),
                }
            )
    # Confirming the observed Mens-vs-Womens spend gap with 80% power.
    e = eff[(eff["comparison"] == "Mens vs Womens") & (eff["metric"] == "spend")].iloc[0]
    var_m = df.loc[df["arm"] == "Mens", "spend"].var(ddof=1)
    var_w = df.loc[df["arm"] == "Womens", "spend"].var(ddof=1)
    rows.append(
        {
            "question": "confirm the observed Mens-vs-Womens spend gap (if it is the true gap)",
            "metric": "spend",
            "true_difference": e["abs_lift"],
            "customers_per_group": su.n_per_arm_needed(e["abs_lift"], var_m, var_w),
        }
    )
    out = pd.DataFrame(rows)
    out["customers_per_group"] = np.ceil(out["customers_per_group"]).astype(int)
    out["multiple_of_this_test"] = out["customers_per_group"] / df["arm"].value_counts().min()
    return out


def chart(df: pd.DataFrame, pw: pd.DataFrame, path: Path) -> Path:
    ctrl = df[df["arm"] == "Control"]
    n_now = int(df["arm"].value_counts().min())
    n = np.logspace(np.log10(2_000), np.log10(400_000), 200)
    fig, ax = plots.new_figure()
    labels = {"visit": "Visit rate", "conversion": "Conversion rate", "spend": "Spend per customer"}
    for m in cfg.METRICS:
        cv = ctrl[m].std(ddof=1) / ctrl[m].mean()
        rel = su.MDE_MULTIPLIER * cv * np.sqrt(2 / n) * 100
        ax.plot(n, rel, color=COLOR[m], lw=2, label=labels[m])
        now = pw[(pw["comparison"] == "Mens vs Control") & (pw["metric"] == m)].iloc[0]["mde_design_rel"] * 100
        ax.scatter([n_now], [now], color=COLOR[m], zorder=3, s=28)
        ax.text(n_now * 1.08, now, f"{now:.0f}%", color=COLOR[m], va="bottom", fontsize=8, fontweight="bold")
    ax.axvline(n_now, color="#666666", ls=":", lw=1)
    ax.text(n_now * 0.96, 3.2, f"this test:\n{n_now:,} per group", ha="right", va="bottom", fontsize=7.5, color="#444444")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 400_000],
                  ["2k", "5k", "10k", "20k", "50k", "100k", "200k", "400k"])
    ax.set_yticks([3, 5, 10, 20, 50, 100, 200], ["3%", "5%", "10%", "20%", "50%", "100%", "200%"])
    ax.minorticks_off()
    ax.set_ylim(2.5, 260)
    ax.set_xlabel("Customers per group (log scale)")
    ax.set_ylabel("Smallest detectable lift, % of control (log)")
    ax.grid(True, which="major", color="#E5E5E5", lw=0.6)
    ax.legend(loc="upper right")
    v = pw[(pw["comparison"] == "Mens vs Control") & (pw["metric"] == "visit")].iloc[0]["mde_design_rel"]
    s = pw[(pw["comparison"] == "Mens vs Control") & (pw["metric"] == "spend")].iloc[0]["mde_design_rel"]
    title = f"This test could detect {_a(v)} {v:.0%} lift in visits, but only {_a(s)} {s:.0%} lift in spend"
    subtitle = (
        "Minimum detectable effect at 80% power, two-sided alpha = 0.05, equal groups, control-group variance. "
        "Spend is mostly zeros,\nso it needs far more customers than visits for the same precision."
    )
    fig.subplots_adjust(left=0.1, right=0.97)
    return plots.finish(fig, title, path, subtitle=subtitle, top=0.8, bottom=0.14)


def main() -> dict:
    df = prepare.load_clean()
    eff = pd.read_csv(cfg.RESULTS / "main_effects.csv")
    pw = power_table(df, eff)
    ss = sample_size_table(df, eff)
    pw.to_csv(cfg.RESULTS / "power_mde.csv", index=False)
    ss.to_csv(cfg.RESULTS / "sample_size_planning.csv", index=False)
    chart(df, pw, cfg.IMAGES / "power_mde.png")
    d = pw[pw["comparison"] == "Mens vs Control"].set_index("metric")["mde_design_rel"]
    print("Step 3 power: design MDE (rel.) " + ", ".join(f"{m} {d[m]:.1%}" for m in cfg.METRICS))
    return {"power": pw, "sample_size": ss}


if __name__ == "__main__":
    main()
