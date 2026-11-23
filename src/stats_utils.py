"""Small, tested statistics helpers used by every analysis step.

Conventions: "t" = treatment group, "c" = comparison group (usually the
no-email control). Confidence intervals are 95% and two-sided unless stated.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config as cfg

Z = float(stats.norm.ppf(1 - cfg.ALPHA / 2))          # 1.96
Z_POWER = float(stats.norm.ppf(cfg.POWER))             # 0.84
MDE_MULTIPLIER = Z + Z_POWER                         # 2.80: MDE = 2.80 x standard error


# ---------------------------------------------------------------- two groups
def two_proportion_ztest(x_t: int, n_t: int, x_c: int, n_c: int) -> dict:
    """Difference in proportions.

    Test: pooled-variance z-test (the textbook two-proportion test under H0: p_t = p_c).
    CI:   unpooled (Wald) standard error, which does not assume H0.
    """
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z = diff / se_pool
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    return {
        "mean_t": p_t, "mean_c": p_c, "n_t": n_t, "n_c": n_c, "abs_lift": diff, "se": se,
        "ci_low": diff - Z * se, "ci_high": diff + Z * se,
        "test": "two-proportion z-test", "stat": z, "p_value": 2 * stats.norm.sf(abs(z)),
        "se_mean_t": math.sqrt(p_t * (1 - p_t) / n_t), "se_mean_c": math.sqrt(p_c * (1 - p_c) / n_c),
    }


def welch_ttest(y_t: np.ndarray, y_c: np.ndarray) -> dict:
    """Difference in means with Welch's unequal-variance t-test and t-based CI."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    n_t, n_c = len(y_t), len(y_c)
    m_t, m_c = y_t.mean(), y_c.mean()
    v_t, v_c = y_t.var(ddof=1), y_c.var(ddof=1)
    se = math.sqrt(v_t / n_t + v_c / n_c)
    dof = (v_t / n_t + v_c / n_c) ** 2 / ((v_t / n_t) ** 2 / (n_t - 1) + (v_c / n_c) ** 2 / (n_c - 1))
    t = (m_t - m_c) / se
    tcrit = float(stats.t.ppf(1 - cfg.ALPHA / 2, dof))
    res = stats.ttest_ind(y_t, y_c, equal_var=False)
    assert math.isclose(res.statistic, t, rel_tol=1e-9)
    return {
        "mean_t": m_t, "mean_c": m_c, "n_t": n_t, "n_c": n_c, "abs_lift": m_t - m_c, "se": se,
        "ci_low": m_t - m_c - tcrit * se, "ci_high": m_t - m_c + tcrit * se,
        "test": "Welch t-test", "stat": t, "p_value": float(res.pvalue), "dof": dof,
        "se_mean_t": math.sqrt(v_t / n_t), "se_mean_c": math.sqrt(v_c / n_c),
        "sd_t": math.sqrt(v_t), "sd_c": math.sqrt(v_c),
    }


def diff_in_means(y_t: np.ndarray, y_c: np.ndarray) -> dict:
    """Difference in means with the unpooled (Neyman) standard error and a normal CI.

    Used for subgroup and policy estimates: for 0/1 outcomes this is the unpooled
    two-proportion standard error (with n-1 in the variance), for spend it is Welch's.
    """
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    n_t, n_c = len(y_t), len(y_c)
    m_t, m_c = y_t.mean(), y_c.mean()
    se = math.sqrt(y_t.var(ddof=1) / n_t + y_c.var(ddof=1) / n_c)
    diff = m_t - m_c
    z = diff / se if se > 0 else float("nan")
    return {
        "mean_t": m_t, "mean_c": m_c, "n_t": n_t, "n_c": n_c, "abs_lift": diff, "se": se,
        "ci_low": diff - Z * se, "ci_high": diff + Z * se,
        "p_value": 2 * stats.norm.sf(abs(z)) if se > 0 else float("nan"),
    }


def relative_lift_ci(m_t: float, se_t: float, m_c: float, se_c: float) -> tuple[float, float, float]:
    """Relative lift m_t / m_c - 1 with a delta-method CI on the log ratio.

    For proportions this is the Katz log interval for a risk ratio.
    """
    rel = m_t / m_c - 1
    se_log = math.sqrt((se_t / m_t) ** 2 + (se_c / m_c) ** 2)
    log_r = math.log(m_t / m_c)
    return rel, math.exp(log_r - Z * se_log) - 1, math.exp(log_r + Z * se_log) - 1


# ---------------------------------------------------------------- many tests
def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    return multipletests(np.asarray(p_values, float), alpha=cfg.ALPHA, method="holm")[1]


def cochran_q(estimates, ses) -> dict:
    """Test that several independent estimates are equal (Cochran's Q, chi-square)."""
    est, se = np.asarray(estimates, float), np.asarray(ses, float)
    w = 1 / se**2
    pooled = float((w * est).sum() / w.sum())
    q = float((w * (est - pooled) ** 2).sum())
    dof = len(est) - 1
    return {"q_stat": q, "q_df": dof, "q_p_value": float(stats.chi2.sf(q, dof)), "pooled": pooled}


# ---------------------------------------------------------------- design checks
def smd(x_t: np.ndarray, x_c: np.ndarray) -> float:
    """Standardized mean difference: (mean_t - mean_c) / sqrt((var_t + var_c) / 2)."""
    x_t, x_c = np.asarray(x_t, float), np.asarray(x_c, float)
    denom = math.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    return (x_t.mean() - x_c.mean()) / denom if denom > 0 else 0.0


def mde(se: float) -> float:
    """Minimum detectable effect (two-sided alpha = 0.05, 80% power) for a given standard error."""
    return MDE_MULTIPLIER * se


def n_per_arm_needed(delta: float, var_t: float, var_c: float) -> float:
    """Customers per arm to detect a true difference `delta` with 80% power at alpha = 0.05."""
    return MDE_MULTIPLIER**2 * (var_t + var_c) / delta**2


# ---------------------------------------------------------------- bootstrap
def bootstrap_mean_zero_inflated(y: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of a mean for data that is mostly zeros.

    Identical in distribution to the ordinary row bootstrap (draw n rows with
    replacement), but much cheaper: in each resample the number of non-zero rows
    drawn is Binomial(n, k/n), and each of those is a uniform draw from the k
    non-zero values. Zeros add nothing to the sum, so they never need drawing.
    """
    y = np.asarray(y, float)
    n = len(y)
    nonzero = y[y != 0]
    k = len(nonzero)
    counts = rng.binomial(n, k / n, size=n_boot)
    picks = nonzero[rng.integers(0, k, size=int(counts.sum()))]
    sums = np.bincount(np.repeat(np.arange(n_boot), counts), weights=picks, minlength=n_boot)
    return sums / n


def bootstrap_mean_rows(y: np.ndarray, n_boot: int, rng: np.random.Generator, chunk: int = 200) -> np.ndarray:
    """Textbook row bootstrap of a mean (used only to check the shortcut above)."""
    y = np.asarray(y, float)
    n = len(y)
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        out[start:start + b] = y[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return out


def percentile_ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [100 * cfg.ALPHA / 2, 100 * (1 - cfg.ALPHA / 2)])
    return float(lo), float(hi)
