"""gate_config.py

Candidate counts and marginal-uniformity gate thresholds with their null calibration, including the calibration
script (run as ``python gate_config.py [n_reps]``).

The thresholds apply to the KS distance D_max of the model-generated margins of every candidate. Constants shared by fold_pipeline, model_evaluation, synthetic_study and the
notebooks live here.
"""
import functools
import math
import sys

import numpy as np
import pandas as pd


# Model candidate counts and shared marginal-uniformity screening thresholds (null-calibrated per check type).
#
# Threshold of the reference check: 2 KS tests (two margins), exact sampler, n = 1,000 draws.
# For one KS test with n iid draws from a continuous distribution, P(D > x) <= 2 exp(-2 n x^2) for every n and x
# (Dvoretzky-Kiefer-Wolfowitz inequality with Massart's tight constant; Massart, 1990, Annals of Probability, 18(3),
# 1269-1283, Corollary 1). At x = 0.07 and n = 1,000 the bound is 1.1e-4 per test, so 0.07 is above the 99.98th
# percentile of the null of each margin; the simulated 99.9th percentile of the two-test check is about 0.064.
# Calibration rule, fixed before the production run: keep 0.07 if it is
# at least the simulated 99.9th percentile of the null of the reference check; otherwise use that percentile
# rounded up to 0.005. check_ks_calibration() applies the rule and is run by notebook 05 and by this file as a script.
# KS_D_MAX_LOOSE (0.10) is used for the sensitivity MCS in the appendix.
KS_D_MAX = 0.07
KS_D_MAX_LOOSE = 0.10
KS_D_MAX_STRICT = 0.05
KS_SENSITIVITY_LEVELS = (KS_D_MAX_STRICT, KS_D_MAX, KS_D_MAX_LOOSE)
KS_N_SAMPLES = 1000
KS_NULL_REPS = 4000               # Monte Carlo replicates of the null distribution of D_max
KS_CALIB_REPS = 20000             # replicates for the 99.9th percentile used by the calibration rule
KS_NULL_QUANTILE = 0.99           # quantile of that null distribution used to align the thresholds
KS_REF_TESTS = 2                  # number of KS tests of the reference check
KS_GRID_N = 200                   # cells per dimension of the CODINE / C-CODINE grid sampler
KS_PERCENTILES = (0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95)
# Self-consistency check (does the fitted density integrate to one?): one rule for every network. It is made at the same seven
# w-quantiles as the KS check, with 10 independent Monte Carlo integrals of 10,000 uniform points, and a network passes at a w
# when |mean integral - 1| <= SELF_CONSISTENCY_TOL (an effect-size rule, like the KS gate; the standard error is a diagnostic only).
SELF_CONSISTENCY_PERCENTILES = KS_PERCENTILES
SELF_CONSISTENCY_TOL = 0.01
SELF_CONSISTENCY_N_SEEDS = 10
SELF_CONSISTENCY_N_MC = 10_000
ACAR_N_EVAL = 30
Q_GRID = (0.80, 0.85, 0.90, 0.95)    # upper-tail thresholds q of chi(q) and of the joint-exceedance probability (one definition for the whole study)
# Stamp hours (UTC) classified as night and day. Notebook 00 derives them from local solar time (lake longitude / 15 h shift,
# night = [18, 06)) and asserts that the result equals these tuples; every other module reads them from here.
NIGHT_HOURS_UTC = (18, 21, 0, 3)
DAY_HOURS_UTC = (6, 9, 12, 15)
N_SEEDS = 3
SEEDS = (0, 1, 2)

# ── candidate counts ─────────────────────────────────────────────────────────
N_FAMILIES = 7                    # Gaussian, t, Clayton, Gumbel, Gumbel (180°), Frank, Joe
N_CCODINE_VARIANTS = 3            # w_only, spline_basis, FiLM
N_CCOPFLOW_VARIANTS = 6           # 3 injection modes x {affine, spline} coupling
N_NONPARAMETRIC = 2               # Gijbels et al. (2011) and the empirical checkerboard Bernstein copula (Lu & Ghosh, 2024)
N_FOLDS = 4

assert len(SEEDS) == N_SEEDS


def n_candidates():
    """Number of candidates in the pool: 76 = (7+1) + (49+1) + (7+3+6) + 2.

    Unconditional: 7 families + CODINE. Binary split: 7x7 family pairs + split CODINE.
    Continuous: 7 Acar curves + 3 C-CODINE + 6 C-CopFlow variants. Nonparametric: Gijbels et al. (2011) and the ECBC of Lu and Ghosh (2024).

    Returns:
    int: The number of candidates.
    """
    return ((N_FAMILIES + 1)
            + (N_FAMILIES ** 2 + 1)
            + (N_FAMILIES + N_CCODINE_VARIANTS + N_CCOPFLOW_VARIANTS)
            + N_NONPARAMETRIC)


N_CANDIDATES = n_candidates()
N_CANDIDATES_SYNTHETIC = N_CANDIDATES - N_NONPARAMETRIC    # the synthetic summary has the 74 parametric, Acar and network candidates; the two
                                                           # nonparametric estimators are scored in their own frames
assert N_CANDIDATES == 76 and N_CANDIDATES_SYNTHETIC == 74, (N_CANDIDATES, N_CANDIDATES_SYNTHETIC)


def n_networks():
    """Number of network trainings: folds x seeds x (1 CODINE + 2 regime CODINE + 3 C-CODINE + 6 C-CopFlow) = 144.

    Returns:
    int: The number of network trainings.
    """
    return N_FOLDS * N_SEEDS * (1 + 2 + N_CCODINE_VARIANTS + N_CCOPFLOW_VARIANTS)


N_NETWORK_TRAININGS = n_networks()
assert N_NETWORK_TRAININGS == 144, N_NETWORK_TRAININGS


# ── pure gate logic (no I/O) ─────────────────────────────────────────────────


def _ks_statistics(x):
    """One-sample Kolmogorov-Smirnov distance to U(0,1) of every row of x.

    Parameters:
    x (np.ndarray): (n_tests, n) array of samples.

    Returns:
    np.ndarray: (n_tests,) array of KS distances.
    """
    x = np.sort(x, axis=1)
    n = x.shape[1]
    i = np.arange(1, n + 1)
    return np.maximum((i / n - x).max(axis=1), (x - (i - 1) / n).max(axis=1))


@functools.lru_cache(maxsize=None)
def ks_null_quantile(n_tests=KS_REF_TESTS, grid=False, q=KS_NULL_QUANTILE, reps=KS_NULL_REPS):
    """Quantile of the null distribution of D_max, the maximum over n_tests KS distances.

    Null: every margin is exactly uniform and the draws come from the sampler that the check uses.
    grid=False: exact iid uniform draws (parametric copulas, Acar, flows).
    grid=True: the grid sampler of CODINE / C-CODINE (KS_GRID_N cells per dimension, uniform jitter,
    clipped to [0.001, 0.999]) applied to a flat density, so its discretisation is included.
    The seed is fixed, so the value is reproducible.

    Parameters:
    n_tests (int): Number of KS tests whose maximum is taken. Default KS_REF_TESTS.
    grid (bool): Use the grid sampler instead of exact uniform draws. Default False.
    q (float): Quantile to return. Default KS_NULL_QUANTILE.
    reps (int): Monte Carlo replicates of the null distribution. Default KS_NULL_REPS.

    Returns:
    float: The q-quantile of D_max under the null.
    """
    rng = np.random.default_rng(20261014 + 7 * int(n_tests) + int(bool(grid)))
    n = KS_N_SAMPLES
    if grid:
        g = np.linspace(0.005, 0.995, KS_GRID_N)
        step = g[1] - g[0]
    out = np.empty(int(reps))
    for r in range(int(reps)):
        if grid:
            x = g[rng.integers(0, KS_GRID_N, (n_tests, n))] + rng.uniform(-step / 2, step / 2, (n_tests, n))
            x = np.clip(x, 0.001, 0.999)
        else:
            x = rng.uniform(size=(n_tests, n))
        out[r] = _ks_statistics(x).max()
    return float(np.quantile(out, q))


def ks_threshold(n_tests=KS_REF_TESTS, grid=False):
    """Admissibility threshold of a check with n_tests KS tests and the given sampler.

    The reference check (KS_REF_TESTS tests, exact sampler) has threshold KS_D_MAX. A check with more
    tests, or a sampler with discretisation error, has a larger null D_max, so its threshold is raised
    by the difference of the null quantiles (never below KS_D_MAX). All checks then have the same margin
    between the null distribution and the threshold.

    Parameters:
    n_tests (int): Number of KS tests of the check. Default KS_REF_TESTS.
    grid (bool): Whether the check uses the grid sampler. Default False.

    Returns:
    float: The threshold for D_max.
    """
    return max(KS_D_MAX, KS_D_MAX + ks_null_quantile(int(n_tests), bool(grid)) - ks_null_quantile(KS_REF_TESTS, False))


def check_ks_calibration(raise_on_fail=True):
    """Apply the pre-declared calibration rule to KS_D_MAX and return a small report.

    Simulated null of the reference check (2 tests, exact sampler): 99th and 99.9th percentile. Rule: KS_D_MAX
    must be at least the 99.9th percentile; otherwise raise it to that percentile rounded up to 0.005. The
    thresholds actually applied to the other checks (14 tests, grid sampler) are listed with their null
    percentiles so the margin is visible.

    Parameters:
    raise_on_fail (bool): Raise RuntimeError if KS_D_MAX is below the simulated 99.9th percentile. Default True.

    Returns:
    dict: ref_null_q999, ks_d_max, recommended, ok (bool) and table (list of dicts with n_tests, grid_sampler,
        null_q99, null_q999, threshold).
    """
    ref999 = ks_null_quantile(KS_REF_TESTS, False, 0.999, KS_CALIB_REPS)
    recommended = max(KS_D_MAX, math.ceil(ref999 / 0.005) * 0.005) if KS_D_MAX < ref999 else KS_D_MAX
    rows = []
    for n_tests, grid in ((KS_REF_TESTS, False), (2 * len(KS_PERCENTILES), False),
                          (KS_REF_TESTS, True), (2 * len(KS_PERCENTILES), True)):
        rows.append(dict(n_tests=n_tests, grid_sampler=grid,
                         null_q99=ks_null_quantile(n_tests, grid),
                         null_q999=ks_null_quantile(n_tests, grid, 0.999, KS_CALIB_REPS),
                         threshold=ks_threshold(n_tests, grid)))
    report = dict(ref_null_q999=ref999, ks_d_max=KS_D_MAX, recommended=recommended,
                  ok=bool(KS_D_MAX >= ref999), table=rows)
    if raise_on_fail and not report["ok"]:
        raise RuntimeError(f"KS_D_MAX={KS_D_MAX} is below the simulated 99.9th percentile of the null "
                           f"({ref999:.4f}); the pre-declared rule says use {recommended:.3f}.")
    return report


def ks_seed_pass(d_max, n_tests=KS_REF_TESTS, grid=False):
    """Decide whether one seed's D_max passes the gate.

    Parameters:
    d_max (float or None): KS distance of the seed. A missing or non-numeric value counts as failure.
    n_tests (int): Number of KS tests of the check. Default KS_REF_TESTS.
    grid (bool): Whether the check uses the grid sampler. Default False.

    Returns:
    bool: True iff d_max is a finite number <= ks_threshold(n_tests, grid).
    """
    if d_max is None:
        return False
    try:
        d_max = float(d_max)
    except (TypeError, ValueError):
        return False
    return math.isfinite(d_max) and d_max <= ks_threshold(n_tests, grid)


# ── calibration script ───────────────────────────────────────────────────────
def analytic_null(n_reps=2000, n=KS_N_SAMPLES, seed=11):
    """Null distribution of D_max for analytic copulas sampled through the pyvinecopulib sampler.

    Draws n samples from Gaussian(0.5), Frank(5), Clayton(2) and Gumbel(2) copulas, n_reps times each, and
    summarises the maximum KS distance of the two margins.

    Parameters:
    n_reps (int): Replicates per copula. Default 2000.
    n (int): Sample size per replicate. Default KS_N_SAMPLES.
    seed (int): Base seed; replicate r uses seed + r. Default 11.

    Returns:
    pandas.DataFrame: One row per copula with q50, q99, q999, max and the share of replicates above KS_D_MAX.
    """
    import pyvinecopulib as pv
    from copula_families import sample_bicop
    fams = {
        "Gaussian(0.5)": pv.Bicop(pv.BicopFamily.gaussian, parameters=np.array([[0.5]])),
        "Frank(5)": pv.Bicop(pv.BicopFamily.frank, parameters=np.array([[5.0]])),
        "Clayton(2)": pv.Bicop(pv.BicopFamily.clayton, parameters=np.array([[2.0]])),
        "Gumbel(2)": pv.Bicop(pv.BicopFamily.gumbel, parameters=np.array([[2.0]])),
    }
    rows = []
    for name, cop in fams.items():
        d = np.empty(n_reps)
        for r in range(n_reps):
            x = np.asarray(sample_bicop(cop, n, seed + r)).T               # (2, n)
            d[r] = _ks_statistics(x).max()
        rows.append(dict(copula=name, n_reps=n_reps, q50=np.quantile(d, .5), q99=np.quantile(d, .99),
                         q999=np.quantile(d, .999), max=d.max(), share_above_threshold=float((d > KS_D_MAX).mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rep = check_ks_calibration(raise_on_fail=False)
    print(f"KS_D_MAX = {rep['ks_d_max']}; simulated 99.9th percentile of the reference null = {rep['ref_null_q999']:.4f}; "
          f"rule satisfied: {rep['ok']} (recommended value {rep['recommended']:.3f})")
    print(pd.DataFrame(rep["table"]).round(4).to_string(index=False))
    try:
        print("\nAnalytic copulas through the pyvinecopulib sampler (2 KS tests):")
        print(analytic_null(n_reps).round(4).to_string(index=False))
    except Exception as exc:                                                  # pyvinecopulib missing
        print("analytic-copula check skipped:", exc)
