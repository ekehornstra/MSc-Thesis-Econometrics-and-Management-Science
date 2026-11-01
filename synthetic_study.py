"""synthetic_study.py

Simulation study with known truth.

Ten data-generating processes (DGPs), one change at a time (family OR shape of tau(w), never both), each at two strengths:

  3a  t       monotone  weak    3b  t       monotone  strong
  3c  t       hump      weak    3d  t       hump      strong
  3e  Gumbel  monotone  weak    3f  Gumbel  monotone  strong
  3g  Clayton monotone  weak    3h  Clayton monotone  strong
  3i  Gumbel-Clayton mixture, monotone, weak    3j  the same mixture, monotone, strong

The mixture is 50% Gumbel and 50% Clayton, both components with Kendall tau(w). The mixture's own tau equals tau(w) to within
0.003 (checked numerically). It has upper AND lower tail dependence of different strength and lies outside all seven
candidate families, so every fitted family is misspecified.

Weak: tau(w) from about 0.005 to 0.10 over w in (-1.5, 1.5) (rho = 0.08 (1 + tanh w), tau = 2/pi arcsin rho).
Strong: tau(w) from 0 to 0.91 (tau = 0.91 (tanh(1.75 w) + tanh(2.625)) / (2 tanh(2.625))); there is no sign change, so no
rotation and the tail corner of every family stays fixed.
Hump (Student-t family with df = 4; it isolates the SHAPE of tau(w)): tau(w) = tau_lo + (tau_hi - tau_lo) b(w), with
b(w) = (exp(-w^2 / (2 0.75^2)) - e) / (1 - e) and e = exp(-1.5^2 / (2 0.75^2)), so b = 1 at w = 0 and b = 0 at |w| = 1.5.
(tau_lo, tau_hi) = (0, 0.10) weak and (0, 0.91) strong. A median split of w is blind to a symmetric hump by construction,
because both halves have the same average tau.
All designs have tau >= 0, so no copula is rotated and the tail corner of each family is fixed: Gumbel has upper-tail
dependence, Clayton lower-tail dependence, the mixture both. At tau = 0 Gumbel and Clayton reduce to independence, so the
curve is continuous.

Candidate families (copula_families.py): Gaussian, t, Clayton, Gumbel, Gumbel (180 degrees), Frank, Joe. Clayton, Gumbel,
Gumbel (180 degrees) and Joe are fitted as signed families (own rotation for tau >= 0, turned branch for tau < 0, chosen by
maximum likelihood and not counted in AIC); with tau >= 0 throughout, the own branch is the relevant one.

Models fitted per DGP and training seed: seven parametric families unconditional; binary split at the median of w
(7 x 7 = 49 pairs by indexing, AIC-selected pair); Acar continuous with the same seven families; unconditional CODINE and the
two-net split CODINE; 3 C-CODINE and 6 C-CopFlow nets (74 candidates in total). n = N_TRAIN training draws (2,000 by
default). Monte-Carlo replication: MC_REPS independent data draws per DGP (default 10), one training seed per neural net and
draw, so the spread across draws includes the training noise. Metrics are reported as means over the draws with the SD across
draws. Evaluation is on one fixed fresh out-of-sample draw of 10,000 points.

Metrics: oracle CVPE and each model's excess over it; tau(w) RMSE; finite-threshold tail RMSE (chi(q = 0.95) per corner:
upper-upper, lower-lower, mixed) against the DGP's true copula; Copula Energy Score (ces_score.py).

References:
Coles, S., Heffernan, J., & Tawn, J. (1999). Dependence measures for extreme value analyses. Extremes, 2(4), 339-365.
Demarta, S., & McNeil, A. J. (2005). The t copula and related copulas. International Statistical Review, 73(1), 111-129.
Joe, H. (2014). Dependence modeling with copulas. CRC Press.
Kendall, M. G. (1938). A new measure of rank correlation. Biometrika, 30(1-2), 81-93.
Nelsen, R. B. (2006). An introduction to copulas (2nd ed.). Springer.
Ziel, F., & Berk, K. (2019). Multivariate forecasting evaluation: On sensitive and strictly proper scoring rules. arXiv
    preprint arXiv:1910.07325.
"""
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.stats import kendalltau
from scipy.stats import t as tdist

import ces_score as CES
import fold_pipeline as FP

# Training draw size; the environment variable THESIS_SYNTH_N overrides the default of 2,000.
N_TRAIN = int(os.environ.get("THESIS_SYNTH_N", 2_000))
W_RANGE = (-1.5, 1.5)
DF_T = 4.0
HUMP_SIGMA = 0.75
RHO_MAX = 0.999           # numerical guard only: sin(pi * 0.91 / 2) = 0.990 is below it, so the strong designs reach tau = 0.91
N_OOS = 10_000
OOS_SEED = 7777
Q_TAIL = 0.95     # threshold q of the corner chi(q); must be one of the study-wide thresholds in gate_config.Q_GRID
assert Q_TAIL in FP.GC.Q_GRID, "Q_TAIL is not a threshold of gate_config.Q_GRID"
W_EVAL = np.linspace(-1.4, 1.4, 15)        # grid for the tau / tail RMSE metrics
W_PLOT = np.linspace(W_RANGE[0], W_RANGE[1], 61)   # fine grid over the whole support, used only for the tau(w) figure
DATA_SEED = 42
MC_REPS = 10          # independent data draws per DGP (light: one training seed per net and draw)
TRAIN_SEEDS = (0,)

# Weight decay and all other hyper-parameters are those of the real-data pipeline: the study asks how the same procedure
# behaves under known truth, so nothing is re-tuned per DGP. The weight decay is the calibrated, architecture-group value of
# the real-data pipeline (fold_pipeline.fixed_wd_by_model: one calibrated value per group, from notebook 03).
DEFAULT_WD = FP.fixed_wd_by_model()

DGPS = {
    "3a": dict(family="t", shape="monotone", strength="weak"),
    "3b": dict(family="t", shape="monotone", strength="strong"),
    "3c": dict(family="t", shape="hump", strength="weak"),
    "3d": dict(family="t", shape="hump", strength="strong"),
    "3e": dict(family="gumbel", shape="monotone", strength="weak"),
    "3f": dict(family="gumbel", shape="monotone", strength="strong"),
    "3g": dict(family="clayton", shape="monotone", strength="weak"),
    "3h": dict(family="clayton", shape="monotone", strength="strong"),
    "3i": dict(family="gumbel_clayton", shape="monotone", strength="weak"),
    "3j": dict(family="gumbel_clayton", shape="monotone", strength="strong"),
}
MIX_PI = 0.5          # weight of the Gumbel component in the mixture DGPs


# ── true tau(w), copula at w, sampling ────────────────────────────────────
def tau_true(w, spec):
    """True Kendall tau as a function of the covariate w for one design.

    Weak monotone: tau = 2/pi arcsin(0.08 (1 + tanh w)), the Gaussian-copula/elliptical relation between tau and rho
    (Demarta & McNeil, 2005, eq. 9). Strong monotone: 0.91 (tanh(1.75 w) + e) / (2 e) with e = tanh(1.75 max|W_RANGE|), which is exactly 0 at
    the lower and exactly 0.91 at the upper edge of W_RANGE. Hump: a Gaussian bump (HUMP_SIGMA) rescaled to 1 at w = 0 and 0 at
    the edges, mapped to [tau_lo, tau_hi].

    Parameters:
    w (float or np.ndarray): Covariate value(s).
    spec (dict): Design with the keys 'shape' ('monotone' or 'hump') and 'strength' ('weak' or 'strong').

    Returns:
    np.ndarray: Kendall tau at each w (a 0-d array for a scalar w).

    Raises:
    ValueError: If spec['shape'] is unknown.
    """
    w = np.asarray(w, float)
    strength, shape = spec["strength"], spec["shape"]
    if shape == "monotone":
        if strength == "weak":
            return (2.0 / np.pi) * np.arcsin(0.08 * (1.0 + np.tanh(w)))
        # strong: tau rises from exactly 0 at w = -1.5 to exactly 0.91 at w = +1.5
        edge = np.tanh(1.75 * max(abs(x) for x in W_RANGE))
        return 0.91 * (np.tanh(1.75 * w) + edge) / (2.0 * edge)
    if shape == "hump":
        lo, hi = (0.0, 0.10) if strength == "weak" else (0.0, 0.91)
        # Gaussian bump rescaled to exactly 1 at w = 0 and exactly 0 at the edges |w| = max|W_RANGE|:
        # tau runs from tau_hi (centre) to tau_lo (edges).
        edge = np.exp(-max(abs(x) for x in W_RANGE) ** 2 / (2 * HUMP_SIGMA ** 2))
        bump = (np.exp(-w ** 2 / (2 * HUMP_SIGMA ** 2)) - edge) / (1.0 - edge)
        return lo + (hi - lo) * bump
    raise ValueError(shape)


class MixtureCop:
    """Mixture pi * Gumbel + (1 - pi) * Clayton, both components with the same Kendall tau.

    The components are rotated by 90 degrees when tau < 0. The object has pdf and cdf like a pyvinecopulib Bicop, and hinv1 for
    sampling, where the component is picked first. The Gumbel parameter is 1 / (1 - tau) and the Clayton parameter is
    2 tau / (1 - tau) (Nelsen, 2006; Joe, 2014).

    Parameters:
    a (float): Kendall tau of both components (absolute value), capped at 0.92 because the Clayton parameter is bounded by 28 in
        pyvinecopulib.
    rot (int): Rotation in degrees, 0 or 90.
    pi (float): Weight of the Gumbel component. Default MIX_PI.
    """

    def __init__(self, a, rot, pi=MIX_PI):
        """Build the Gumbel and Clayton components with the same Kendall tau.

        Parameters:
        a (float): Kendall tau of both components (absolute value), capped at 0.92.
        rot (int): Rotation in degrees, 0 or 90.
        pi (float): Weight of the Gumbel component. Default MIX_PI.
        """
        a = min(a, 0.92)                                  # Clayton theta <= 28 in pyvinecopulib
        self.pi = pi
        self.g = pv.Bicop(family=pv.BicopFamily.gumbel, rotation=rot,
                          parameters=np.array([[max(1.0 / (1 - a), 1.0 + 1e-4)]]))
        self.c = pv.Bicop(family=pv.BicopFamily.clayton, rotation=rot,
                          parameters=np.array([[max(2 * a / (1 - a), 1e-4)]]))

    def pdf(self, pts):
        """Mixture density.

        Parameters:
        pts (np.ndarray): (n, 2) array of points in the unit square.

        Returns:
        np.ndarray: Density at each point.
        """
        return self.pi * self.g.pdf(pts) + (1 - self.pi) * self.c.pdf(pts)

    def cdf(self, pts):
        """Mixture distribution function.

        Parameters:
        pts (np.ndarray): (n, 2) array of points in the unit square.

        Returns:
        np.ndarray: C(u1, u2) at each point.
        """
        return self.pi * self.g.cdf(pts) + (1 - self.pi) * self.c.cdf(pts)

    def hinv1(self, pts, gumbel):
        """Conditional inverse of U2 given U1 under the chosen mixture component.

        Parameters:
        pts (np.ndarray): (n, 2) array whose columns are U1 and the conditional probability level.
        gumbel (np.ndarray): Boolean per row; True takes the Gumbel component, False the Clayton component.

        Returns:
        np.ndarray: The U2 values.
        """
        pts = np.asarray(pts, float)
        return np.where(gumbel, np.asarray(self.g.hinv1(pts)), np.asarray(self.c.hinv1(pts)))


def true_cop(tau, spec):
    """Copula of the DGP at one value of tau.

    Parameters:
    tau (float): Kendall tau. Negative values give a 90-degree rotation of Clayton, Gumbel and the mixture.
    spec (dict): Design with the key 'family' ('t', 'clayton', 'gumbel' or 'gumbel_clayton').

    Returns:
    pv.Bicop or MixtureCop: The copula. The Student-t copula has df = DF_T and rho = sin(pi tau / 2) clipped to
        +-RHO_MAX.

    Raises:
    ValueError: If spec['family'] is unknown.
    """
    fam = spec["family"]
    if fam == "gumbel_clayton":
        return MixtureCop(abs(float(tau)), 90 if tau < 0 else 0)
    if fam == "t":
        rho = float(np.clip(np.sin(np.pi * tau / 2), -RHO_MAX, RHO_MAX))
        return pv.Bicop(family=pv.BicopFamily.student, parameters=np.array([[rho], [DF_T]]))
    a = min(abs(float(tau)), 0.97)
    rot = 90 if tau < 0 else 0
    if fam == "clayton":
        theta = max(2 * a / (1 - a), 1e-4)
        f = pv.BicopFamily.clayton
    elif fam == "gumbel":
        theta = max(1.0 / (1 - a), 1.0 + 1e-4)
        f = pv.BicopFamily.gumbel
    else:
        raise ValueError(fam)
    return pv.Bicop(family=f, rotation=rot, parameters=np.array([[theta]]))


def covariate_stream(n, seed, w_range=W_RANGE):
    """Random generator of a training draw together with its covariate, the first random numbers the generator produces.

    draw_dgp continues with this generator for the remaining variables, and notebooks that need only the training covariate of a
    draw (the uniformity check of AAA_summary) take it from here, so both always see the same w.

    Parameters:
    n (int): Sample size.
    seed (int): Seed of the random generator.
    w_range (tuple): Support of the covariate. Default W_RANGE.

    Returns:
    tuple: (rng, w) with rng a numpy Generator positioned after the draw of w and w a float array of length n.
    """
    rng = np.random.default_rng(seed)
    return rng, rng.uniform(*w_range, size=n)


def draw_dgp(n, spec, seed, w_range=W_RANGE):
    """Draw one training sample (u1, u2, w) from a DGP.

    The covariate is uniform on w_range. The Student-t design uses the conditional Student-t sampler of codine (one rho
    per observation). Clayton, Gumbel and the mixture use conditional inversion (hinv1) with one Bicop per observation, rotated by
    90 degrees where tau < 0. The pseudo-observations are clipped to [1e-6, 1 - 1e-6].

    Parameters:
    n (int): Sample size.
    spec (dict): Design, see tau_true and true_cop.
    seed (int): Seed of the random generator.
    w_range (tuple): Support of the covariate. Default W_RANGE.

    Returns:
    tuple: (u1, u2, w, tau) as float arrays of length n, with tau the true Kendall tau at each w.
    """
    rng, w = covariate_stream(n, seed, w_range)
    tau = tau_true(w, spec)
    if spec["family"] == "t":
        from codine import simulate_conditional_student_t_copula
        rho_fn = lambda x: np.sin(np.pi * tau_true(x, spec) / 2)
        u1, u2, _, _ = simulate_conditional_student_t_copula(n, w, rho_fn, df=DF_T, seed=int(rng.integers(1, 2 ** 31 - 1)))
        return u1.astype(float), u2.astype(float), w, tau
    a = rng.uniform(size=n); v = rng.uniform(size=n)
    u1 = a.copy(); u2 = np.empty(n)
    pick_g = rng.uniform(size=n) < MIX_PI
    for i in range(n):
        cop = true_cop(tau[i], spec)
        if spec["family"] == "gumbel_clayton":
            u2[i] = float(np.asarray(cop.hinv1(np.array([[a[i], v[i]]]), pick_g[i])).ravel()[0])
        else:
            u2[i] = float(np.asarray(cop.hinv1(np.array([[a[i], v[i]]])))[0])
    eps = 1e-6
    return np.clip(u1, eps, 1 - eps), np.clip(u2, eps, 1 - eps), w, tau


def check_tau_in_bins(u1, u2, w, spec, n_bins=8):
    """Empirical Kendall tau in bins of w against the formula. Run before fitting any model.

    The columns are w_mid, n, tau_empirical, tau_formula (bin mean of the DGP's tau(w)) and tau_se. tau_se is the standard error
    of Kendall's tau under independence, sqrt(2 (2n + 5) / (9 n (n - 1))) (Kendall, 1938, Section 13, Eq. 6); for dependent data it is an
    approximation that is conservative for large |tau|. The notebooks compare the largest absolute deviation with the absolute
    tolerance 0.12 and report the z-score; the tolerance is a plausibility screen for the sampler, not a test. Within a bin tau(w)
    varies, so the empirical tau of the pooled bin differs slightly from the mean of tau(w) (most for the strong designs).

    Parameters:
    u1 (np.ndarray): First pseudo-observation.
    u2 (np.ndarray): Second pseudo-observation.
    w (np.ndarray): Covariate.
    spec (dict): Design, see tau_true.
    n_bins (int): Number of equal-width bins of w. Default 8. Bins with fewer than 20 points are skipped.

    Returns:
    pd.DataFrame: One row per bin.
    """
    edges = np.linspace(w.min(), w.max(), n_bins + 1)
    rows = []
    for k in range(n_bins):
        m = (w >= edges[k]) & (w < edges[k + 1] + (1e-12 if k == n_bins - 1 else 0))
        if m.sum() < 20:
            continue
        wm = float(w[m].mean())
        # tau(w) varies within a bin, so the formula value is the bin mean of tau(w)
        nb = int(m.sum())
        se = float(np.sqrt(2.0 * (2 * nb + 5) / (9.0 * nb * (nb - 1))))     # s.e. of Kendall tau under independence (upper bound for |tau| > 0)
        rows.append(dict(w_mid=wm, n=nb, tau_empirical=float(kendalltau(u1[m], u2[m])[0]),
                         tau_formula=float(np.mean(tau_true(w[m], spec))), tau_se=se))
    return pd.DataFrame(rows)


def lambda_true(tau, spec):
    """Asymptotic tail-dependence coefficient of the DGP and the corner it belongs to.

    Student-t: lambda = 2 t_{df+1}(-sqrt((df + 1)(1 - rho)/(1 + rho))) (Demarta & McNeil, 2005, eq. 13). Clayton: 2^(-1/theta) in the lower
    corner. Gumbel: 2 - 2^(1/theta) in the upper corner (Joe, 2014). Mixture: the weighted coefficient of each part; the value
    returned is the larger of the two and the text gives both.

    Parameters:
    tau (float): Kendall tau.
    spec (dict): Design, see true_cop.

    Returns:
    tuple: (float, str): the coefficient and a text naming the corner(s). Negative tau moves the corners to UL and LU.
    """
    fam = spec["family"]
    if fam == "t":
        rho = float(np.clip(np.sin(np.pi * tau / 2), -RHO_MAX, RHO_MAX))
        lam = 2 * tdist.cdf(-np.sqrt((DF_T + 1) * (1 - rho) / (1 + rho)), df=DF_T + 1)
        return float(lam), "UU and LL" if tau >= 0 else "UL and LU"
    a = min(abs(tau), 0.97)
    if fam == "gumbel_clayton":
        a = min(a, 0.92)
        lg = MIX_PI * (2 - 2 ** (1 / max(1 / (1 - a), 1 + 1e-4)))
        lc = (1 - MIX_PI) * 2 ** (-1 / max(2 * a / (1 - a), 1e-4))
        return float(max(lg, lc)), (f"UU {lg:.3f} (Gumbel part), LL {lc:.3f} (Clayton part)" if tau >= 0
                                    else f"LU {lg:.3f} (Gumbel part), UL {lc:.3f} (Clayton part)")
    if fam == "clayton":
        th = max(2 * a / (1 - a), 1e-4); lam = 2 ** (-1 / th)
        return float(lam), ("LL" if tau >= 0 else "UL (U1 large, U2 small)")
    th = max(1 / (1 - a), 1 + 1e-4); lam = 2 - 2 ** (1 / th)
    return float(lam), ("UU" if tau >= 0 else "LU (U1 small, U2 large)")


# ── corner probabilities (finite-threshold tail coefficients) from a cdf or a density ───
CORNERS = ("UU", "LL", "UL", "LU")


def corner_chi_from_cdf(cdf, q=Q_TAIL):
    """Finite-threshold corner dependence chi(q) for the four corners of a copula.

    chi_corner(q) = P(corner) / (1 - q), with P(UU) = 1 - 2q + C(q, q), P(LL) = C(1 - q, 1 - q), P(UL) = (1 - q) - C(q, 1 - q) and
    P(LU) = (1 - q) - C(1 - q, q). For the upper-upper corner this is the conditional probability P(V > q | U > q), whose limit for q -> 1 defines
    chi in Coles et al. (1999, Section 3.3.1).

    Parameters:
    cdf (callable): cdf(a, b) -> C(a, b).
    q (float): Threshold. Default Q_TAIL.

    Returns:
    dict: 'UU', 'LL', 'UL', 'LU' -> chi value.
    """
    a = 1.0 - q
    return {"UU": (1 - 2 * q + cdf(q, q)) / a, "LL": cdf(a, a) / a,
            "UL": (a - cdf(q, a)) / a, "LU": (a - cdf(a, q)) / a}


def corner_chi_from_density(dens, w_value, q=Q_TAIL, n_cells=60):
    """The four corner quantities of corner_chi_from_cdf by midpoint integration of a density over the corner squares.

    Parameters:
    dens (callable): dens(points, w_value) -> density at the (n, 2) float32 points.
    w_value (float or None): Covariate value passed to dens (None for an unconditional density).
    q (float): Threshold. Default Q_TAIL.
    n_cells (int): Cells per side of each corner square. Default 60.

    Returns:
    dict: 'UU', 'LL', 'UL', 'LU' -> chi value.
    """
    a = 1.0 - q
    hi = q + a * (np.arange(n_cells) + 0.5) / n_cells
    lo = a * (np.arange(n_cells) + 0.5) / n_cells

    def rect(x, y):
        """Probability mass of the density on the rectangle of the cell centres x by y.

        Parameters:
        x (np.ndarray): Cell centres on the first axis.
        y (np.ndarray): Cell centres on the second axis.

        Returns:
        float: Midpoint sum of the density times the cell area, with negative density values set to zero.
        """
        g1, g2 = np.meshgrid(x, y, indexing="ij")
        pts = np.column_stack([g1.ravel(), g2.ravel()]).astype(np.float32)
        return float(np.sum(np.maximum(np.asarray(dens(pts, w_value), float), 0)) * (a / n_cells) ** 2)
    return {"UU": rect(hi, hi) / a, "LL": rect(lo, lo) / a, "UL": rect(hi, lo) / a, "LU": rect(lo, hi) / a}


def _bicop_cdf(cop):
    """Wrap the cdf of a pyvinecopulib-like copula as a scalar function cdf(x, y).

    Parameters:
    cop: Copula with a cdf method.

    Returns:
    callable: Function (x, y) -> float.
    """
    return lambda x, y: float(np.asarray(cop.cdf(np.array([[x, y]])))[0])


def true_corner_curves(spec, w_grid=W_EVAL, q=Q_TAIL):
    """True corner chi(q) of the DGP on a grid of covariate values.

    Parameters:
    spec (dict): Design, see tau_true.
    w_grid (np.ndarray): Covariate values. Default W_EVAL.
    q (float): Threshold. Default Q_TAIL.

    Returns:
    pd.DataFrame: Columns UU, LL, UL, LU; index w_grid.
    """
    rows = []
    for wv in w_grid:
        c = corner_chi_from_cdf(_bicop_cdf(true_cop(float(tau_true(wv, spec)), spec)), q)
        rows.append(c)
    return pd.DataFrame(rows, index=w_grid)


def tau_of_cop(cop):
    """Kendall tau of a fitted copula, or NaN if the object has no parameters_to_tau.

    Parameters:
    cop: Fitted copula.

    Returns:
    float: Kendall's tau.
    """
    return float(cop.parameters_to_tau(cop.parameters)) if hasattr(cop, "parameters_to_tau") else np.nan


# ── oracle and per-point loss ─────────────────────────────────────────────
def oracle_predictions(u1, u2, w, spec):
    """E[U1 | U2, w] and E[U2 | U1, w] under the true copula at each point's own w.

    Parameters:
    u1 (np.ndarray): First pseudo-observation.
    u2 (np.ndarray): Second pseudo-observation.
    w (np.ndarray): Covariate.
    spec (dict): Design, see tau_true.

    Returns:
    tuple: (e1, e2), the two conditional expectations at each point.
    """
    from copula_families import copula_conditional_expectation
    e1 = np.empty(len(u1)); e2 = np.empty(len(u1))
    tau = tau_true(w, spec)
    for i in range(len(u1)):
        E1, E2 = copula_conditional_expectation(true_cop(float(tau[i]), spec))
        e1[i] = E1(float(u2[i])); e2[i] = E2(float(u1[i]))
    return e1, e2


def loss_vec(u1, u2, e1, e2):
    """Per-point squared prediction loss (u1 - e1)^2 + (u2 - e2)^2.

    Parameters:
    u1 (array-like): First observed coordinate.
    u2 (array-like): Second observed coordinate.
    e1 (array-like): Predicted first coordinate.
    e2 (array-like): Predicted second coordinate.

    Returns:
    np.ndarray: The loss of each point.
    """
    return (np.asarray(u1) - e1) ** 2 + (np.asarray(u2) - e2) ** 2


# ── data bundle shared by all jobs of one DGP ─────────────────────────────
def make_oos(dgp_id, n_oos=N_OOS):
    """Fixed out-of-sample draw with the oracle loss, shared by all replications of one DGP.

    Parameters:
    dgp_id (str): Key of DGPS.
    n_oos (int): Number of out-of-sample points. Default N_OOS. The seed is OOS_SEED.

    Returns:
    dict: o1, o2, ow (float32), oracle_loss and true_corners (true corner chi curves on W_EVAL).
    """
    spec = DGPS[dgp_id]
    o1, o2, ow, _ = draw_dgp(n_oos, spec, OOS_SEED)
    oe1, oe2 = oracle_predictions(o1, o2, ow, spec)
    return dict(o1=o1, o2=o2, ow=ow.astype(np.float32), oracle_loss=loss_vec(o1, o2, oe1, oe2),
                true_corners=true_corner_curves(spec))


def make_bundle(dgp_id, n_train=N_TRAIN, n_oos=N_OOS, data_seed=DATA_SEED, rep=0, oos=None):
    """Training draw number `rep` (data seed data_seed + rep) plus the shared out-of-sample draw.

    Parameters:
    dgp_id (str): Key of DGPS.
    n_train (int): Training sample size. Default N_TRAIN.
    n_oos (int): Out-of-sample size, used only when oos is None.
    data_seed (int): Base seed of the training draws. Default DATA_SEED.
    rep (int): Replication index.
    oos (dict or None): Result of make_oos to reuse. Default None, which builds it.

    Returns:
    dict: dgp, rep, spec, u1, u2, w (float32), tau, w_median and the entries of the out-of-sample draw.
    """
    spec = DGPS[dgp_id]
    u1, u2, w, tau = draw_dgp(n_train, spec, data_seed + rep)
    oos = make_oos(dgp_id, n_oos) if oos is None else oos
    return dict(dgp=dgp_id, rep=rep, spec=spec, u1=u1, u2=u2, w=w.astype(np.float32), tau=tau,
                w_median=float(np.median(w)), **oos)


def make_bundles(dgp_id, reps=MC_REPS, n_train=N_TRAIN, n_oos=N_OOS):
    """Bundles of `reps` independent training draws that share one out-of-sample draw.

    Parameters:
    dgp_id (str): Key of DGPS.
    reps (int): Number of replications. Default MC_REPS.
    n_train (int): Training sample size. Default N_TRAIN.
    n_oos (int): Out-of-sample size. Default N_OOS.

    Returns:
    list: `reps` bundles, see make_bundle.
    """
    oos = make_oos(dgp_id, n_oos)
    return [make_bundle(dgp_id, n_train, n_oos, rep=r, oos=oos) for r in range(reps)]


@dataclass(eq=False)
class SynthJob(FP._JobAccess):
    """One unit of work of the simulation study on one data draw.

    The kind is 'param_acar' (the parametric and Acar fits) or one network ('codine_uncond', 'codine_regime', 'ccodine',
    'ccopflow'). Dict-style access is inherited from fold_pipeline._JobAccess.

    Parameters:
    bundle (dict): Data draw, see make_bundle.
    kind (str): Job kind, see above.
    model (str): Model name (networks only).
    seed (int): Training seed (networks only).
    wd (float): Weight decay (networks only). Default None.
    regime (str): 'low' or 'high' for kind 'codine_regime'.
    hyper (dict): Hyper-parameters of the network (see fold_pipeline.default_hyper).
    smoke (bool): Reduced Acar grid for a plumbing test.
    n_eval (int): Evaluation points of the Acar calibration curve.
    """
    bundle: dict
    kind: str = ""
    model: str = ""
    seed: int = 0
    wd: float = None
    regime: str = None
    hyper: dict = field(default_factory=dict)
    smoke: bool = False
    n_eval: int = FP.GC.ACAR_N_EVAL


# ── parametric and Acar models (one job per DGP, deterministic fits) ──────
def _cop_fit(family_name, u1, u2):
    """Fit one candidate family, rotation included, by maximum likelihood.

    Parameters:
    family_name (str): Key of the candidate family (copula_families).
    u1 (np.ndarray): First pseudo-observation.
    u2 (np.ndarray): Second pseudo-observation.

    Returns:
    tuple: (fitted copula, AIC).
    """
    from copula_families import fit_family
    cop = fit_family(family_name, np.column_stack([u1, u2]))
    ll = cop.loglik(np.column_stack([u1, u2]))
    return cop, -2 * ll + 2 * cop.npars


def parametric_acar_job(job):
    """Fit the parametric and Acar models of one data draw and score them out of sample.

    The models are the seven unconditional families, the binary split at the median of w (49 family pairs, with the AIC-selected
    pair) and the Acar continuous estimator for each of the seven families (bandwidth by purged cross-validation on the training
    draw). The fits are deterministic, so there is one job per draw. The Copula Energy Score is computed on the same out-of-sample
    points; for parametric copulas the margins are exactly uniform, so its raw and margin-calibrated values coincide.

    Parameters:
    job (SynthJob): Job of kind 'param_acar'; job['bundle'] is the data draw.

    Returns:
    dict: 'models' (name -> group, cvpe, excess, tau_curve and chi curves on W_EVAL, aic, tau_plot on W_PLOT), 'selected_pair',
        'selected_uncond', 'ces' (name -> raw, cal), 'failed' and 'ces_failed' (only if present) and 'runtime'.
    """
    from copula_families import DEFAULT_FAMILIES, FAMILY_DISPLAY_NAMES, copula_expectations
    t0 = time.time()
    B = job["bundle"]; w_grid = W_EVAL
    u1, u2, w = B["u1"], B["u2"], B["w"].astype(float)
    o1, o2, ow = B["o1"], B["o2"], B["ow"].astype(float)
    low_tr = w < B["w_median"]; low_oos = ow < B["w_median"]
    fams = list(DEFAULT_FAMILIES)
    out = {"models": {}}

    def record(name, group, loss_oos, tau_curve, chi_curves, aic=np.nan, tau_plot=None):
        """Store the out-of-sample summary and the tau and chi curves of one model in out['models'].

        Parameters:
        name (str): Model name.
        group (str): Model group.
        loss_oos (np.ndarray): Out-of-sample loss per point.
        tau_curve (array-like): Kendall's tau on the evaluation grid.
        chi_curves (dict): Corner chi curves on the evaluation grid.
        aic (float): AIC of the fit. Default NaN.
        tau_plot (array-like): Tau on the fine plotting grid. Default None.
        """
        out["models"][name] = dict(group=group, cvpe=float(np.mean(loss_oos)),
                                   excess=float(np.mean(loss_oos - B["oracle_loss"])),
                                   tau_curve=np.asarray(tau_curve), chi=chi_curves, aic=aic,
                                   tau_plot=None if tau_plot is None else np.asarray(tau_plot))

    def oos_loss(cop, mask=None):
        """Per-point OOS loss of one fixed copula on the OOS points in `mask` (NaN elsewhere).

        Parameters:
        cop: Fitted copula.
        mask (np.ndarray): Boolean mask of the out-of-sample points to score. Default None (all points).

        Returns:
        np.ndarray: Loss per out-of-sample point.
        """
        mask = np.ones(len(o1), bool) if mask is None else mask
        L = np.full(len(o1), np.nan)
        idx = np.flatnonzero(mask)
        e1, e2 = copula_expectations(cop, o1[idx], o2[idx])
        L[idx] = (o1[idx] - e1) ** 2 + (o2[idx] - e2) ** 2
        return L

    def chi_of(copf):
        """Corner chi curves on W_EVAL of the copula copf(w_value).

        Parameters:
        copf (callable): Covariate value -> copula.

        Returns:
        dict: Corner ('UU', 'LL', 'UL', 'LU') -> array over the evaluation grid.
        """
        return {c: np.array([corner_chi_from_cdf(_bicop_cdf(copf(wv)))[c] for wv in w_grid]) for c in CORNERS}

    # Copula Energy Score: deterministic quadrature on the same out-of-sample points (ces_score.py).
    o_pts = np.column_stack([o1, o2]); n_oos = len(o1); KQ = CES.DEFAULT_K
    ces = {}; ces_failed = []

    def ces_set(name, total, n):
        """Store the mean Copula Energy Score (raw = calibrated for parametric copulas) of one model.

        Parameters:
        name (str): Model name.
        total (float): Sum of the energy score over the out-of-sample points.
        n (int): Number of points.
        """
        ces[name] = dict(raw=float(total / n), cal=float(total / n))

    def es_sum(cop, mask):
        """Sum of the energy score of a fixed copula over the out-of-sample points in mask.

        Parameters:
        cop: Fitted copula.
        mask (np.ndarray): Boolean mask of the points.

        Returns:
        float: The summed score.
        """
        return float(np.sum(CES.es_from_masses(CES.masses_from_cdf(cop, KQ), o_pts[mask], KQ)))

    # unconditional
    unc, aic_u = {}, {}
    for f in fams:
        cop, aic = _cop_fit(f, u1, u2); nm = FAMILY_DISPLAY_NAMES[f]
        unc[nm] = cop; aic_u[nm] = aic
        L = oos_loss(cop)
        record(f"Unconditional: {nm}", "Unconditional", L, np.full(len(w_grid), tau_of_cop(cop)),
               chi_of(lambda wv, c=cop: c), aic, tau_plot=np.full(len(W_PLOT), tau_of_cop(cop)))
        try:
            ces_set(f"Unconditional: {nm}", es_sum(cop, np.ones(n_oos, bool)), n_oos)
        except Exception as exc:
            ces_failed.append((f"Unconditional: {nm}", repr(exc)))
    # binary split at the median of w
    reg = {}
    for rname, mtr in (("low", low_tr), ("high", ~low_tr)):
        reg[rname] = {}
        for f in fams:
            cop, aic = _cop_fit(f, u1[mtr], u2[mtr]); reg[rname][FAMILY_DISPLAY_NAMES[f]] = (cop, aic)
    reg_loss = {"low": {nm: oos_loss(c, low_oos) for nm, (c, _) in reg["low"].items()},
                "high": {nm: oos_loss(c, ~low_oos) for nm, (c, _) in reg["high"].items()}}
    pair_aic = {}
    ces_reg = {"low": {}, "high": {}}
    for _rn, _mask in (("low", low_oos), ("high", ~low_oos)):
        for _nm, (_c, _) in reg[_rn].items():
            try:
                ces_reg[_rn][_nm] = es_sum(_c, _mask)
            except Exception as exc:
                ces_failed.append((f"regime {_rn} {_nm}", repr(exc)))
    wmid = B["w_median"]
    for fl, (cl, al) in reg["low"].items():
        for fh, (ch, ah) in reg["high"].items():
            L = np.where(low_oos, reg_loss["low"][fl], reg_loss["high"][fh])
            copf = lambda wv, cl=cl, ch=ch: cl if wv < wmid else ch
            tc = np.array([tau_of_cop(copf(wv)) for wv in w_grid])
            nm = f"Divergence split: {fl} (low) / {fh} (high)"
            record(nm, "Binary", L, tc, chi_of(copf), al + ah, tau_plot=np.where(W_PLOT < wmid, tau_of_cop(cl), tau_of_cop(ch)))
            pair_aic[(fl, fh)] = al + ah
            if fl in ces_reg["low"] and fh in ces_reg["high"]:
                ces_set(nm, ces_reg["low"][fl] + ces_reg["high"][fh], n_oos)
    out["selected_pair"] = min(pair_aic, key=pair_aic.get)
    out["selected_uncond"] = min(aic_u, key=aic_u.get)
    # Acar continuous
    from conditional_copula import select_model, estimate_calibration_curve, fit_global
    from fold_models import make_acar_predictor
    bands = np.linspace(0.5, 1.1 * float(np.max(np.abs(w))), 3 if job.get("smoke") else 8).round(2)
    # Synthetic pairs are iid, so no purge is needed. The same 100-point subsample and relaxed optimiser as on the real data keep the selection light.
    for fam in FP.ACAR_FAMILIES:
        try:
            df_h, _, _ = select_model(u1, u2, w, families=(fam,), bandwidths=bands, purge=0, n_splits=3,
                                      bfgs_options={'gtol': 1e-3, 'maxiter': 50}, n_jobs=1, max_eval_points=100)
            sub = df_h[df_h.family == fam]
            if sub.empty or not np.isfinite(sub.cv_score.to_numpy(float)).any():
                raise RuntimeError("no Acar bandwidth with a finite CV score; candidate recorded as failed instead of "
                                   "silently using the first bandwidth")
            h = float(sub.loc[sub.cv_score.idxmax(), "bandwidth"])
            curve = estimate_calibration_curve(u1, u2, w, family=fam, h=h, n_eval=job.get("n_eval", FP.GC.ACAR_N_EVAL), ci_boot=0)
            _, fixed, _ = fit_global(u1, u2, fam)
            e1, e2 = make_acar_predictor(fam, {fam: curve}, {fam: fixed}, ow)(o1, o2)
            L = loss_vec(o1, o2, e1, e2)
            tc = np.interp(w_grid, curve["x"].values, curve["tau"].values)
            copf = lambda wv, c=curve, fx=fixed, fm=fam: FP._acar_cop_at(fm, c, fx, wv)
            record(FP.acar_name(fam), "Acar", L, tc, chi_of(copf),
                   tau_plot=np.interp(W_PLOT, curve["x"].values, curve["tau"].values))
            try:
                gw = np.linspace(float(ow.min()), float(ow.max()), CES.DEFAULT_GRID_POINTS)
                v = CES.ces_conditional(lambda wv: CES.masses_from_cdf(copf(wv), KQ), ow, o_pts, gw, KQ)
                ces_set(FP.acar_name(fam), float(np.sum(v)), n_oos)
            except Exception as exc:
                ces_failed.append((FP.acar_name(fam), repr(exc)))
        except Exception as e:                       # recorded; summarize() reports missing candidates
            warnings.warn(f"Acar {fam} failed on {B['dgp']}: {e}")
            out.setdefault("failed", []).append((fam, repr(e)))
    out["ces"] = ces
    if ces_failed:
        out["ces_failed"] = ces_failed
        warnings.warn(f"CES failed for {len(ces_failed)} items on {B['dgp']}: {ces_failed[:2]}")
    out["runtime"] = time.time() - t0
    return out


def oracle_ces(bundle, K=CES.DEFAULT_K, G=CES.DEFAULT_GRID_POINTS):
    """Per-point Copula Energy Score of the true conditional copula (oracle) and of independence on the bundle's out-of-sample draw.

    Used for reporting (excess CES) only, never for tuning or selection. It is computed here and not stored in the bundle.

    Parameters:
    bundle (dict): Data draw, see make_bundle.
    K (int): Grid size of the discretised copula. Default ces_score.DEFAULT_K.
    G (int): Number of covariate grid points. Default ces_score.DEFAULT_GRID_POINTS.

    Returns:
    dict: 'oracle' and 'independence' -> arrays of the score per out-of-sample point.
    """
    spec = bundle["spec"]
    o = np.column_stack([bundle["o1"], bundle["o2"]]); ow = np.asarray(bundle["ow"], float)
    gw = np.linspace(float(ow.min()), float(ow.max()), G)
    mf = lambda wv: CES.masses_from_cdf(true_cop(float(tau_true(wv, spec)), spec), K)
    return dict(oracle=CES.ces_conditional(mf, ow, o, gw, K), independence=CES.es_independence(o, K))


# ── neural nets: one job per (DGP, seed, net) ─────────────────────────────
def _kendall_tau_from_samples(samp):
    """Kendall tau of a sample given as an (n, 2) array.

    Parameters:
    samp (np.ndarray): Sample, shape (n, 2).

    Returns:
    float: Kendall's tau.
    """
    return float(kendalltau(samp[:, 0], samp[:, 1])[0])


def neural_synth_job(job, state_dir=None):
    """Train one network on a synthetic training draw and score it out of sample.

    The network is trained as in the real-data pipeline (fold_pipeline.train_net, validation purge 1) on all N_TRAIN points. The
    out-of-sample loss, the tau(w) curve and the corner chi curves (W_EVAL grid, and the W_PLOT grid for the first draw) are
    computed from the best-checkpoint network. With state_dir (set by run_batch) the fitted network is stored for the CES stage
    (fold_models.save_state).

    Parameters:
    job (SynthJob): Job of a network kind.
    state_dir (str or Path or None): The states directory of the batch. Default None, which stores nothing.

    Returns:
    dict: dgp, rep, seed, model, kind, regime, best_epoch, best_val_cvpe, loss_oos, tau_curve, chi and margin_ks (conditional nets) or
        tau_const and chi_const (CODINE nets), tau_plot (first draw), state (file info or error) and runtime.
    """
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    B, kind, model, seed = job["bundle"], job["kind"], job["model"], int(job["seed"])
    hp = FP._hyper(job)
    u1, u2, w = B["u1"], B["u2"], B["w"]
    n = len(u1)
    lowm = np.asarray(w < B["w_median"])
    rmask = np.ones(n, bool)
    if kind == "codine_regime":
        rmask = ~lowm if job["regime"] == "high" else lowm
    T = FP.train_net(kind, model, seed, u1, u2, w, rmask, hp, job.get("wd", 0.0), val_purge=1)
    obj, dens = T["obj"], T["dens"]
    state_info = None
    if state_dir:
        try:
            import fold_models as MS
            key = synth_key(job)
            side = MS.save_state(obj, MS.state_path(Path(state_dir).parent, key),
                                 meta=dict(key=key, dgp=B["dgp"], rep=B["rep"], seed=seed, model=model, kind=kind,
                                           regime=job.get("regime"), best_epoch=T["res"]["best_epoch"]))
            state_info = dict(file=f"{MS.STATE_SUBDIR}/{key}.pt", sha256=side["sha256"], version=MS.VERSION)
        except Exception as exc:                  # a storage problem must not discard a finished training
            state_info = dict(error=repr(exc))
            warnings.warn(f"state not stored for {model} seed {seed}: {exc!r}")
    o1, o2, ow = B["o1"], B["o2"], B["ow"]
    wg = W_EVAL
    out = dict(dgp=B["dgp"], rep=B["rep"], seed=seed, model=model, kind=kind, regime=job.get("regime"),
               best_epoch=T["res"]["best_epoch"], best_val_cvpe=T["res"]["best_val_cvpe"])
    if kind in ("codine_uncond", "codine_regime"):
        from fold_models import make_codine_predictor
        from codine import sample_from_codine
        sel = np.ones(len(o1), bool) if kind == "codine_uncond" else ((ow < B["w_median"]) if job["regime"] == "low" else (ow >= B["w_median"]))
        e1, e2 = make_codine_predictor(obj, grid_size=hp["grid_n"])(o1[sel].astype(np.float32), o2[sel].astype(np.float32))
        L = np.full(len(o1), np.nan); L[sel] = loss_vec(o1[sel], o2[sel], e1, e2)
        out["loss_oos"] = L
        samp = sample_from_codine(obj, n_samples=hp["ks_n_samples"] * 20, seed=1)
        out["tau_const"] = _kendall_tau_from_samples(np.asarray(samp))
        cc = corner_chi_from_density(dens, None, n_cells=hp["n_cells"])
        out["chi_const"] = cc
    else:
        from model_evaluation import neural_conditional_predictions
        e1, e2, _ = neural_conditional_predictions(dens, o1, o2, ow.astype(float), grid_n=hp["grid_n"])
        out["loss_oos"] = loss_vec(o1, o2, e1, e2)
        from scipy.stats import kstest
        tau_c, chi_c, ks_c = [], {c: [] for c in CORNERS}, []
        for wv in wg:
            smp = np.asarray(T["sample_fn"](float(wv), hp["ks_n_samples"] * 10, 1))
            tau_c.append(_kendall_tau_from_samples(smp))
            ks_c.append(max(kstest(smp[:, 0], "uniform").statistic, kstest(smp[:, 1], "uniform").statistic))
            cc = corner_chi_from_density(dens, float(wv), n_cells=hp["n_cells"])
            for c in CORNERS:
                chi_c[c].append(cc[c])
        out["margin_ks"] = np.array(ks_c)      # worst KS distance of the two margins to U(0,1), per w in W_EVAL
        out["tau_curve"] = np.array(tau_c); out["chi"] = {c: np.array(v) for c, v in chi_c.items()}
        if B["rep"] == 0:      # fine grid for the tau(w) figure, first draw only (the figure shows one draw)
            out["tau_plot"] = np.array([_kendall_tau_from_samples(np.asarray(T["sample_fn"](float(wv), hp["ks_n_samples"] * 5, 1)))
                                        for wv in W_PLOT])
    out["state"] = state_info
    out["runtime"] = time.time() - t0
    return out


neural_synth_job._ram_heavy = True      # run_batch applies the RAM-aware worker cap to this job
neural_synth_job._persist_state = True  # run_batch passes state_dir=<batch_dir>/states


def ces_synth_job(job):
    """Copula Energy Score of one stored synthetic network on the shared out-of-sample draw.

    The score is computed for the raw density and for the margin-calibrated one (Sinkhorn scaling to uniform margins). The fitted
    network is reloaded with fold_models.load_fitted; a job without a state file is reported as 'not reconstructable' and is not
    retrained.

    Parameters:
    job (dict): Job built by run_ces_synth (bundle, net_kind, model, seed, regime, K, G, batch_dir, key).

    Returns:
    dict: dgp, rep, seed, model, kind, regime, status, reason, es_raw and es_cal (per out-of-sample point; NaN outside the regime
        of a split-CODINE half), max_margin_error, min_raw_total, max_raw_total and runtime.
    """
    import torch
    torch.set_num_threads(1)
    import fold_models as MS
    import ces_score as CR
    t0 = time.time()
    B, kind, model, K, G = job["bundle"], job["net_kind"], job["model"], int(job["K"]), int(job["G"])
    out = dict(dgp=B["dgp"], rep=B["rep"], seed=int(job["seed"]), model=model, kind=kind, regime=job.get("regime"),
               status="ok", reason="", es_raw=None, es_cal=None)
    try:
        net, meta = MS.load_fitted(job["batch_dir"], job["key"])
    except MS.NotReconstructable as exc:
        out.update(status="not reconstructable", reason=str(exc)[:200], runtime=time.time() - t0)
        return out
    o = np.column_stack([B["o1"], B["o2"]]); ow = np.asarray(B["ow"], float)
    sel = np.ones(len(o), bool)
    if kind == "codine_regime":
        sel = (ow < B["w_median"]) if job["regime"] == "low" else (ow >= B["w_median"])
    dens = CR._density_fn(net, kind); diag = {}
    with torch.no_grad():
        if kind in ("codine_uncond", "codine_regime"):
            m, tot = CES.masses_from_density(dens, None, K)
            raw, cal = CES.es_from_masses_many([m, CES.sinkhorn_uniform_margins(m, K)], o[sel], K)
            diag = dict(margin_error=[CES.margin_error(m, K)], raw_total=[tot])
        else:
            gw = np.linspace(float(ow[sel].min()), float(ow[sel].max()), G)
            mf = lambda wv: CES.masses_from_density(dens, float(wv), K)
            raw, cal = CES.ces_conditional_both(mf, ow[sel], o[sel], gw, K, diagnostics=diag)
    full_raw = np.full(len(o), np.nan); full_raw[sel] = raw
    full_cal = np.full(len(o), np.nan); full_cal[sel] = cal
    out.update(es_raw=full_raw, es_cal=full_cal, max_margin_error=float(np.max(diag["margin_error"])),
               min_raw_total=float(np.min(diag["raw_total"])), max_raw_total=float(np.max(diag["raw_total"])),
               runtime=time.time() - t0)
    return out


ces_synth_job._ram_heavy = True


def ces_synth_key(j):
    """Batch key of the CES job that belongs to the network job with key j['key'].

    Parameters:
    j (dict): Network job.

    Returns:
    str: The key.
    """
    return "ces__" + j["key"]


def run_ces_synth(jobs, max_workers=None, K=CES.DEFAULT_K, G=CES.DEFAULT_GRID_POINTS, batch_dir=None):
    """Resumable CES stage over the neural jobs of build_neural_synth_jobs.

    Reads the stored networks and writes the results to <batch_dir>/ces.

    Parameters:
    jobs (list): Jobs of build_neural_synth_jobs.
    max_workers (int): Number of worker processes. Default None.
    K (int): Grid size of the discretised copula. Default ces_score.DEFAULT_K.
    G (int): Number of covariate grid points. Default ces_score.DEFAULT_GRID_POINTS.
    batch_dir (str or Path or None): Batch directory. Default <fold_pipeline.BATCH_DIR>/synthetic.

    Returns:
    dict: key -> result of ces_synth_job.
    """
    bd = Path(batch_dir) if batch_dir is not None else FP.BATCH_DIR / "synthetic"
    cj = [dict(kind="ces_synth", bundle=j["bundle"], net_kind=j["kind"], model=j["model"], seed=j["seed"],
               regime=j.get("regime"), K=K, G=G, batch_dir=str(bd), key=synth_key(j)) for j in jobs]
    return FP.run_batch(cj, max_workers=max_workers, batch_dir=bd / "ces", fn=ces_synth_job, key_fn=ces_synth_key,
                        desc="CES of synthetic networks")


def _neural_ces_by_model(ces_results, rep):
    """Mean raw and margin-calibrated CES per network seed, by model, for one replication.

    The two split-CODINE halves of a seed are merged; a seed with only one half is dropped. Failed jobs are skipped.

    Parameters:
    ces_results (dict): Results of run_ces_synth.
    rep (int): Replication index.

    Returns:
    dict: model -> list of (raw mean, calibrated mean), one per seed.
    """
    per = {}
    for r in ces_results.values():
        if r.get("rep") != rep or r["status"] != "ok":
            continue
        m = r["model"].split("|")[0] if r["model"].startswith(FP.SPLIT_CODINE) else r["model"]
        per.setdefault((m, r["seed"]), []).append(r)
    out = {}
    for (m, sd), rs in per.items():
        if m == FP.SPLIT_CODINE and len(rs) < 2:
            continue
        raw = np.vstack([x["es_raw"] for x in rs]); cal = np.vstack([x["es_cal"] for x in rs])
        fin = np.isfinite(raw).any(axis=0)
        out.setdefault(m, []).append((float(np.nansum(raw, 0)[fin].mean()), float(np.nansum(cal, 0)[fin].mean())))
    return out


def build_neural_synth_jobs(bundles, wd_by_model, seeds=TRAIN_SEEDS, smoke=False):
    """The 12 network jobs per training seed and bundle.

    The nets are the unconditional CODINE, the two halves of the split CODINE, the three C-CODINE variants and the six C-CopFlow
    variants.

    Parameters:
    bundles (dict or list): One bundle or a list of bundles (replications).
    wd_by_model (dict): model -> weight decay for the conditional nets (DEFAULT_WD).
    seeds (tuple): Training seeds. Default TRAIN_SEEDS.
    smoke (bool): Shrink the training budget for a plumbing test.

    Returns:
    list: SynthJob objects.
    """
    bundles = [bundles] if isinstance(bundles, dict) else list(bundles)
    hyper = FP.default_hyper()
    if smoke:
        hyper.update(max_epochs=40, checkpoint_every=20, codine_checkpoint_every=20, patience=2, n_cells=15,
                     ks_n_samples=100, grid_n=30)
    jobs = []
    for bundle in bundles:
        for s in seeds:
            base = dict(bundle=bundle, seed=s, hyper=hyper)
            jobs.append(SynthJob(**base, model=FP.UNCOND_CODINE, kind="codine_uncond"))
            for reg in ("low", "high"):
                jobs.append(SynthJob(**base, model=f"{FP.SPLIT_CODINE}|{reg}", kind="codine_regime", regime=reg))
            for v in FP.CCODINE_VARIANTS:
                jobs.append(SynthJob(**base, model=v, kind="ccodine", wd=wd_by_model[v]))
            for v in FP.CCOPFLOW_VARIANTS:
                jobs.append(SynthJob(**base, model=v, kind="ccopflow", wd=wd_by_model[v]))
    return jobs


def synth_key(j):
    """Batch key of one network job: DGP, replication, seed and model name.

    Parameters:
    j (dict): Network job with 'bundle', 'seed' and 'model'.

    Returns:
    str: The key.
    """
    return (f"{j['bundle']['dgp']}__r{j['bundle']['rep']}__s{j['seed']}__"
            f"{''.join(c if c.isalnum() else '_' for c in j['model'])}")


def param_key(bundle):
    """Batch key of the parametric and Acar job of one bundle.

    Parameters:
    bundle (dict): Synthetic bundle with 'dgp' and 'rep'.

    Returns:
    str: The key.
    """
    return f"{bundle['dgp']}__r{bundle['rep']}__param_acar"


# ── summaries ─────────────────────────────────────────────────────────────
def _curve_rmse(curve, truth):
    """Root mean squared difference between a curve and the truth.

    Parameters:
    curve (array-like): Estimated curve.
    truth (array-like): True curve on the same grid.

    Returns:
    float: The RMSE.
    """
    return float(np.sqrt(np.mean((np.asarray(curve) - np.asarray(truth)) ** 2)))


def summarize(bundle, param_out, neural_results, require_all=False, ces_results=None, ces_ref=None):
    """One row per model for one data draw.

    The columns are the oracle-excess CVPE, the tau RMSE, the tail RMSE (UU, LL and the mean of the two mixed corners), the peak
    location of tau(w) and, if ces_ref is given, the Copula Energy Score columns. Neural models are averaged over their training
    seeds with the seed SD of the excess; the parametric and Acar fits are deterministic. All gate_config.N_CANDIDATES_SYNTHETIC (74)
    candidates are expected; absent ones are reported with a warning, or an error if require_all.

    Parameters:
    bundle (dict): Data draw, see make_bundle.
    param_out (dict): Result of parametric_acar_job.
    neural_results (dict): Results of neural_synth_job, keyed by job key.
    require_all (bool): Raise instead of warn when candidates are missing.
    ces_results (dict or None): Results of run_ces_synth.
    ces_ref (dict or None): Result of oracle_ces; the CES columns are filled only if given.

    Returns:
    pd.DataFrame: Indexed by model, sorted by excess; attrs hold oracle_cvpe (and oracle_ces, independence_ces).

    Raises:
    RuntimeError: If candidates are missing and require_all is True.
    """
    spec = bundle["spec"]
    tt = tau_true(W_EVAL, spec)
    tc = bundle["true_corners"]
    rows = []

    def tail_metrics(chi):
        """Tail RMSEs against the true corner curves: (UU, LL, mean of UL and LU).

        Parameters:
        chi (dict): Estimated corner curves ('UU', 'LL', 'UL', 'LU').

        Returns:
        tuple: (RMSE of UU, RMSE of LL, mean of the RMSEs of UL and LU).
        """
        uu = _curve_rmse(chi["UU"], tc["UU"]); ll = _curve_rmse(chi["LL"], tc["LL"])
        mixed = 0.5 * (_curve_rmse(chi["UL"], tc["UL"]) + _curve_rmse(chi["LU"], tc["LU"]))
        return uu, ll, mixed

    for name, m in param_out["models"].items():
        uu, ll, mx = tail_metrics(m["chi"])
        rows.append(dict(model=name, group=m["group"], n_seeds=1, cvpe=m["cvpe"], excess=m["excess"],
                         excess_sd=np.nan, tau_rmse=_curve_rmse(m["tau_curve"], tt), tail_uu=uu, tail_ll=ll,
                         tail_mixed=mx, peak_w=float(W_EVAL[int(np.argmax(m["tau_curve"]))])))
    by_model = {}
    for r in neural_results.values():
        by_model.setdefault(r["model"], []).append(r)
    def per_seed_rows(model):
        """Out-of-sample loss, tau curve and chi curves of a model per seed; the split CODINE joins its two nets.

        Parameters:
        model (str): Model name.

        Returns:
        list: One entry per seed.
        """
        out = []
        if model == FP.SPLIT_CODINE:
            lows = {r["seed"]: r for r in by_model.get(f"{FP.SPLIT_CODINE}|low", [])}
            highs = {r["seed"]: r for r in by_model.get(f"{FP.SPLIT_CODINE}|high", [])}
            for s in sorted(set(lows) & set(highs)):
                L = np.where(np.isfinite(lows[s]["loss_oos"]), lows[s]["loss_oos"], highs[s]["loss_oos"])
                tcur = np.where(W_EVAL < bundle["w_median"], lows[s]["tau_const"], highs[s]["tau_const"])
                chi = {c: np.where(W_EVAL < bundle["w_median"], lows[s]["chi_const"][c], highs[s]["chi_const"][c]) for c in CORNERS}
                out.append((L, tcur, chi))
        else:
            for r in by_model.get(model, []):
                if "tau_curve" in r:
                    out.append((r["loss_oos"], r["tau_curve"], r["chi"]))
                else:
                    out.append((r["loss_oos"], np.full(len(W_EVAL), r["tau_const"]),
                                {c: np.full(len(W_EVAL), r["chi_const"][c]) for c in CORNERS}))
        return out
    for model in [FP.UNCOND_CODINE, FP.SPLIT_CODINE] + FP.NEURAL_MODELS:
        ps = per_seed_rows(model)
        if not ps:
            continue
        ex = [float(np.mean(L - bundle["oracle_loss"])) for L, _, _ in ps]
        cv = [float(np.mean(L)) for L, _, _ in ps]
        tr = [_curve_rmse(t, tt) for _, t, _ in ps]
        tl = [tail_metrics(c) for _, _, c in ps]
        pk = [float(W_EVAL[int(np.argmax(t))]) for _, t, _ in ps]
        group = "Unconditional" if model == FP.UNCOND_CODINE else "Binary" if model == FP.SPLIT_CODINE else "Neural"
        rows.append(dict(model=model, group=group, n_seeds=len(ps), cvpe=float(np.mean(cv)), excess=float(np.mean(ex)),
                         excess_sd=float(np.std(ex, ddof=1)) if len(ex) > 1 else np.nan,
                         tau_rmse=float(np.mean(tr)), tail_uu=float(np.mean([x[0] for x in tl])),
                         tail_ll=float(np.mean([x[1] for x in tl])), tail_mixed=float(np.mean([x[2] for x in tl])),
                         peak_w=float(np.mean(pk))))
    df = pd.DataFrame(rows).set_index("model")
    # Copula Energy Score columns (NaN = not computed)
    df["ces"] = np.nan; df["ces_calibrated"] = np.nan; df["ces_seeds"] = np.nan
    if ces_ref is not None:
        for name, c in (param_out.get("ces") or {}).items():
            if name in df.index:
                df.loc[name, ["ces", "ces_calibrated", "ces_seeds"]] = [c["raw"], c["cal"], 1]
        for m, lst in (_neural_ces_by_model(ces_results, bundle["rep"]) if ces_results else {}).items():
            if m in df.index:
                df.loc[m, ["ces", "ces_calibrated", "ces_seeds"]] = [np.mean([a for a, _ in lst]), np.mean([b for _, b in lst]), len(lst)]
        orc, ind = float(np.mean(ces_ref["oracle"])), float(np.mean(ces_ref["independence"]))
        df["excess_ces"] = df["ces"] - orc
        df["ces_vs_independence"] = ind - df["ces"]
        df.attrs["oracle_ces"], df.attrs["independence_ces"] = orc, ind
    missing = [m for m in FP.candidate_table(nonparametric=False).model if m not in df.index]
    if missing:
        msg = (f"summarize: {len(df)} of {FP.GC.N_CANDIDATES_SYNTHETIC} candidates present for DGP {bundle['dgp']} "
               f"rep {bundle['rep']}; missing: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
        if require_all:
            raise RuntimeError(msg)
        warnings.warn(msg)
    df.attrs["oracle_cvpe"] = float(np.mean(bundle["oracle_loss"]))
    return df.sort_values("excess")


def summarize_mc(bundles, param_outs, neural_results, ces_results=None):
    """Monte-Carlo summary over the replications.

    Parameters:
    bundles (list): Bundles of the replications.
    param_outs (dict): rep -> result of parametric_acar_job.
    neural_results (dict): Merged neural results of all replications.
    ces_results (dict or None): Results of run_ces_synth.

    Returns:
    tuple: (mean table, long table). The mean table has the columns of summarize averaged over the draws, `<metric>_sd` (SD across
        draws) for every metric, n_reps and excess_se. The long table has one row per model and draw.
    """
    long = []
    ces_ref = oracle_ces(bundles[0]) if any((param_outs[b["rep"]].get("ces") for b in bundles)) else None   # one shared OOS draw
    for b in bundles:
        r = b["rep"]
        sub = {k: v for k, v in neural_results.items() if v["rep"] == r}
        t = summarize(b, param_outs[r], sub, ces_results=ces_results, ces_ref=ces_ref).reset_index()
        t["rep"] = r
        long.append(t)
    long = pd.concat(long, ignore_index=True)
    metrics = ["cvpe", "excess", "tau_rmse", "tail_uu", "tail_ll", "tail_mixed", "peak_w"]
    ces_cols = [c for c in ("ces", "ces_calibrated", "excess_ces", "ces_vs_independence") if c in long.columns and long[c].notna().any()]
    metrics = metrics + ces_cols
    g = long.groupby("model")
    mean = g[metrics].mean()
    for m in metrics:
        mean[f"{m}_sd"] = g[m].std(ddof=1)
    mean["n_reps"] = g["rep"].nunique()
    mean.insert(0, "group", g["group"].first())
    mean["excess_se"] = mean["excess_sd"] / np.sqrt(mean["n_reps"])
    mean.attrs["oracle_cvpe"] = float(np.mean(bundles[0]["oracle_loss"]))
    if ces_ref is not None:
        mean.attrs["oracle_ces"] = float(np.mean(ces_ref["oracle"])); mean.attrs["independence_ces"] = float(np.mean(ces_ref["independence"]))
    return mean.sort_values("excess"), long


def hump_diagnostics(bundle, param_out, neural_results):
    """tau(w) curves of the AIC-selected binary pair, the unconditional fit, the Acar fits and the first-seed networks.

    The result gives the location of each curve's peak and its number of sign changes, for the hump designs where the shape of
    tau(w) is the question. The fine-grid curves (W_PLOT, first draw) feed the figure with edge close-ups.

    Parameters:
    bundle (dict): Data draw, see make_bundle.
    param_out (dict): Result of parametric_acar_job.
    neural_results (dict): Results of neural_synth_job.

    Returns:
    dict: curves (W_EVAL grid), zero_crossings, peak_location and fine (W_PLOT grid).
    """
    spec = bundle["spec"]
    curves = {"w": W_EVAL, "truth": tau_true(W_EVAL, spec)}
    sel = param_out["selected_pair"]
    nm = f"Divergence split: {sel[0]} (low) / {sel[1]} (high)"
    curves["binary (AIC-selected pair)"] = param_out["models"][nm]["tau_curve"]
    unc = param_out["selected_uncond"]
    curves["unconditional (AIC-selected)"] = param_out["models"][f"Unconditional: {unc}"]["tau_curve"]
    for k, v in param_out["models"].items():
        if k.startswith("Acar"):
            curves[k] = v["tau_curve"]
    for r in neural_results.values():
        if "tau_curve" in r and r["seed"] == 0:
            curves[r["model"]] = r["tau_curve"]
    zc = {k: int(np.sum(np.diff(np.sign(v)) != 0)) for k, v in curves.items() if k not in ("w",)}
    peak = {k: float(W_EVAL[int(np.argmax(v))]) for k, v in curves.items() if k not in ("w",)}
    # fine-grid curves (W_PLOT) of the same models for the figure with edge close-ups (first draw)
    fine = {"w": W_PLOT, "truth": tau_true(W_PLOT, spec)}
    for k in ("binary (AIC-selected pair)", "unconditional (AIC-selected)"):
        src = nm if k.startswith("binary") else f"Unconditional: {unc}"
        if param_out["models"][src].get("tau_plot") is not None:
            fine[k] = param_out["models"][src]["tau_plot"]
    for k, v in param_out["models"].items():
        if k.startswith("Acar") and v.get("tau_plot") is not None:
            fine[k] = v["tau_plot"]
    for r in neural_results.values():
        if "tau_plot" in r and r["seed"] == 0:
            fine[r["model"]] = r["tau_plot"]
    return dict(curves=curves, zero_crossings=zc, peak_location=peak, fine=fine)
