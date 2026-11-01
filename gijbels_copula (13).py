"""gijbels_copula.py

Kernel benchmark for the synthetic study and real-data folds: the transformed conditional-copula estimator of Gijbels, Veraverbeke, and Omelka
(2011), used on synthetic data and real-data folds.

Estimator (covariate x, pseudo-observations (U1, U2), training sample (X_i, U1i, U2i)):
  1. conditional margins     F_{j|x,g}(y) = sum_k a_k(x, g) 1(U_jk <= y), with kernel weights a_k (local-linear)
  2. conditional PITs        U~_ji = F_{j|X_i,g}(U_ji), computed on the training data only
  3. weighted empirical cdf  G~_{x,h}(v1, v2) = sum_i b_i(x, h) 1(U~_1i <= v1, U~_2i <= v2), with kernel weights b_i
  4. copula                  C~_{x,h}(u1, u2) = G~_{x,h}(G~_{1,x,h}^{-1}(u1), G~_{2,x,h}^{-1}(u2))

C~ is a weighted empirical copula of the points (G~_1(U~_1i), G~_2(U~_2i)). It is therefore a discrete copula (2-increasing and
grounded) up to the jump size of its margins, which ces_score.margin_error reports. The copula is represented by its K x K cell
masses.

Implementation:
  * The estimator follows the specification of the paper: local-linear weights, triweight kernel, and bandwidths
    from the AMISE plug-in rule of its Section 2.3 (iterated pilot bandwidths hV and hB = 2 * IQR * hV * n^(1/10)), with the
    asymptotic bias and variance of Veraverbeke, Omelka, and Gijbels (2011): bias (h^2 mu2 / 2) d2C_x/dx2 (local-linear weights,
    so no design term) and variance R(K) sigma^2_x(u) / (n h f_X(x)). The margin bandwidths g_1 and g_2 come from the univariate
    version of the same rule applied to the conditional cdf. The authors' bandwidth code is unpublished ("details on request"),
    so the rule is re-derived from the published asymptotics; g comes from the univariate adaptation of the same rule (Section 2.3),
    not from 'lokern', which the paper used for g in its data examples. The bandwidths are global, as the paper recommends
    for a whole curve: the AMSE is integrated over u and over the covariate. The integration runs over the 5% to 95% covariate
    quantiles; that trimming is a choice of this implementation.
  * Choices the paper does not specify: local-linear weights can be negative, so the conditional margin cdf values are
    clipped to [0, 1], the cumulative weights used for the quantile inversion are made monotone (running maximum), and negative
    cell masses are clipped to 0 and the rest renormalised (the weights need not be positive, Veraverbeke et al., 2011,
    p. 768). Plug-in bandwidths are capped at the covariate range, because the theoretical optimum is infinite when the true
    quantity does not vary with x; 'capped' flags it.

Gijbels, I., Veraverbeke, N., & Omelka, M. (2011). Conditional copulas, association measures and their applications.
Computational Statistics & Data Analysis, 55(5), 1919-1932.
Veraverbeke, N., Omelka, M., & Gijbels, I. (2011). Estimation of a conditional copula and association measures. Scandinavian
Journal of Statistics, 38(4), 766-780.
Nelsen, R. B. (2006). An introduction to copulas (2nd ed.). Springer.
Coles, S., Heffernan, J., & Tawn, J. (1999). Dependence measures for extreme value analyses. Extremes, 2(4), 339-365.
"""
import ast
import hashlib
import inspect
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_support import resolve_pool_workers

import ces_score as CES

LOSS_CELLS = 32                      # CVPE conditional means; aggregate existing 64 x 64 masses (same for estimator and oracle)


def triweight(t):
    """Triweight kernel K(t) = (35/32) (1 - t^2)^3 on |t| < 1, zero elsewhere.

    Parameters:
    t (array-like): Points at which to evaluate the kernel.

    Returns:
    np.ndarray: Kernel values, same shape as t.
    """
    t = np.asarray(t, float)
    return np.where(np.abs(t) < 1.0, (35.0 / 32.0) * (1.0 - t * t) ** 3, 0.0)


def ll_weights(x0, X, h):
    """Local-linear weights of the sample X at x0.

    The weights sum to one and can be negative.

    Parameters:
    x0 (float): Covariate value at which the weights are computed.
    X (array-like): Covariate values of the sample.
    h (float): Bandwidth.

    Returns:
    np.ndarray: The weights, or an all-NaN row if the local design is degenerate.
    """
    X = np.asarray(X, float); d = X - x0
    k = triweight(d / h)
    S0, S1, S2 = k.sum(), (k * d).sum(), (k * d * d).sum()
    den = S0 * S2 - S1 * S1
    if not (S0 > 0 and den > 1e-12 * max(S0 * S2, 1e-300)):
        return np.full(len(X), np.nan)
    return k * (S2 - d * S1) / den


def effective_n(weights):
    """Effective sample size 1 / sum(w^2) of a weight vector.

    Parameters:
    weights (array-like): Kernel weights.

    Returns:
    float: The effective sample size, or 0.0 if the weights are not all finite or do not sum to a positive value.
    """
    w = np.asarray(weights, float)
    return float(1.0 / np.sum(w ** 2)) if np.all(np.isfinite(w)) and w.sum() > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────
# Step 1-2: conditional PITs of the training sample
# ──────────────────────────────────────────────────────────────────────────
def _pit_weights_rows(X, rows, g):
    """Local-linear kernel weights of the sample X at the covariate values X[rows].

    Rows are processed in blocks by conditional_pits, so memory is block x n and not n x n (n = 10,000 would otherwise need
    several GB per matrix).

    Parameters:
    X (np.ndarray): Covariate values of the sample.
    rows (np.ndarray): Indices of the sample points at which the weights are computed.
    g (float): Bandwidth.

    Returns:
    np.ndarray: Weights of shape (len(rows), len(X)); a row is all NaN where the weights are undefined.
    """
    D = X[None, :] - X[rows][:, None]
    Kmat = triweight(D / g)
    S0 = Kmat.sum(1); S1 = (Kmat * D).sum(1); S2 = (Kmat * D * D).sum(1)
    den = S0 * S2 - S1 ** 2
    okrow = (S0 > 0) & (den > 1e-12 * np.maximum(S0 * S2, 1e-300))
    return Kmat * (S2[:, None] - D * S1[:, None]) / np.where(okrow, den, np.nan)[:, None]


PIT_BLOCK = 500


def conditional_pits(X, U, g, block=PIT_BLOCK):
    """Conditional PITs U~[:, j] = F_{j|X_i,g_j}(U_ji) of the sample with local-linear weights, clipped to [0, 1].

    Local-linear weights can leave the unit interval, hence the clipping. The computation runs in row blocks (memory block x n)
    and gives the same result as the full-matrix computation.

    Parameters:
    X (array-like): Covariate values, length n.
    U (array-like): Pseudo-observations, shape (n, 2).
    g (float or tuple): One bandwidth, or a pair (g1, g2) for the two margins.
    block (int): Number of rows per block. Default PIT_BLOCK.

    Returns:
    np.ndarray: The conditional PITs, shape (n, 2); NaN where a point has no usable neighbourhood.
    """
    X = np.asarray(X, float); U = np.asarray(U, float)
    gs = (float(g), float(g)) if np.isscalar(g) else (float(g[0]), float(g[1]))
    n = len(X)
    out = np.empty((n, 2))
    for a in range(0, n, block):
        rows = np.arange(a, min(a + block, n))
        W = None
        for j in range(2):
            if W is None or gs[j] != gs[0] or j == 0:
                W = _pit_weights_rows(X, rows, gs[j])
            ind = (U[None, :, j] <= U[rows, j][:, None])
            out[rows, j] = (W * ind).sum(1)
    return np.clip(out, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────
# Step 3-4: the conditional copula on the K x K grid
# ──────────────────────────────────────────────────────────────────────────
def _weighted_quantile_index(v_sorted, cumw, levels):
    """Smallest index i with cumw[i] >= level, the generalised inverse of the weighted empirical cdf.

    Parameters:
    v_sorted (np.ndarray): Sorted values; only its length is used.
    cumw (np.ndarray): Cumulative weights, nondecreasing, ending at 1.
    levels (array-like): Probability levels.

    Returns:
    np.ndarray: One index per level, at most len(v_sorted) - 1.
    """
    return np.minimum(np.searchsorted(cumw, np.asarray(levels) - 1e-12, side="left"), len(v_sorted) - 1)


def copula_cdf_on_grid(Ut, b, K):
    """Evaluate the copula C~(a_k, a_l) at the grid edges a_k = k / K (k = 0, ..., K).

    The margins of the weighted empirical cdf of the PITs are made uniform through their weighted quantile functions (step 4 of
    the estimator).

    Parameters:
    Ut (np.ndarray): Conditional PITs of the training sample, shape (n, 2).
    b (np.ndarray): Kernel weights of the training sample at the target covariate value, length n (may be negative).
    K (int): Number of cells per margin.

    Returns:
    np.ndarray: The copula cdf on the grid, shape (K + 1, K + 1), scaled so that its last entry is 1.
    """
    levels = np.arange(K + 1) / K
    q = []                                                           # per margin: grid-edge -> rank threshold in U~
    for j in range(2):
        order = np.argsort(Ut[:, j], kind="stable")
        cw = np.maximum.accumulate(np.cumsum(b[order]))          # monotone even with negative (local-linear) weights
        cw = cw / cw[-1]
        idx = _weighted_quantile_index(Ut[order, j], cw, levels)
        qv = Ut[order, j][idx]
        qv[0] = -np.inf                                              # u=0 -> empty set
        q.append(qv)
    # bin each point by the first grid edge whose quantile covers it, per margin
    bins = [np.searchsorted(q[j], Ut[:, j], side="left") for j in range(2)]   # in 0..K+1 ; K+1 = beyond last edge
    H = np.zeros((K + 2, K + 2))
    np.add.at(H, (bins[0], bins[1]), b)
    cdf = np.cumsum(np.cumsum(H, 0), 1)[: K + 1, : K + 1]
    return cdf / max(cdf[K, K], 1e-300)


def masses_from_cdf_grid(cdf):
    """Cell masses of a copula cdf given on a grid, by second differences.

    Parameters:
    cdf (np.ndarray): Copula cdf on the grid edges, shape (K + 1, K + 1).

    Returns:
    np.ndarray: Cell masses, shape (K, K), possibly with small negative entries.
    """
    m = cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]
    return m


class GijbelsFit:
    """Training-only fit of the transformed estimator, evaluated through the K x K cell masses of C~_{x0,h}.

    Training points without a usable neighbourhood in g (undefined PITs) are dropped, and the number of dropped points is
    kept in bad_pit.

    Parameters:
    X (array-like): Covariate values of the training sample.
    U (array-like): Pseudo-observations of the training sample, shape (n, 2).
    h (float): Bandwidth of the copula estimator.
    g (float or tuple): Bandwidth of the conditional margins, one value or a pair.
    K (int): Number of cells per margin. Default ces_score.DEFAULT_K.
    """

    def __init__(self, X, U, h, g, K=CES.DEFAULT_K):
        """Compute the conditional PITs of the training sample.

        Parameters:
        X (array-like): Covariate values of the training sample.
        U (array-like): Pseudo-observations of the training sample, shape (n, 2).
        h (float): Bandwidth of the copula estimator.
        g (float or tuple): Bandwidth of the conditional margins, one value or a pair.
        K (int): Number of cells per margin. Default ces_score.DEFAULT_K.
        """
        self.X = np.asarray(X, float).copy(); self.U = np.asarray(U, float).copy()
        self.h, self.g, self.K = float(h), g, int(K)
        self.Ut = conditional_pits(self.X, self.U, self.g)
        ok = np.all(np.isfinite(self.Ut), axis=1)
        self.bad_pit = int(np.sum(~ok))
        if self.bad_pit:                                             # training points with no usable neighbourhood in g
            self.X, self.U, self.Ut = self.X[ok], self.U[ok], self.Ut[ok]
        self.log = {}

    def cdf_grid(self, x0):
        """Copula cdf on the K-grid at covariate value x0.

        Parameters:
        x0 (float): Covariate value.

        Returns:
        tuple: (cdf, effective n) with cdf of shape (K + 1, K + 1), or (None, 0.0) if the weights at x0 are undefined.
        """
        b = ll_weights(x0, self.X, self.h)
        if not np.all(np.isfinite(b)):
            return None, 0.0
        return copula_cdf_on_grid(self.Ut, b, self.K), effective_n(b)

    def masses(self, x0):
        """Cell masses of the estimated copula at covariate value x0.

        Negative masses (possible with local-linear weights) are clipped to zero and the rest renormalised. The largest
        clipped mass seen so far is kept in self.log['clipped_mass'].

        Parameters:
        x0 (float): Covariate value.

        Returns:
        np.ndarray: The K * K masses, summing to one, in row-major order (u1 by u2).

        Raises:
        ValueError: If no training point lies within the kernel support of x0.
        """
        cdf, neff = self.cdf_grid(x0)
        if cdf is None:
            raise ValueError(f"no usable training neighbourhood within h={self.h:.3f} of x={x0:.3f}")
        raw = masses_from_cdf_grid(cdf)
        neg = float(-raw[raw < 0].sum())
        self.log["clipped_mass"] = max(self.log.get("clipped_mass", 0.0), neg)
        m = np.clip(raw, 0.0, None)
        return (m / m.sum()).ravel()

    def kendall_tau(self, x0):
        """Kendall's tau of the estimated copula at covariate value x0.

        Parameters:
        x0 (float): Covariate value.

        Returns:
        float: Kendall's tau.
        """
        return tau_from_masses(self.masses(x0), self.K)


def tau_from_masses(masses, K):
    """Kendall's tau of a discrete copula given by its K x K cell masses.

    Uses tau = 4 E[C(U, V)] - 1 (Nelsen, 2006), with C evaluated as the average of its four corner values in each cell.

    Parameters:
    masses (array-like): Cell masses, K * K values summing to one.
    K (int): Number of cells per margin.

    Returns:
    float: Kendall's tau.
    """
    m = np.asarray(masses, float).reshape(K, K)
    cdf = np.zeros((K + 1, K + 1)); cdf[1:, 1:] = np.cumsum(np.cumsum(m, 0), 1)
    Cc = 0.25 * (cdf[:-1, :-1] + cdf[1:, :-1] + cdf[:-1, 1:] + cdf[1:, 1:])
    return float(4.0 * np.sum(m * Cc) - 1.0)


CORNERS = ("UU", "LL", "UL", "LU")


def corner_chi_from_masses(masses, K, q=0.95):
    """Finite-threshold corner coefficients chi_corner(q) = P(corner) / (1 - q) of a discrete copula.

    These are the same four quantities as synthetic_study.corner_chi_from_cdf (upper-upper, lower-lower and the two mixed
    corners; the upper-upper one is the conditional probability whose limit defines chi in Coles, Heffernan, & Tawn, 1999,
    Section 3.3.1). The cdf on the K-grid is interpolated bilinearly at the corner thresholds, because
    1 - q and q are not grid edges in general.

    Parameters:
    masses (array-like): Cell masses, K * K values summing to one.
    K (int): Number of cells per margin.
    q (float): Threshold of the upper tail; 1 - q is used for the lower tail. Default 0.95.

    Returns:
    dict: Corner name ('UU', 'LL', 'UL', 'LU') -> coefficient.
    """
    m = np.asarray(masses, float).reshape(K, K)
    cdf = np.zeros((K + 1, K + 1)); cdf[1:, 1:] = np.cumsum(np.cumsum(m, 0), 1)

    def C(x, y):
        """Bilinear interpolation of the copula cdf at (x, y).

        Parameters:
        x (float): First argument, in [0, 1].
        y (float): Second argument, in [0, 1].

        Returns:
        float: The interpolated cdf value.
        """
        fx, fy = min(max(x, 0.0), 1.0) * K, min(max(y, 0.0), 1.0) * K
        i, j = min(int(fx), K - 1), min(int(fy), K - 1)
        tx, ty = fx - i, fy - j
        return float((1 - tx) * (1 - ty) * cdf[i, j] + tx * (1 - ty) * cdf[i + 1, j]
                     + (1 - tx) * ty * cdf[i, j + 1] + tx * ty * cdf[i + 1, j + 1])
    a = 1.0 - q
    return {"UU": (1 - 2 * q + C(q, q)) / a, "LL": C(a, a) / a, "UL": (a - C(q, a)) / a, "LU": (a - C(a, q)) / a}


def joint_exceed_from_masses(masses, K, q):
    """Joint-exceedance probability P(U1 > q, U2 > q) = 1 - 2q + C(q, q) of a discrete copula given by its cell masses.

    The cdf on the K-grid is interpolated bilinearly at (q, q), because q is not a grid edge in general.

    Parameters:
    masses (array-like): Cell masses, K * K values summing to one.
    K (int): Number of cells per margin.
    q (float): Threshold in [0, 1).

    Returns:
    float: The probability, clipped to [0, 1].
    """
    m = np.asarray(masses, float).reshape(K, K)
    cdf = np.zeros((K + 1, K + 1)); cdf[1:, 1:] = np.cumsum(np.cumsum(m, 0), 1)
    f = min(max(q, 0.0), 1.0) * K
    i = min(int(f), K - 1); t = f - i
    c = ((1 - t) ** 2 * cdf[i, i] + t * (1 - t) * (cdf[i + 1, i] + cdf[i, i + 1]) + t ** 2 * cdf[i + 1, i + 1])
    return float(np.clip(1.0 - 2.0 * q + c, 0.0, 1.0))


def sample_from_masses(masses, K, n, seed=0):
    """Draw n points from a discrete copula: a cell with probability equal to its mass, then uniform within the cell.

    Parameters:
    masses (array-like): Cell masses, K * K values summing to one.
    K (int): Number of cells per margin.
    n (int): Number of draws.
    seed (int): Seed of the generator. Default 0.

    Returns:
    np.ndarray: Array of shape (n, 2).
    """
    rng = np.random.default_rng(int(seed))
    p = np.clip(np.asarray(masses, float).ravel(), 0.0, None)
    cell = rng.choice(K * K, size=int(n), p=p / p.sum())
    return np.column_stack([(cell // K + rng.random(int(n))) / K, (cell % K + rng.random(int(n))) / K])


# ──────────────────────────────────────────────────────────────────────────
# AMISE plug-in bandwidths (paper Section 2.3; asymptotics of Veraverbeke, Omelka and Gijbels 2011)
# ──────────────────────────────────────────────────────────────────────────
def _kernel_moments(n_grid=200001):
    """Roughness R(K) and second moment mu_2(K) of the triweight kernel, by numerical integration.

    Parameters:
    n_grid (int): Number of integration points on [-1, 1]. Default 200001.

    Returns:
    tuple: (R(K), mu_2(K)).
    """
    t = np.linspace(-1, 1, n_grid); k = triweight(t); dt = t[1] - t[0]
    return float(np.sum(k * k) * dt), float(np.sum(t * t * k) * dt)          # R(K), mu_2(K)


R_K, MU2_K = _kernel_moments()


def _local_poly(X, Y, x_grid, bw, order=2):
    """Local polynomial fits of every column of Y on X at each point of x_grid.

    Parameters:
    X (array-like): Covariate values.
    Y (array-like): Responses, shape (n, m).
    x_grid (array-like): Points at which the fits are evaluated.
    bw (float): Bandwidth of the triweight kernel.
    order (int): Degree of the local polynomial. Default 2.

    Returns:
    tuple: (intercept, second derivative), both of shape (len(x_grid), m); NaN where the local fit is degenerate.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    f0 = np.full((len(x_grid), Y.shape[1]), np.nan); f2 = f0.copy()
    for a, x0 in enumerate(x_grid):
        d = X - x0; w = triweight(d / bw)
        sel = w > 0
        if sel.sum() < order + 3:
            continue
        A = np.vander(d[sel], order + 1, increasing=True)
        AtW = A.T * w[sel]
        try:
            beta = np.linalg.solve(AtW @ A, AtW @ Y[sel])
        except np.linalg.LinAlgError:
            continue
        f0[a] = beta[0]; f2[a] = 2.0 * beta[2]
    return f0, f2


def _fx(X, x_grid):
    """Gaussian kernel density estimate of the covariate at the points of x_grid.

    The bandwidth is the normal-reference rule of thumb 1.06 * min(sd, IQR / 1.349) * n^(-1/5), with a robust scale estimate.
    The density only serves as the f_X(x) weight in the plug-in rule.

    Parameters:
    X (array-like): Covariate values.
    x_grid (np.ndarray): Points at which the density is evaluated.

    Returns:
    np.ndarray: Density estimates, same length as x_grid.
    """
    X = np.asarray(X, float); n = len(X)
    bw = 1.06 * min(np.std(X, ddof=1), (np.subtract(*np.percentile(X, [75, 25]))) / 1.349) * n ** (-0.2)
    return np.exp(-0.5 * ((x_grid[:, None] - X[None, :]) / bw) ** 2).sum(1) / (n * bw * np.sqrt(2 * np.pi))


def _x_grid(X, m=30):
    """Equally spaced covariate grid over the 5% to 95% quantiles of X.

    Parameters:
    X (array-like): Covariate values.
    m (int): Number of grid points. Default 30.

    Returns:
    tuple: (grid, length of the interval).
    """
    lo, hi = np.percentile(X, [5, 95])
    return np.linspace(lo, hi, m), (hi - lo)


def _iterate_plugin(X, bias_var_fn, h0, max_iter=10, tol=0.01):
    """Iterate the plug-in rule of the paper until the bandwidth stabilises.

    Each step sets hB = 2 * IQR * hV * n^(1/10), estimates the bias integral with hB and the variance integral with hV, and
    sets h = (R(K) V / (n mu_2^2 B))^(1/5), the minimiser of the AMISE; then hV is replaced by h. The bandwidth is capped at the
    covariate range.

    Parameters:
    X (np.ndarray): Covariate values.
    bias_var_fn (callable): (hV, hB) -> (variance integral, bias integral).
    h0 (float): Starting bandwidth.
    max_iter (int): Maximum number of iterations. Default 10.
    tol (float): Relative change of the bandwidth at which the iteration stops. Default 0.01.

    Returns:
    tuple: (bandwidth, dict with 'iterations', 'capped' and 'hB').
    """
    n = len(X); iqr = float(np.subtract(*np.percentile(X, [75, 25]))); rng = float(X.max() - X.min())
    hV = h0
    for it in range(max_iter):
        hB = min(2.0 * iqr * hV * n ** 0.1, rng)
        Vbar, Bbar = bias_var_fn(hV, hB)
        h_new = (R_K * Vbar / (n * MU2_K ** 2 * Bbar)) ** 0.2 if Bbar > 0 and np.isfinite(Bbar) and np.isfinite(Vbar) else np.inf
        capped = h_new > rng
        h_new = min(h_new, rng)
        if abs(h_new - hV) / hV < tol:
            hV = h_new; break
        hV = h_new
    return float(hV), dict(iterations=it + 1, capped=bool(capped), hB=float(hB))


def plugin_g(X, Uj, ny=20):
    """Plug-in bandwidth g_j of the conditional cdf of one margin.

    Univariate version of the plug-in rule with local-linear weights:
    AMISE(g) = R(K) / (n g) * Int Int F (1 - F) / f_X dy dx + g^4 mu_2^2 / 4 * Int Int (d2F/dx2)^2 dy dx.

    Parameters:
    X (array-like): Covariate values.
    Uj (array-like): Pseudo-observation of the margin.
    ny (int): Number of evaluation points y, at the 5% to 95% quantiles of Uj. Default 20.

    Returns:
    tuple: (bandwidth, dict with 'iterations', 'capped' and 'hB').
    """
    X = np.asarray(X, float); Uj = np.asarray(Uj, float); n = len(X)
    xg, xlen = _x_grid(X); fx = _fx(X, xg)
    ys = np.quantile(Uj, np.linspace(0.05, 0.95, ny)); dy = np.gradient(ys)
    Y = (Uj[:, None] <= ys[None, :]).astype(float)
    Fglob = Y.mean(0)
    Vbar = float(np.mean(1.0 / np.maximum(fx, 1e-12)) * xlen * np.sum(Fglob * (1 - Fglob) * dy))

    def fn(_hV, hB):
        """Variance and bias integrals at the bias pilot bandwidth hB (the variance integral of a margin needs no pilot bandwidth).

        Parameters:
        _hV (float): Pilot bandwidth of the variance integral; not used.
        hB (float): Pilot bandwidth of the bias integral.

        Returns:
        tuple: (variance integral, bias integral); the bias integral is NaN if no covariate value has a usable fit.
        """
        _, d2 = _local_poly(X, Y, xg, hB)
        ok = np.all(np.isfinite(d2), axis=1)
        if not ok.any():
            return Vbar, np.nan
        Bbar = float(np.mean(np.sum(d2[ok] ** 2 * dy[None, :], axis=1)) * xlen)
        return Vbar, Bbar
    h0 = 0.5 * float(np.subtract(*np.percentile(X, [75, 25]))) * n ** (-0.2)
    return _iterate_plugin(X, fn, h0)


def plugin_h(X, Ut, ncell=12):
    """Plug-in bandwidth h of the copula estimator.

    AMISE(h) = R(K) / (n h) * Int Int sigma^2_x(u) / f_X(x) du dx + h^4 mu_2^2 / 4 * Int Int (d2C_x/dx2)^2 du dx, where sigma^2
    is the variance function of the limit process (Veraverbeke et al., 2011, Corollary 2):
    C (1 - C) + C1^2 u1 (1 - u1) + C2^2 u2 (1 - u2) - 2 C1 C (1 - u1) - 2 C2 C (1 - u2) + 2 C1 C2 (C - u1 u2).
    The integrals are evaluated on an ncell x ncell grid of levels.

    Parameters:
    X (array-like): Covariate values.
    Ut (array-like): Conditional PITs, shape (n, 2).
    ncell (int): Number of grid levels per margin. Default 12.

    Returns:
    tuple: (bandwidth, dict with 'iterations', 'capped' and 'hB').
    """
    X = np.asarray(X, float); Ut = np.asarray(Ut, float); n = len(X)
    xg, xlen = _x_grid(X); fx = _fx(X, xg)
    lev = np.arange(1, ncell + 1) / (ncell + 1.0)
    Y = ((Ut[:, 0][:, None] <= lev[None, :])[:, :, None] & (Ut[:, 1][:, None] <= lev[None, :])[:, None, :]).reshape(n, -1).astype(float)

    def fn(hV, hB):
        """Variance and bias integrals for the pilot bandwidths hV and hB.

        Parameters:
        hV (float): Pilot bandwidth of the variance integral.
        hB (float): Pilot bandwidth of the bias integral.

        Returns:
        tuple: (variance integral, bias integral); both NaN if no covariate value has a usable fit.
        """
        C0, _ = _local_poly(X, Y, xg, max(hV, 1e-6), order=2)
        _, d2 = _local_poly(X, Y, xg, hB)
        ok = np.all(np.isfinite(C0), axis=1) & np.all(np.isfinite(d2), axis=1)
        if not ok.any():
            return np.nan, np.nan
        C = np.clip(C0[ok].reshape(-1, ncell, ncell), 1e-6, 1 - 1e-6)
        u1 = lev[None, :, None]; u2 = lev[None, None, :]
        C1 = np.gradient(C, lev, axis=1); C2 = np.gradient(C, lev, axis=2)
        sig2 = (C * (1 - C) + C1 ** 2 * u1 * (1 - u1) + C2 ** 2 * u2 * (1 - u2) - 2 * C1 * C * (1 - u1)
                - 2 * C2 * C * (1 - u2) + 2 * C1 * C2 * (C - u1 * u2))
        sig2 = np.clip(sig2, 0.0, None)
        Vbar = float(np.mean(sig2.mean((1, 2)) / np.maximum(fx[ok], 1e-12)) * xlen)
        Bbar = float(np.mean((d2[ok] ** 2).mean(1)) * xlen)
        return Vbar, Bbar
    h0 = 0.5 * float(np.subtract(*np.percentile(X, [75, 25]))) * n ** (-0.2)
    return _iterate_plugin(X, fn, h0)


# ──────────────────────────────────────────────────────────────────────────
# Pointwise evaluation on a covariate grid (CVPE and CES under one discretisation)
# ──────────────────────────────────────────────────────────────────────────
def cvpe_predictions_from_masses(m, o1, o2, K):
    """Conditional means E[U1 | U2 = o2] and E[U2 | U1 = o1] from cell masses.

    The means use cell centres and linear interpolation between neighbouring centres, constant outside the first and last centres.
    With K=64, the masses are merged to the 32x32 CVPE grid to limit noise in sparse conditional bins.

    Parameters:
    m (array-like): Cell masses, K * K values summing to one.
    o1 (array-like): First coordinate of the observations.
    o2 (array-like): Second coordinate of the observations.
    K (int): Number of cells per margin.

    Returns:
    tuple: Predicted first and second coordinates, each one value per observation.
    """
    M = np.asarray(m, float).reshape(K, K)
    if K > LOSS_CELLS:
        if K % LOSS_CELLS:
            raise ValueError("K must be a multiple of LOSS_CELLS to merge cell masses")
        f = K // LOSS_CELLS
        M = M.reshape(LOSS_CELLS, f, LOSS_CELLS, f).sum((1, 3))
        K = LOSS_CELLS
    c = (np.arange(K) + 0.5) / K
    colm = M.sum(0); rowm = M.sum(1)
    E1_given_u2 = (c[:, None] * M).sum(0) / np.maximum(colm, 1e-300)
    E2_given_u1 = (c[None, :] * M).sum(1) / np.maximum(rowm, 1e-300)
    e1 = np.interp(np.asarray(o2, float), c, E1_given_u2)
    e2 = np.interp(np.asarray(o1, float), c, E2_given_u1)
    return e1, e2


def cvpe_from_masses(m, o1, o2, K):
    """Squared-error CVPE loss from conditional means calculated from K x K cell masses.

    Parameters:
    m (array-like): Cell masses, K * K values summing to one.
    o1 (array-like): First coordinate of the observations.
    o2 (array-like): Second coordinate of the observations.
    K (int): Number of cells per margin.

    Returns:
    np.ndarray: The loss of each observation.
    """
    e1, e2 = cvpe_predictions_from_masses(m, o1, o2, K)
    return (np.asarray(o1) - e1) ** 2 + (np.asarray(o2) - e2) ** 2


def pointwise_on_grid(mass_fn, w_obs, o1, o2, w_grid, K, calibrated=False):
    """Per-observation CVPE loss and energy score from masses on a covariate grid.

    Conditional predictions are interpolated between the two bracketing covariate grid points before the squared CVPE loss is
    calculated. The energy score retains the linear score interpolation of ces_score.ces_conditional.

    Parameters:
    mass_fn (callable): Covariate value -> K * K cell masses.
    w_obs (array-like): Covariate value of each observation.
    o1 (array-like): First coordinate of the observations.
    o2 (array-like): Second coordinate of the observations.
    w_grid (np.ndarray): Increasing covariate grid.
    K (int): Number of cells per margin.
    calibrated (bool): If True, the masses are first adjusted to uniform margins with ces_score.sinkhorn_uniform_margins.
        Default False.

    Returns:
    dict: 'loss' (CVPE loss, see cvpe_from_masses) and 'es' (energy score), each an array with one value per observation.
    """
    w_obs = np.asarray(w_obs, float)
    o1, o2 = np.asarray(o1, float), np.asarray(o2, float)
    obs = np.column_stack([o1, o2])
    pos = np.clip(np.searchsorted(w_grid, w_obs, side="right") - 1, 0, len(w_grid) - 2)
    frac = np.clip((w_obs - w_grid[pos]) / (w_grid[pos + 1] - w_grid[pos]), 0.0, 1.0)
    lo = {k: np.full(len(w_obs), np.nan) for k in ("pred1", "pred2", "es")}
    hi = {k: np.full(len(w_obs), np.nan) for k in ("pred1", "pred2", "es")}
    for g in range(len(w_grid)):
        a = np.flatnonzero(pos == g); b = np.flatnonzero(pos + 1 == g)
        if not (len(a) or len(b)):
            continue
        m = np.asarray(mass_fn(float(w_grid[g])), float)
        if calibrated:
            m = CES.sinkhorn_uniform_margins(m, K)
        for idx, store in ((a, lo), (b, hi)):
            if len(idx):
                store["pred1"][idx], store["pred2"][idx] = cvpe_predictions_from_masses(m, o1[idx], o2[idx], K)
                store["es"][idx] = CES.es_from_masses(m, obs[idx], K)
    e1 = (1 - frac) * lo["pred1"] + frac * hi["pred1"]
    e2 = (1 - frac) * lo["pred2"] + frac * hi["pred2"]
    return dict(loss=(o1 - e1) ** 2 + (o2 - e2) ** 2, es=(1 - frac) * lo["es"] + frac * hi["es"])


def pointwise_on_grid_both(mass_fn, w_obs, o1, o2, w_grid, K):
    """pointwise_on_grid for the raw and the margin-calibrated masses in one pass.

    Gives the same two results as pointwise_on_grid(..., calibrated=False) and pointwise_on_grid(..., calibrated=True),
    but mass_fn (a kernel-weighted estimate over the whole training sample) runs once per grid point instead of twice,
    and the raw and calibrated energy scores share one distance matrix.

    Parameters:
    mass_fn (callable): Covariate value -> K * K cell masses.
    w_obs (array-like): Covariate value of each observation.
    o1 (array-like): First coordinate of the observations.
    o2 (array-like): Second coordinate of the observations.
    w_grid (np.ndarray): Increasing covariate grid.
    K (int): Number of cells per margin.

    Returns:
    tuple: (raw, calibrated), each a dict as returned by pointwise_on_grid.
    """
    w_obs = np.asarray(w_obs, float)
    o1, o2 = np.asarray(o1, float), np.asarray(o2, float)
    obs = np.column_stack([o1, o2])
    pos = np.clip(np.searchsorted(w_grid, w_obs, side="right") - 1, 0, len(w_grid) - 2)
    frac = np.clip((w_obs - w_grid[pos]) / (w_grid[pos + 1] - w_grid[pos]), 0.0, 1.0)
    keys = ("pred1", "pred2", "es")
    lo = {t: {k: np.full(len(w_obs), np.nan) for k in keys} for t in ("raw", "cal")}
    hi = {t: {k: np.full(len(w_obs), np.nan) for k in keys} for t in ("raw", "cal")}
    for g in range(len(w_grid)):
        a = np.flatnonzero(pos == g); b = np.flatnonzero(pos + 1 == g)
        if not (len(a) or len(b)):
            continue
        m_raw = np.asarray(mass_fn(float(w_grid[g])), float)
        ms = {"raw": m_raw, "cal": CES.sinkhorn_uniform_margins(m_raw, K)}
        for idx, side in ((a, lo), (b, hi)):
            if len(idx):
                es_raw, es_cal = CES.es_from_masses_many([ms["raw"], ms["cal"]], obs[idx], K)
                for t, es in (("raw", es_raw), ("cal", es_cal)):
                    side[t]["pred1"][idx], side[t]["pred2"][idx] = cvpe_predictions_from_masses(ms[t], o1[idx], o2[idx], K)
                    side[t]["es"][idx] = es
    res = []
    for t in ("raw", "cal"):
        e1 = (1 - frac) * lo[t]["pred1"] + frac * hi[t]["pred1"]
        e2 = (1 - frac) * lo[t]["pred2"] + frac * hi[t]["pred2"]
        res.append(dict(loss=(o1 - e1) ** 2 + (o2 - e2) ** 2, es=(1 - frac) * lo[t]["es"] + frac * hi[t]["es"]))
    return res[0], res[1]


# ──────────────────────────────────────────────────────────────────────────
# Synthetic-study job
# ──────────────────────────────────────────────────────────────────────────
LABEL = "Gijbels et al. (2011) transformed, LL weights, AMISE plug-in h"


def gijbels_synth_job(bundle, K=CES.DEFAULT_K, G=CES.DEFAULT_GRID_POINTS, w_eval=None, h=None, g=None):
    """Fit the transformed estimator on one synthetic training bundle and score it on the bundle's out-of-sample draw.

    Nothing from the out-of-sample draw enters the fit or the bandwidth choice. The estimator is the specification of the paper
    (see the module docstring).

    Parameters:
    bundle (dict): Synthetic bundle with 'w', 'u1', 'u2' (training), 'ow', 'o1', 'o2' (out of sample), 'spec', and optionally
        'rep' (data draw index; the fine-grid tau curve is computed for rep 0).
    K (int): Number of cells per margin. Default ces_score.DEFAULT_K.
    G (int): Number of covariate grid points for the pointwise scores. Default ces_score.DEFAULT_GRID_POINTS.
    w_eval (np.ndarray or None): Covariate grid of the tau and tail curves. Default synthetic_study.W_EVAL.
    h (float or None): Fixed copula bandwidth; None selects it. Default None.
    g (float or tuple or None): Fixed margin bandwidth(s); None selects them. Default None.

    Returns:
    dict: Scores ('ces', 'ces_cal', 'cvpe_disc', per-observation 'loss_oos', 'n_oos'), curves ('tau_curve', 'chi', 'tau_plot'),
        the bandwidths ('h', 'g', 'info'), its label, and diagnostics ('clipped_mass', 'margin_error_max',
        'neff_min', 'bad_pit', 'oos_outside_support', 'runtime').
    """
    import synthetic_study as SS        # imported here: synthetic_study pulls in fold_pipeline (and with it torch)
    t0 = time.time()
    w_eval = SS.W_EVAL if w_eval is None else w_eval
    X = np.asarray(bundle["w"], float); U = np.column_stack([bundle["u1"], bundle["u2"]]).astype(float)
    info = {}
    if g is None:
        (g1, i1), (g2, i2) = plugin_g(X, U[:, 0]), plugin_g(X, U[:, 1])
        g = (g1, g2); info["g_info"] = [i1, i2]
    if h is None:
        Ut = conditional_pits(X, U, g)
        ok = np.all(np.isfinite(Ut), axis=1)
        h, ih = plugin_h(X[ok], Ut[ok]); info["h_info"] = ih
    fit = GijbelsFit(X, U, h, g, K)
    ow = np.asarray(bundle["ow"], float)
    o1, o2 = np.asarray(bundle["o1"], float), np.asarray(bundle["o2"], float)
    gw = np.linspace(float(ow.min()), float(ow.max()), G)
    # the estimator has no data outside the training support: the number of such points is reported
    outside = int(np.sum((ow < X.min()) | (ow > X.max())))
    raw, cal = pointwise_on_grid_both(fit.masses, ow, o1, o2, gw, K)
    ms_eval = [fit.masses(float(np.clip(x, X.min(), X.max()))) for x in w_eval]
    tau = np.array([tau_from_masses(m, K) for m in ms_eval])
    chi = {c: np.array([corner_chi_from_masses(m, K)[c] for m in ms_eval]) for c in CORNERS}   # tail curves on w_eval
    tau_plot = None                                                       # fine grid for the tau figure, first data draw only
    if int(bundle.get("rep", 0)) == 0:
        tau_plot = np.array([fit.kendall_tau(float(np.clip(x, X.min(), X.max()))) for x in SS.W_PLOT])
    mg = [CES.margin_error(fit.masses(float(x)), K) for x in np.linspace(X.min(), X.max(), 9)]
    neff = [fit.cdf_grid(float(x))[1] for x in np.linspace(X.min(), X.max(), 9)]
    return dict(ces=float(np.sum(raw["es"])), ces_cal=float(np.sum(cal["es"])), n_oos=len(o1),
                cvpe_disc=float(np.mean(raw["loss"])), loss_oos=raw["loss"], tau_curve=tau, chi=chi, tau_plot=tau_plot,
                h=float(h), g=[float(x) for x in np.atleast_1d(g)], label=LABEL, info=info,
                clipped_mass=float(fit.log.get("clipped_mass", 0.0)),
                margin_error_max=float(np.max(mg)), neff_min=float(np.min(neff)), bad_pit=fit.bad_pit,
                oos_outside_support=outside, runtime=time.time() - t0)


def oracle_discretised(bundle, K=CES.DEFAULT_K, G=CES.DEFAULT_GRID_POINTS):
    """Evaluate the true conditional copula through the same pointwise and discretisation scheme as the estimator.

    The excess of the estimator over this oracle is then free of the K-cell discretisation bias of cvpe_from_masses.

    Parameters:
    bundle (dict): Synthetic bundle with 'spec', 'ow', 'o1' and 'o2'.
    K (int): Number of cells per margin. Default ces_score.DEFAULT_K.
    G (int): Number of covariate grid points. Default ces_score.DEFAULT_GRID_POINTS.

    Returns:
    dict: 'loss' and 'es', per-observation arrays (see pointwise_on_grid).
    """
    import synthetic_study as SS        # imported here: synthetic_study pulls in fold_pipeline (and with it torch)
    spec = bundle["spec"]
    ow = np.asarray(bundle["ow"], float)
    gw = np.linspace(float(ow.min()), float(ow.max()), G)
    mf = lambda wv: CES.masses_from_cdf(SS.true_cop(float(SS.tau_true(wv, spec)), spec), K)
    return pointwise_on_grid(mf, ow, np.asarray(bundle["o1"], float), np.asarray(bundle["o2"], float), gw, K)


def gijbels_row(bundle, res, orc=None):
    """Build one summary row comparable to the columns of synthetic_study.summarize.

    The excess columns are in the discretisation of this module.

    Parameters:
    bundle (dict): Synthetic bundle (uses 'spec' and optionally 'true_corners').
    res (dict): Result of gijbels_synth_job.
    orc (dict or None): Result of oracle_discretised; the excess columns are added if given. Default None.

    Returns:
    dict: Model label, tau RMSE, per-observation CES (raw and calibrated), bandwidths, diagnostics, the tail RMSEs if
        the bundle has true corner curves, and the excess CVPE and CES if orc is given.
    """
    import synthetic_study as SS        # imported here: synthetic_study pulls in fold_pipeline (and with it torch)
    spec = bundle["spec"]
    row = dict(model=res["label"], group="Nonparametric", n_seeds=1,
               tau_rmse=float(np.sqrt(np.mean((res["tau_curve"] - SS.tau_true(SS.W_EVAL, spec)) ** 2))),
               ces=res["ces"] / res["n_oos"], ces_calibrated=res["ces_cal"] / res["n_oos"],
               h=res["h"], g=float(np.mean(res["g"])), clipped_mass=res["clipped_mass"], margin_error_max=res["margin_error_max"], neff_min=res["neff_min"])
    tc = bundle.get("true_corners")
    if tc is not None and res.get("chi") is not None:                  # tail RMSE exactly as in synthetic_study.summarize
        row["tail_uu"] = SS._curve_rmse(res["chi"]["UU"], tc["UU"])
        row["tail_ll"] = SS._curve_rmse(res["chi"]["LL"], tc["LL"])
        row["tail_mixed"] = 0.5 * (SS._curve_rmse(res["chi"]["UL"], tc["UL"]) + SS._curve_rmse(res["chi"]["LU"], tc["LU"]))
    if orc is not None:
        row["excess"] = float(np.mean(res["loss_oos"]) - np.mean(orc["loss"]))
        row["excess_ces"] = row["ces"] - float(np.mean(orc["es"]))
    return row


# ──────────────────────────────────────────────────────────────────────────
# Resume-safe runner for the notebooks (one pickle per data draw; code + data fingerprint)
# ──────────────────────────────────────────────────────────────────────────
def _code_digest(module):
    """AST dump of a module's source with the docstrings removed.

    Comments, layout and docstrings do not change it, so editing them does not invalidate cached results.

    Parameters:
    module (module): The module to digest.

    Returns:
    str: The AST dump.
    """
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def _fingerprint(bundle, K, G):
    """Hash of the code, the bundle's data arrays and the run settings.

    The code is this module, ces_score (the scores) and synthetic_study (the true copula that defines the oracle). A cached
    result is reused only if the fingerprint matches, so changing any of them or the data recomputes it, whereas editing
    comments or docstrings does not.

    Parameters:
    bundle (dict): Synthetic bundle with 'u1', 'u2', 'w', 'o1', 'o2' and 'ow'.
    K (int): Number of cells per margin.
    G (int): Number of covariate grid points.

    Returns:
    str: Hexadecimal SHA-256 digest.
    """
    import synthetic_study as SS        # imported here: synthetic_study pulls in fold_pipeline (and with it torch)
    h = hashlib.sha256()
    for module in (sys.modules[__name__], CES, SS):
        h.update(_code_digest(module).encode())
    for k in ("u1", "u2", "w", "o1", "o2", "ow"):
        h.update(np.ascontiguousarray(bundle[k]).tobytes())
    h.update(f"{K}|{G}".encode())
    return h.hexdigest()


def _one(args):
    """Worker of run_gijbels: fit and score one bundle in a single-threaded process.

    Parameters:
    args (tuple): (bundle, K, G).

    Returns:
    dict: {'status': 'ok', 'row': summary row, 'res': job result without the per-observation loss}, or
        {'status': 'failed', 'error': repr of the exception}.
    """
    bundle, K, G = args
    try:
        with threadpool_limits(limits=1):       # one BLAS thread per worker, so max_workers processes do not oversubscribe
            res = gijbels_synth_job(bundle, K=K, G=G)
            orc = oracle_discretised(bundle, K=K, G=G)
        return dict(status="ok", row=gijbels_row(bundle, res, orc), res={k: v for k, v in res.items() if k != "loss_oos"})
    except Exception as exc:                                  # recorded, never silently dropped
        return dict(status="failed", error=repr(exc))


def run_gijbels(bundles, cache_dir, max_workers=None, K=CES.DEFAULT_K, G=CES.DEFAULT_GRID_POINTS):
    """Run the estimator for each bundle, or load the cached result.

    One pickle per bundle is written to cache_dir and reused when its fingerprint (code, data, settings) matches.

    Parameters:
    bundles (list): Synthetic bundles of one design; each needs the keys of gijbels_synth_job plus 'dgp' and 'rep'.
    cache_dir (str or pathlib.Path): Folder of the cached results; created if missing.
    max_workers (int): Optional upper limit; default uses CPU, task count and RAM budget.
    K (int): Number of cells per margin. Default ces_score.DEFAULT_K.
    G (int): Number of covariate grid points. Default ces_score.DEFAULT_GRID_POINTS.

    Returns:
    tuple: (pandas.DataFrame with one row per successful bundle, list of (key, error) for the failed ones). The
        frame's attrs['tau_plot'] is the fine-grid tau(w) curve of the first data draw (None if there is none).

    Raises:
    ValueError: If the bundles belong to more than one design (more than one 'dgp').
    """
    if len({b["dgp"] for b in bundles}) > 1:
        raise ValueError("run_gijbels takes the bundles of one design per call: the results are keyed by data draw "
                         "and the returned frame and tau curves are per design.")
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for b in bundles:
        f = cache / f"gijbels_{b['dgp']}_rep{b['rep']}.pkl"; fp = _fingerprint(b, K, G)
        if f.exists():
            try:
                d = pickle.load(open(f, "rb"))
                if d.get("fp") == fp:
                    out[b["rep"]] = d["r"]; continue
            except Exception:
                pass
        todo.append((b, f, fp))
    if todo:
        # Many cached data draws are independent, but each fit carries dense arrays.
        worker_gib = float(os.getenv('THESIS_NONPARAM_WORKER_GIB', '2.5'))
        workers = resolve_pool_workers(n_tasks=len(todo), max_workers=max_workers, worker_gib=worker_gib)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (b, f, fp), r in zip(todo, ex.map(_one, [(b, K, G) for b, _, _ in todo])):
                out[b["rep"]] = r
                if r["status"] == "ok":
                    tmp = f.with_suffix(".tmp"); pickle.dump(dict(fp=fp, r=r), open(tmp, "wb")); tmp.replace(f)
    rows = [dict(rep=k, **v["row"]) for k, v in sorted(out.items()) if v["status"] == "ok"]
    fails = [(k, v["error"]) for k, v in sorted(out.items()) if v["status"] != "ok"]
    df = pd.DataFrame(rows)
    # tau(w) curve on the fine grid (first data draw), for the tau figure
    df.attrs["tau_plot"] = next((v["res"]["tau_plot"] for _, v in sorted(out.items())
                                 if v["status"] == "ok" and v["res"].get("tau_plot") is not None), None)
    return df, fails
