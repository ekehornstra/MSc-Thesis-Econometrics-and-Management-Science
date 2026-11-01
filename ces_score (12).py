"""ces_score.py

Copula Energy Score (CES; Ziel & Berk, 2019): the energy score (Gneiting & Raftery, 2007) of a model's conditional
copula at the observed pseudo-observation pair, and the real-data comparison built on it.

The score of observation u_t under model m is ES = E||U - u_t|| - 0.5 * E||U - U'||, with U and U' iid from the
model's copula given w_t (lower is better). Both expectations are computed by deterministic quadrature on a K x K
grid of the unit square (default K = 64), with the cell masses taken from the CDF of a parametric copula or from a
density evaluated at the cell centres, so no sampling error enters. Conditional models are evaluated on a grid of
covariate values (default 81 points), and the score of each observation is interpolated linearly in w. Each score
comes in two versions: raw, and margin-calibrated (the cell masses projected onto uniform margins by iterative
proportional fitting, Sinkhorn).

The second part scores the real-data candidates: unconditional and regime parametric copulas (ces_parametric),
Acar continuous (ces_acar), the nonparametric estimators (ces_nonparametric) and fitted networks as resumable jobs (build_ces_jobs, run_ces_jobs, collect_neural).
It then assembles the store in the layout of the CVPE store (assemble_ces_store) and the summary table
(ces_table).

References:
Deming, W. E., & Stephan, F. F. (1940). On a least squares adjustment of a sampled frequency table when the expected
    marginal totals are known. Annals of Mathematical Statistics, 11(4), 427-444.
Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the
    American Statistical Association, 102(477), 359-378.
Sinkhorn, R., & Knopp, P. (1967). Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of
    Mathematics, 21(2), 343-348.
Ziel, F., & Berk, K. (2019). Multivariate forecasting evaluation: On sensitive and strictly proper scoring rules.
    arXiv:1910.07325.
"""
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_K = 64
# Covariate grid of ces_conditional. Linear interpolation over 81 points gave a mean absolute error of 6e-7 and a
# maximum of 9e-6 per observation on a Clayton test case (tau from 0.2 to 0.7 in w, K = 64, 400 observations).
DEFAULT_GRID_POINTS = 81


# ── grid and distances ──────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def grid(K=DEFAULT_K):
    """Cell edges, cell centres and cell area of the K x K grid on the unit square (cached).

    Parameters:
    K (int): Number of cells per dimension. Default DEFAULT_K.

    Returns:
    tuple: (edges, centres, area). edges has shape (K + 1,), centres has shape (K * K, 2) and area is 1 / K^2.
    """
    edges = np.linspace(0.0, 1.0, K + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    g1, g2 = np.meshgrid(mid, mid, indexing="ij")
    centres = np.column_stack([g1.ravel(), g2.ravel()])
    return edges, centres, 1.0 / K ** 2


# E||U - U'|| for two independent uniform points in the unit square, (2 + sqrt(2) + 5 ln(1 + sqrt(2))) / 15
WITHIN_CELL = 0.5214054331647207


@lru_cache(maxsize=2)
def _D(K):
    """Centre-to-centre distance matrix (K^2 x K^2) as float32, built in row chunks and cached.

    It takes 67 MB at K = 64 and is built once per process. For K > 64 it is not built, and second_term chunks the
    computation on the fly.

    Parameters:
    K (int): Number of cells per dimension.

    Returns:
    np.ndarray or None: The distance matrix, or None if K > 64.
    """
    if K > 64:
        return None
    _, c, _ = grid(K)
    D = np.empty((len(c), len(c)), np.float32)
    for i in range(0, len(c), 256):
        D[i:i + 256] = np.sqrt(((c[i:i + 256, None, :] - c[None, :, :]) ** 2).sum(-1))
    return D


def second_term(m, K=None, chunk=512):
    """E||U - U'|| for cell masses m.

    The between-cell double sum m^T D m is completed by the within-cell part (sum_c m_c^2) * 0.5214 / K (two
    independent uniform points inside one cell), which removes the O(1/K) bias of treating the diagonal of D as zero.

    Parameters:
    m (np.ndarray): Cell masses, K * K values summing to 1.
    K (int): Number of cells per dimension. Default: inferred from the length of m.
    chunk (int): Rows per block when K > 64. Default 512.

    Returns:
    float: The expected distance between two independent draws.
    """
    m = np.asarray(m, float).ravel()
    K = K or int(round(math.sqrt(len(m))))
    within = float((m ** 2).sum()) * WITHIN_CELL / K
    D = _D(K)
    if D is not None:
        return float(m @ (D @ m.astype(np.float32)).astype(float)) + within
    _, c, _ = grid(K)
    tot = 0.0
    for i in range(0, len(c), chunk):
        blk = np.sqrt(((c[i:i + chunk, None, :] - c[None, :, :]) ** 2).sum(-1))
        tot += float(m[i:i + chunk] @ (blk @ m))
    return tot + within


def first_term(m, obs, K=None, chunk=2048):
    """E||U - u_t|| for every row u_t of obs.

    Parameters:
    m (np.ndarray): Cell masses, K * K values summing to 1.
    obs (np.ndarray): Observed pairs, shape (N, 2).
    K (int): Number of cells per dimension. Default: inferred from the length of m.
    chunk (int): Observations per block. Default 2048.

    Returns:
    np.ndarray: (N,) expected distances from the model to each observation.
    """
    m = np.asarray(m, float).ravel()
    K = K or int(round(math.sqrt(len(m))))
    _, c, _ = grid(K)
    obs = np.asarray(obs, float)
    out = np.empty(len(obs))
    for i in range(0, len(obs), chunk):
        o = obs[i:i + chunk]
        d = np.sqrt((o[:, None, 0] - c[None, :, 0]) ** 2 + (o[:, None, 1] - c[None, :, 1]) ** 2)
        out[i:i + chunk] = d @ m
    return out


def first_term_many(M, obs, K=None, chunk=2048):
    """E||U - u_t|| for every row u_t of obs and every model in the columns of M, sharing the distance matrix.

    Same as calling first_term once per column, but the (N, K * K) distances are computed once and multiplied with all
    models in one matrix product.

    Parameters:
    M (np.ndarray): Cell masses of r models, shape (K * K, r).
    obs (np.ndarray): Observed pairs, shape (N, 2).
    K (int): Number of cells per dimension. Default: inferred from the number of rows of M.
    chunk (int): Observations per block. Default 2048.

    Returns:
    np.ndarray: (N, r) expected distances.
    """
    M = np.asarray(M, float)
    K = K or int(round(math.sqrt(M.shape[0])))
    _, c, _ = grid(K)
    obs = np.asarray(obs, float)
    out = np.empty((len(obs), M.shape[1]))
    for i in range(0, len(obs), chunk):
        o = obs[i:i + chunk]
        d = np.sqrt((o[:, None, 0] - c[None, :, 0]) ** 2 + (o[:, None, 1] - c[None, :, 1]) ** 2)
        out[i:i + chunk] = d @ M
    return out


def es_from_masses_many(ms, obs, K=None):
    """Per-observation energy scores of several distributions on the same observations.

    Parameters:
    ms (list): Cell-mass vectors, each K * K values summing to 1.
    obs (np.ndarray): Observed pairs, shape (N, 2).
    K (int): Number of cells per dimension. Default: inferred from the length of the first vector.

    Returns:
    list: One (N,) score array per distribution, in the order of ms.
    """
    if not ms:
        return []
    M = np.column_stack([np.asarray(m, float).ravel() for m in ms])
    K = K or int(round(math.sqrt(M.shape[0])))
    f = first_term_many(M, obs, K)
    return [f[:, j] - 0.5 * second_term(M[:, j], K) for j in range(M.shape[1])]


def es_from_masses(m, obs, K=None):
    """Per-observation energy score of one distribution given by cell masses.

    Parameters:
    m (np.ndarray): Cell masses, K * K values summing to 1.
    obs (np.ndarray): Observed pairs, shape (N, 2).
    K (int): Number of cells per dimension. Default: inferred from the length of m.

    Returns:
    np.ndarray: (N,) scores.
    """
    return first_term(m, obs, K) - 0.5 * second_term(m, K)


def es_independence(obs, K=DEFAULT_K):
    """Per-observation energy score of independent uniform margins, with the same discretisation as the models.

    Parameters:
    obs (np.ndarray): Observed pairs, shape (N, 2).
    K (int): Number of cells per dimension. Default DEFAULT_K.

    Returns:
    np.ndarray: (N,) scores.
    """
    m = np.full(K * K, 1.0 / K ** 2)
    return es_from_masses(m, obs, K)


# ── cell masses ─────────────────────────────────────────────────────────────
def masses_from_cdf(cop, K=DEFAULT_K):
    """Exact cell masses of a pyvinecopulib Bicop from its CDF (the margins are exactly uniform).

    Parameters:
    cop (pyvinecopulib.Bicop): The copula.
    K (int): Number of cells per dimension. Default DEFAULT_K.

    Returns:
    np.ndarray: (K * K,) cell masses summing to 1.
    """
    edges, _, _ = grid(K)
    g1, g2 = np.meshgrid(edges, edges, indexing="ij")
    pts = np.column_stack([g1.ravel(), g2.ravel()])
    pts = np.clip(pts, 1e-12, 1 - 1e-12)
    C = np.asarray(cop.cdf(pts), float).reshape(K + 1, K + 1)
    C[0, :] = 0.0; C[:, 0] = 0.0
    C[K, :] = edges; C[:, K] = edges                     # boundary values of a copula CDF are exact
    m = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    m = np.clip(m, 0.0, None)
    return (m / m.sum()).ravel()


def masses_from_density(density_fn, w=None, K=DEFAULT_K):
    """Cell masses from a density evaluated at the cell centres, renormalised to sum to 1.

    Parameters:
    density_fn (callable): density_fn(points, w) -> densities at the (K * K, 2) points.
    w (float): Covariate value passed to density_fn. None for an unconditional density. Optional.
    K (int): Number of cells per dimension. Default DEFAULT_K.

    Returns:
    tuple: (masses, raw_total). raw_total is the integral of the density on the grid before renormalisation (1 for
        a correctly normalised model).
    """
    _, c, area = grid(K)
    d = np.asarray(density_fn(c.astype(np.float32), w), float).ravel()
    d = np.where(np.isfinite(d), d, 0.0)
    m = np.clip(d, 0.0, None) * area
    tot = float(m.sum())
    if not tot > 0:
        raise ValueError("density integrates to zero on the grid")
    return m / tot, tot


def sinkhorn_uniform_margins(m, K=None, iters=300, tol=1e-10):
    """Project cell masses onto uniform margins (1/K per row and column) by iterative proportional fitting.

    The row and column scaling is the Sinkhorn iteration (Sinkhorn & Knopp, 1967), a form of iterative proportional fitting
    (Deming & Stephan, 1940).

    Deming, W. E., & Stephan, F. F. (1940). On a least squares adjustment of a sampled frequency table when the expected
    marginal totals are known. Annals of Mathematical Statistics, 11(4), 427-444.
    Sinkhorn, R., & Knopp, P. (1967). Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal of
    Mathematics, 21(2), 343-348.

    Parameters:
    m (np.ndarray): Cell masses, K * K values.
    K (int): Number of cells per dimension. Default: inferred from the length of m.
    iters (int): Maximum number of iterations. Default 300.
    tol (float): Stop when the largest margin deviation is below this. Default 1e-10.

    Returns:
    np.ndarray: (K * K,) projected masses summing to 1.
    """
    K = K or int(round(math.sqrt(len(m))))
    M = np.asarray(m, float).reshape(K, K).copy() + 1e-300
    r = np.full(K, 1.0 / K); c = np.full(K, 1.0 / K)
    for _ in range(iters):
        M *= (r / M.sum(1))[:, None]
        M *= (c / M.sum(0))[None, :]
        if max(np.abs(M.sum(1) - r).max(), np.abs(M.sum(0) - c).max()) < tol:
            break
    return (M / M.sum()).ravel()


def margin_error(m, K=None):
    """Largest deviation of the two cumulative margins of the cell masses from the uniform CDF (a KS-type distance).

    Parameters:
    m (np.ndarray): Cell masses, K * K values.
    K (int): Number of cells per dimension. Default: inferred from the length of m.

    Returns:
    float: The distance.
    """
    K = K or int(round(math.sqrt(len(m))))
    M = np.asarray(m, float).reshape(K, K)
    grid_ = np.arange(1, K + 1) / K
    return float(max(np.abs(np.cumsum(M.sum(1)) - grid_).max(), np.abs(np.cumsum(M.sum(0)) - grid_).max()))


# ── conditional scores ──────────────────────────────────────────────────────
def ces_conditional(mass_fn, w_obs, obs, w_grid, K=DEFAULT_K, calibrated=False, diagnostics=None):
    """Per-observation energy score of a covariate-dependent distribution.

    The score of each observation is computed at the two grid values of w that bracket it and interpolated linearly.
    A grid point that gets zero weight for an observation (the observation sits exactly on the other grid point) is
    not evaluated for it. w_grid must cover w_obs. Values outside are clamped to the grid ends, so the grid should be built from the range
    of the observations themselves.

    Parameters:
    mass_fn (callable): mass_fn(w) -> cell masses (K * K,), or (masses, raw_total) as from masses_from_density.
    w_obs (np.ndarray): Covariate value of each observation, shape (N,).
    obs (np.ndarray): Observed pairs, shape (N, 2).
    w_grid (np.ndarray): Increasing grid of covariate values.
    K (int): Number of cells per dimension. Default DEFAULT_K.
    calibrated (bool): Project the masses onto uniform margins before scoring. Default False.
    diagnostics (dict): If given, receives per grid point the lists "w", "margin_error" (of the raw masses) and
        "raw_total". Optional.

    Returns:
    np.ndarray: (N,) scores.
    """
    w_obs = np.asarray(w_obs, float); obs = np.asarray(obs, float)
    w_grid = np.asarray(w_grid, float)
    pos = np.clip(np.searchsorted(w_grid, w_obs, side="right") - 1, 0, len(w_grid) - 2)
    frac = np.clip((w_obs - w_grid[pos]) / (w_grid[pos + 1] - w_grid[pos]), 0.0, 1.0)
    need = {}
    for g in range(len(w_grid)):
        idx = np.flatnonzero(((pos == g) & (frac < 1.0)) | ((pos + 1 == g) & (frac > 0.0)))
        if len(idx):
            need[g] = idx
    es_at = {}
    for g, idx in need.items():
        out = mass_fn(float(w_grid[g]))
        tot = np.nan
        if isinstance(out, tuple):
            out, tot = out
        raw_m = np.asarray(out, float)
        m = sinkhorn_uniform_margins(raw_m, K) if calibrated else raw_m
        es_at[g] = (idx, es_from_masses(m, obs[idx], K))
        if diagnostics is not None:
            diagnostics.setdefault("w", []).append(float(w_grid[g]))
            diagnostics.setdefault("margin_error", []).append(margin_error(raw_m, K))
            diagnostics.setdefault("raw_total", []).append(float(tot))
    lo = np.zeros(len(obs)); hi = np.zeros(len(obs))
    for g, (idx, v) in es_at.items():
        sel_lo = pos[idx] == g
        lo[idx[sel_lo]] = v[sel_lo]
        sel_hi = pos[idx] + 1 == g
        hi[idx[sel_hi]] = v[sel_hi]
    return (1.0 - frac) * lo + frac * hi


def ces_conditional_both(mass_fn, w_obs, obs, w_grid, K=DEFAULT_K, diagnostics=None):
    """Raw and margin-calibrated ces_conditional in one pass: the masses at each grid point are computed once.

    Gives the same scores as ces_conditional(..., calibrated=False) and ces_conditional(..., calibrated=True), but
    mass_fn (a network forward pass or a copula CDF on the K * K cells) is evaluated once per grid point instead of
    twice, and both score vectors share one distance matrix.

    Parameters:
    mass_fn (callable): mass_fn(w) -> cell masses (K * K,), or (masses, raw_total) as from masses_from_density.
    w_obs (np.ndarray): Covariate value of each observation, shape (N,).
    obs (np.ndarray): Observed pairs, shape (N, 2).
    w_grid (np.ndarray): Increasing covariate grid covering w_obs.
    K (int): Number of cells per dimension. Default DEFAULT_K.
    diagnostics (dict): If given, receives per grid point the lists "w", "margin_error" (of the raw masses) and
        "raw_total". Optional.

    Returns:
    tuple: (raw, calibrated), each an (N,) array of scores.
    """
    w_obs = np.asarray(w_obs, float); obs = np.asarray(obs, float)
    w_grid = np.asarray(w_grid, float)
    pos = np.clip(np.searchsorted(w_grid, w_obs, side="right") - 1, 0, len(w_grid) - 2)
    frac = np.clip((w_obs - w_grid[pos]) / (w_grid[pos + 1] - w_grid[pos]), 0.0, 1.0)
    lo = {k: np.zeros(len(obs)) for k in ("raw", "cal")}
    hi = {k: np.zeros(len(obs)) for k in ("raw", "cal")}
    for g in range(len(w_grid)):
        idx = np.flatnonzero(((pos == g) & (frac < 1.0)) | ((pos + 1 == g) & (frac > 0.0)))
        if not len(idx):
            continue
        out = mass_fn(float(w_grid[g]))
        tot = np.nan
        if isinstance(out, tuple):
            out, tot = out
        raw_m = np.asarray(out, float)
        v_raw, v_cal = es_from_masses_many([raw_m, sinkhorn_uniform_margins(raw_m, K)], obs[idx], K)
        sel_lo = pos[idx] == g
        sel_hi = pos[idx] + 1 == g
        for key, v in (("raw", v_raw), ("cal", v_cal)):
            lo[key][idx[sel_lo]] = v[sel_lo]
            hi[key][idx[sel_hi]] = v[sel_hi]
        if diagnostics is not None:
            diagnostics.setdefault("w", []).append(float(w_grid[g]))
            diagnostics.setdefault("margin_error", []).append(margin_error(raw_m, K))
            diagnostics.setdefault("raw_total", []).append(float(tot))
    return ((1.0 - frac) * lo["raw"] + frac * hi["raw"],
            (1.0 - frac) * lo["cal"] + frac * hi["cal"])


# ── parametric copula helper ────────────────────────────────────────────────
def cop_from_row(row):
    """Rebuild a Bicop from a stripped parametric-fit row.

    Parameters:
    row (dict): Row with the keys "name", "params" and optionally "rotation".

    Returns:
    pyvinecopulib.Bicop: The copula.
    """
    from copula_families import make_bicop
    return make_bicop(row["name"], row["params"], row.get("rotation"))


# ── real-data comparison ────────────────────────────────────────────────────
TARGETS = ("raw", "calibrated")


# ── parametric and Acar models (in-process) ─────────────────────────────────
def _es_both(m, obs, K):
    """Raw and margin-calibrated per-observation scores of one set of cell masses.

    Parameters:
    m (np.ndarray): Cell masses, K * K values.
    obs (np.ndarray): Observed pairs, shape (N, 2).
    K (int): Number of cells per dimension.

    Returns:
    dict: {"raw": (N,), "calibrated": (N,)}.
    """
    return dict(raw=es_from_masses(m, obs, K),
                calibrated=es_from_masses(sinkhorn_uniform_margins(m, K), obs, K))


def ces_parametric(param, arrays, folds, K=DEFAULT_K):
    """CES of the parametric candidates on the test rows of each fold, plus the independence reference.

    The candidates are the 8 unconditional families and the 49 binary-regime pairs (one family for each of the low and
    high regime). A binary-regime model scores each test row with the family of its own regime.

    Parameters:
    param (dict): Fold -> result of fold_pipeline's parametric_fold_job (kernel pseudo-observations), with the keys
        "rows_uncond" and "rows_regime".
    arrays (dict): Fold -> arrays with "u1", "u2", "test" and "high".
    folds (list): Folds to score.
    K (int): Number of cells per dimension. Default DEFAULT_K.

    Returns:
    dict: {model name: {fold: {"raw": (n,), "calibrated": (n,)}}}, NaN outside the test rows.
    """
    import fold_pipeline as FP
    out = {}
    for f in folds:
        A, P = arrays[f], param[f]
        u = np.column_stack([A["u1"], A["u2"]]); n = len(u)
        te, hi = np.asarray(A["test"], bool), np.asarray(A["high"], bool)
        # unconditional: all families (raw and calibrated) share one distance matrix on the test rows
        rows = list(P["rows_uncond"])
        ms = [masses_from_cdf(cop_from_row(r), K) for r in rows]
        sc = es_from_masses_many([x for m in ms for x in (m, sinkhorn_uniform_margins(m, K))], u[te], K)
        for j, r in enumerate(rows):
            e = dict(raw=sc[2 * j], calibrated=sc[2 * j + 1])
            out.setdefault(FP.unconditional_name(r["name"]), {})[f] = {t: _spread(e[t], te, n) for t in TARGETS}
        # regimes: same, for the test rows of each regime
        reg = {}
        for rn, mask in (("low", te & ~hi), ("high", te & hi)):
            rows = list(P["rows_regime"][rn])
            ms = [masses_from_cdf(cop_from_row(r), K) for r in rows]
            sc = es_from_masses_many([x for m in ms for x in (m, sinkhorn_uniform_margins(m, K))], u[mask], K)
            for j, r in enumerate(rows):
                e = dict(raw=sc[2 * j], calibrated=sc[2 * j + 1])
                reg.setdefault(rn, {})[r["name"]] = {t: _spread(e[t], mask, n) for t in TARGETS}
        for fl, el in reg["low"].items():
            for fh, eh in reg["high"].items():
                name = FP.display_binary_name(fl, fh)
                out.setdefault(name, {})[f] = {t: np.where(np.isfinite(el[t]), el[t], eh[t]) for t in TARGETS}
        ind = es_independence(u[te], K)
        out.setdefault("Independence (reference)", {})[f] = {t: _spread(ind, te, n) for t in TARGETS}
    return out


def _spread(vals, mask, n):
    """Place scores on the rows of a mask in a length-n array, NaN elsewhere.

    Parameters:
    vals (np.ndarray): Scores for the rows where mask is True.
    mask (np.ndarray): Boolean mask of length n.
    n (int): Length of the output.

    Returns:
    np.ndarray: (n,) array.
    """
    a = np.full(n, np.nan); a[mask] = vals
    return a


def ces_acar(acar, arrays, folds, K=DEFAULT_K, G=DEFAULT_GRID_POINTS):
    """CES of the Acar continuous models: masses on a covariate grid, the score of each test step interpolated in w.

    Parameters:
    acar (dict): {(fold, family): result of fold_pipeline's acar_job}. Entries without a curve are skipped.
    arrays (dict): Fold -> arrays with "u1", "u2", "w" and "test".
    folds (list): Folds to score.
    K (int): Number of cells per dimension. Default DEFAULT_K.
    G (int): Number of covariate grid points. Default DEFAULT_GRID_POINTS.

    Returns:
    dict: {model name: {fold: {"raw": (n,), "calibrated": (n,)}}}, NaN outside the test rows.
    """
    import fold_pipeline as FP
    out = {}
    for (f, fam), r in acar.items():
        if f not in folds or r.get("curve") is None:
            continue
        A = arrays[f]
        te = np.asarray(A["test"], bool); n = len(A["u1"])
        u = np.column_stack([A["u1"], A["u2"]])[te]; w = np.asarray(A["w"], float)[te]
        grid = np.linspace(w.min(), w.max(), G)
        mf = lambda wv, fam=fam, r=r: masses_from_cdf(FP._acar_cop_at(fam, r["curve"], r["fixed"], wv), K)
        raw, cal = ces_conditional_both(mf, w, u, grid, K)
        out.setdefault(FP.acar_name(fam), {})[f] = dict(raw=_spread(raw, te, n), calibrated=_spread(cal, te, n))
    return out


def ces_nonparametric(nonparam, folds):
    """CES of the nonparametric estimators, read from the fold jobs (they score their own test rows at K x K cells).

    Parameters:
    nonparam (dict): {(fold, kind): result of fold_pipeline's nonparametric_job}, with the keys model, es_raw and es_cal.
    folds (list): Folds to score.

    Returns:
    dict: {model name: {fold: {"raw": (n,), "calibrated": (n,)}}}, NaN outside the test rows.
    """
    out = {}
    for (f, kind), r in nonparam.items():
        if f in folds and r.get("es_raw") is not None:
            out.setdefault(r["model"], {})[f] = dict(raw=r["es_raw"], calibrated=r["es_cal"])
    return out


# ── neural networks (resumable jobs on reconstructable fitted nets) ─────────
def _density_fn(net, kind):
    """Density function of a fitted network, in the form density_fn(points, w) used by masses_from_density.

    Parameters:
    net: The fitted network.
    kind (str): "codine_uncond" or "codine_regime" (CODINE), "ccodine" (C-CODINE) or anything else (C-CopFlow).

    Returns:
    callable: density_fn(points, w) -> densities.
    """
    if kind in ("codine_uncond", "codine_regime"):
        from codine import eval_density
        return lambda pts, w: eval_density(net, pts)
    if kind == "ccodine":
        from codine import eval_conditional_density
        return lambda pts, w: eval_conditional_density(net, pts, w)
    from conditional_copula_flow import eval_flow_density
    return lambda pts, w: eval_flow_density(net, pts, w)


def ces_net_job(job):
    """CES of one fitted network on the test rows of its fold, raw and margin-calibrated.

    Defined at the top level so that a process pool can run it. A network that cannot be reconstructed is reported with
    status "not reconstructable" and no scores.

    Parameters:
    job (dict): Job from build_ces_jobs, with the keys fold, seed, model, net_kind, regime, arrays, K, G, batch_dir
        and key.

    Returns:
    dict: Keys fold, seed, model, kind, regime, status, reason, es_raw, es_cal (each (n,) with NaN outside the rows
        scored), max_margin_error, mean_raw_total, min_raw_total, max_raw_total, source and runtime.
    """
    import torch
    torch.set_num_threads(1)
    import fold_models as MS
    t0 = time.time()
    K, G = int(job["K"]), int(job["G"])
    out = dict(fold=job["fold"], seed=job["seed"], model=job["model"], kind=job["net_kind"], regime=job.get("regime"),
               status="ok", reason="", es_raw=None, es_cal=None)
    try:
        net, meta = MS.load_fitted(job["batch_dir"], job["key"])
    except MS.NotReconstructable as exc:
        out.update(status="not reconstructable", reason=str(exc)[:200], runtime=time.time() - t0)
        return out
    A = job["arrays"]
    u = np.column_stack([A["u1"], A["u2"]]); w = np.asarray(A["w"], float); n = len(u)
    te, hi = np.asarray(A["test"], bool), np.asarray(A["high"], bool)
    sel = te
    if job["net_kind"] == "codine_regime":
        sel = te & (hi if job["regime"] == "high" else ~hi)
    dens = _density_fn(net, job["net_kind"])
    diag = {}
    with torch.no_grad():
        if job["net_kind"] in ("codine_uncond", "codine_regime"):
            m, tot = masses_from_density(dens, None, K)
            e_raw, e_cal = es_from_masses_many([m, sinkhorn_uniform_margins(m, K)], u[sel], K)
            diag = dict(w=[np.nan], margin_error=[margin_error(m, K)], raw_total=[tot])
        else:
            grid = np.linspace(w[sel].min(), w[sel].max(), G)
            mf = lambda wv: masses_from_density(dens, float(wv), K)
            e_raw, e_cal = ces_conditional_both(mf, w[sel], u[sel], grid, K, diagnostics=diag)
    out["es_raw"], out["es_cal"] = _spread(e_raw, sel, n), _spread(e_cal, sel, n)
    out.update(max_margin_error=float(np.nanmax(diag["margin_error"])), mean_raw_total=float(np.nanmean(diag["raw_total"])),
               min_raw_total=float(np.nanmin(diag["raw_total"])), max_raw_total=float(np.nanmax(diag["raw_total"])),
               source=meta.get("source", "state file"), runtime=time.time() - t0)
    return out


ces_net_job._ram_heavy = True


def build_ces_jobs(arrays, batch_dir, folds=None, seeds=None, K=DEFAULT_K, G=DEFAULT_GRID_POINTS):
    """One job per (fold, seed, network). Only the slim arrays that the score needs travel to the worker.

    Parameters:
    arrays (dict): Fold -> arrays with "u1", "u2", "w", "test" and "high".
    batch_dir (str or Path): Directory of the fitted networks.
    folds (list): Folds. Default fold_pipeline.FOLDS.
    seeds (list): Seeds. Default fold_pipeline.SEEDS.
    K (int): Number of cells per dimension. Default DEFAULT_K.
    G (int): Number of covariate grid points. Default DEFAULT_GRID_POINTS.

    Returns:
    list: Job dicts for ces_net_job.
    """
    import fold_pipeline as FP
    folds = folds or FP.FOLDS; seeds = seeds if seeds is not None else FP.SEEDS
    kinds = [(FP.UNCOND_CODINE, "codine_uncond", None)]
    kinds += [(f"{FP.SPLIT_CODINE}|{r}", "codine_regime", r) for r in ("low", "high")]
    kinds += [(m, "ccodine", None) for m in FP.CCODINE_VARIANTS] + [(m, "ccopflow", None) for m in FP.CCOPFLOW_VARIANTS]
    jobs = []
    for f in folds:
        A = {k: arrays[f][k] for k in ("u1", "u2", "w", "test", "high")}
        for s in seeds:
            for model, nk, reg in kinds:
                jobs.append(dict(kind="ces", fold=f, seed=s, model=model, net_kind=nk, regime=reg, arrays=A, K=K, G=G,
                                 batch_dir=str(batch_dir), key=FP.net_key(f, s, model)))
    return jobs


def ces_job_key(j):
    """Key under which the result of a CES job is stored.

    Parameters:
    j (dict): Job from build_ces_jobs.

    Returns:
    str: "ces__" followed by the key of the fitted network.
    """
    return "ces__" + j["key"]


def run_ces_jobs(arrays, batch_dir=None, max_workers=None, **kw):
    """Run the CES stage as resumable jobs (result files in <batch_dir>/ces).

    Parameters:
    arrays (dict): Fold -> arrays with "u1", "u2", "w", "test" and "high".
    batch_dir (str or Path): Directory of the fitted networks. Default fold_pipeline.BATCH_DIR.
    max_workers (int): Number of worker processes. Optional.
    **kw: Forwarded to build_ces_jobs (folds, seeds, K, G).

    Returns:
    dict: {key: result of ces_net_job}.
    """
    import fold_pipeline as FP
    bd = Path(batch_dir or FP.BATCH_DIR)
    jobs = build_ces_jobs(arrays, bd, **kw)
    return FP.run_batch(jobs, max_workers=max_workers, batch_dir=bd / "ces", fn=ces_net_job, key_fn=ces_job_key,
                        desc="CES of fitted networks")


def collect_neural(results):
    """Average the network scores over the reconstructable seeds and report the coverage.

    The two halves of the split CODINE model (low and high regime) have disjoint test rows and are merged into one
    model score. A split model needs both halves.

    Parameters:
    results (dict): Output of run_ces_jobs.

    Returns:
    tuple: (out, seeds_used, coverage). out is {model: {fold: {"raw", "calibrated"}}}, seeds_used is
        {model: {fold: number of seeds}} and coverage is a DataFrame with the status of every job.
    """
    import fold_pipeline as FP
    per = {}
    cov = []
    for r in results.values():
        m = r["model"].split("|")[0] if r["model"].startswith(FP.SPLIT_CODINE) else r["model"]
        cov.append(dict(model=m, fold=r["fold"], seed=r["seed"], half=r.get("regime"), status=r["status"], reason=r.get("reason", "")))
        if r["status"] != "ok":
            continue
        per.setdefault((m, r["fold"], r["seed"]), []).append(r)
    cov = pd.DataFrame(cov)
    out, seeds_used = {}, {}
    by_mf = {}
    for (m, f, s), rs in per.items():
        need = 2 if m == FP.SPLIT_CODINE else 1
        if len(rs) < need:
            continue                                         # both halves are needed for the split model
        sr = np.vstack([x["es_raw"] for x in rs]); sc = np.vstack([x["es_cal"] for x in rs])
        fin = np.isfinite(sr).any(axis=0)                     # split halves have disjoint test rows
        raw = np.where(fin, np.nansum(sr, axis=0), np.nan); cal = np.where(fin, np.nansum(sc, axis=0), np.nan)
        by_mf.setdefault((m, f), []).append((raw, cal))
    for (m, f), lst in by_mf.items():
        out.setdefault(m, {})[f] = dict(raw=np.mean([a for a, _ in lst], axis=0), calibrated=np.mean([b for _, b in lst], axis=0))
        seeds_used.setdefault(m, {})[f] = len(lst)
    return out, seeds_used, cov


# ── store (same layout as the CVPE store) and tables ────────────────────────
def assemble_ces_store(store, parts, folds, seeds_used=None, target="raw"):
    """CES tensor for the candidates that have a score in every fold, in the model order of the CVPE store.

    Parameters:
    store (dict): The CVPE store, with "models", "tensor", "group" and "in_pool".
    parts (list): Dicts {model: {fold: {"raw", "calibrated"}}} (parametric, Acar, nonparametric, neural).
    folds (list): Folds.
    seeds_used (dict): {model: {fold: number of seeds}}. Optional.
    target (str): "raw" or "calibrated". Default "raw".

    Returns:
    dict: Keys tensor (n, M, F), models, folds, group, in_pool, no_ces (reason per model without a score), target,
        seeds_used and independence (the independence reference, (n, F), or None).
    """
    allm = {}
    for p in parts:
        allm.update(p)
    models = [m for m in store["models"] if m in allm and all(f in allm[m] for f in folds)]
    n = store["tensor"].shape[0]
    tensor = np.full((n, len(models), len(folds)), np.nan)
    for j, m in enumerate(models):
        for k, f in enumerate(folds):
            tensor[:, j, k] = allm[m][f][target]
    missing = {m: ("no CES: " + ("no reconstructable network" if store["group"].get(m) in ("Neural",) or "CODINE" in m else "not computed"))
               for m in store["models"] if m not in models}
    ref = allm.get("Independence (reference)")
    return dict(tensor=tensor, models=models, folds=list(folds), group={m: store["group"][m] for m in models},
                in_pool={m: store["in_pool"][m] for m in models}, no_ces=missing, target=target,
                seeds_used=seeds_used or {}, independence=None if ref is None else np.stack([ref[f][target] for f in folds], axis=1))


def ces_table(ces_store, K=DEFAULT_K, G=DEFAULT_GRID_POINTS):
    """Summary table with one row per model: mean CES and its improvement over independence.

    The mean is the nanmean over folds and then over steps, as for the CVPE.

    Parameters:
    ces_store (dict): Output of assemble_ces_store.
    K (int): Number of cells per dimension, recorded in the table attributes. Default DEFAULT_K.
    G (int): Number of covariate grid points, recorded in the table attributes. Default DEFAULT_GRID_POINTS.

    Returns:
    pd.DataFrame: Indexed by model, sorted by CES, with the columns group, in_pool, ces, vs_independence and
        seeds_used. The attributes hold independence_ces, estimator and target.
    """
    rows = []
    ind = ces_store.get("independence")
    ind_mean = float(np.nanmean(np.nanmean(ind, axis=1))) if ind is not None else np.nan
    for j, m in enumerate(ces_store["models"]):
        v = float(np.nanmean(np.nanmean(ces_store["tensor"][:, j, :], axis=1)))
        su = ces_store["seeds_used"].get(m)
        rows.append(dict(model=m, group=ces_store["group"][m], in_pool=ces_store["in_pool"][m], ces=v,
                         vs_independence=ind_mean - v, seeds_used=(min(su.values()) if su else np.nan)))
    t = pd.DataFrame(rows).set_index("model").sort_values("ces")
    t.attrs.update(independence_ces=ind_mean, estimator=f"deterministic quadrature K={K}, covariate grid G={G} (no Monte Carlo)",
                   target=ces_store["target"])
    return t
