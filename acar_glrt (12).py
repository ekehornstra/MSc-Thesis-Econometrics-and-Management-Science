"""acar_glrt.py

Generalised likelihood-ratio test (GLRT) of the null that the copula calibration function eta(w) is constant in
the covariate w (Acar, Craiu & Yao, 2013), with its real-data wrapper and the pilot simulation that checks its size
and power.

The statistic lambda_n(h) is the gain in log-likelihood of a local-constant kernel estimate of eta(w) (Epanechnikov
kernel, bandwidth h by cross-validated likelihood) over the best constant eta, within one copula family. The
asymptotic reference is a scaled chi-square distribution (kernel_constants, reference, p_asymptotic). For the
serially dependent real data it is reported for comparison, and the p-value comes from a parametric bootstrap
under H0 with the covariate path held fixed (null_parametric). The local-constant estimator (p = 0) is the one
Acar et al. (2013, Section 2.3) recommend for testing constancy. run_glrt_real tests a primary family and, as
exploratory, other families with Holm-adjusted p-values (Holm, 1979). The primary family is the one with the
lowest AIC (aic_primary_family). run() is the pilot simulation: type I error before power, then sample size,
family, bandwidth and a serially dependent covariate.

Usage: python acar_glrt.py [output.json] [--quick]

References:
Acar, E. F., Craiu, R. V., & Yao, F. (2013). Statistical testing of covariate effects in conditional copula
    models. Electronic Journal of Statistics, 7, 2822-2850.
Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics,
    6(2), 65-70.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

from conditional_copula import FAMILY_INFO, fit_global, log_density
from run_support import resolve_pool_workers
import os


# Candidate bandwidths as fractions of |X|: 12 pilot values equally spaced on a log scale, as in Acar et al. (2013), whose
# simulations span 0.11 to 0.99 of |X|. The range here is 0.04 to 0.60.
PILOT_FRACTIONS = np.geomspace(0.04, 0.60, 12)


# ── kernel constants and the asymptotic reference ───────────────────────────
def kernel_constants(n_grid=40001):
    """Constants of the scaled chi-square reference for the Epanechnikov kernel, by numerical integration.

    With K the Epanechnikov kernel and K*K its convolution: c_K = K(0) - 0.5 * int K^2, int_sq = int (K - 0.5 K*K)^2,
    r_K = c_K / int_sq and df_coefficient = r_K * c_K. The reference of the GLRT then is r_K * lambda_n ~ chi-square
    with df = r_K * (|X| / h) * c_K degrees of freedom (see reference). For the Epanechnikov kernel c_K = 0.45 and
    r_K = 2.115, as in Acar et al. (2013); df_coefficient is then 0.952 (the paper prints 0.968).

    Parameters:
    n_grid (int): Number of grid points on [-2, 2] for the integrals. Default 40001.

    Returns:
    dict: Keys K0, int_K2, c_K, int_sq, r_K, df_coefficient.
    """
    t = np.linspace(-2.0, 2.0, n_grid); dt = t[1] - t[0]
    K = np.where(np.abs(t) <= 1, 0.75 * (1 - t ** 2), 0.0)
    K2 = float(np.sum(K ** 2) * dt)
    KK = np.convolve(K, K, mode="same") * dt                     # K * K on the same grid
    c_K = 0.75 - 0.5 * K2
    nu_fac = float(np.sum((K - 0.5 * KK) ** 2) * dt)             # int (K - 0.5 K*K)^2
    r_K = c_K / nu_fac                                           # r_K = 2 mu/nu = c_K / int(...)^2
    return dict(K0=0.75, int_K2=K2, c_K=c_K, int_sq=nu_fac, r_K=r_K, df_coefficient=r_K * c_K)


_KC = None


def _kc():
    """Return the kernel constants, computed once and cached.

    Returns:
    dict: The output of kernel_constants().
    """
    global _KC
    if _KC is None:
        _KC = kernel_constants()
    return _KC


def reference(h, x_range):
    """Scaled chi-square reference of the GLRT for bandwidth h and covariate range |X|.

    mu_n = (|X| / h) * c_K, nu_n = 2 * (|X| / h) * int_sq, r_K = 2 * mu_n / nu_n and df = r_K * mu_n (Acar et al.,
    2013, Theorem 1). The asymptotics need many intervals |X| / h, and this implementation treats the reference as
    not applicable below |X| / h = 5 (`ok` is False).

    Parameters:
    h (float): Bandwidth, on the scale of the covariate.
    x_range (float): Range |X| of the covariate.

    Returns:
    dict: Keys r_K, df, mu_n, nu_n, ok (bool) and note (empty if ok).
    """
    kc = _kc()
    mu = (x_range / h) * kc["c_K"]
    nu = (2 * x_range / h) * kc["int_sq"]
    r = 2 * mu / nu
    ok = (x_range / h) >= 5
    return dict(r_K=r, df=r * mu, mu_n=mu, nu_n=nu, ok=bool(ok),
                note="" if ok else "|X|/h < 5: effective df too small, asymptotic reference not used")


def p_asymptotic(lam, h, x_range):
    """Asymptotic p-value of the GLRT statistic from the scaled chi-square reference.

    Parameters:
    lam (float): Value of lambda_n (non-negative).
    h (float): Bandwidth.
    x_range (float): Range |X| of the covariate.

    Returns:
    tuple: (p, ref). p (float) is the survival function of chi-square(df) at r_K * lam, and ref is the output of
        reference(). p is computed even if ref["ok"] is False.
    """
    from scipy.stats import chi2
    ref = reference(h, x_range)
    return float(chi2.sf(ref["r_K"] * lam, ref["df"])), ref


# ── local-constant likelihood machinery ─────────────────────────────────────
def _eta_grid(family, m=161):
    """Grid of calibration-scale values eta over which the local likelihood is maximised.

    The range depends on the link of the family: theta = tanh(eta) with |rho| <= 0.995 for Gaussian and Student-t,
    and theta = sign(eta) * (par0 + |eta|) for the signed families.

    Parameters:
    family (str): Family key.
    m (int): Number of grid points. Default 161.

    Returns:
    np.ndarray: (m,) grid.
    """
    if family in ("gaussian", "student"):
        return np.linspace(-3.0, 3.0, m)              # theta = tanh(eta), |rho| <= 0.995
    if family == "frank":
        return np.linspace(-20.0, 20.0, m)
    return np.linspace(-12.0, 12.0, m)                # signed families: theta = sign(eta)(par0 + |eta|)


def _loglik_matrix(u1, u2, family, fixed, eta):
    """Log density of every observation at every grid value of eta.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    family (str): Family key.
    fixed (list or None): Fixed extra parameters (the Student-t degrees of freedom).
    eta (np.ndarray): Grid of calibration-scale values, shape (m,).

    Returns:
    np.ndarray: (n, m) matrix with entries l(eta_k; U_i).
    """
    g_inv = FAMILY_INFO[family]["link_inv"]
    cols = [log_density(u1, u2, float(g_inv(e)), family, fixed or None) for e in eta]
    return np.column_stack(cols)


def _kernel_matrix(x, h):
    """Sparse kernel weight matrix W_ij = K((X_j - X_i) / h) with the Epanechnikov kernel, built without Python loops.

    Parameters:
    x (np.ndarray): Covariate values, shape (n,).
    h (float): Bandwidth.

    Returns:
    scipy.sparse.csr_matrix: (n, n) matrix, non-zero only where |X_j - X_i| <= h.
    """
    n = len(x)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    lo = np.searchsorted(xs, xs - h, side="left"); hi = np.searchsorted(xs, xs + h, side="right")
    cnt = hi - lo
    rows_sorted = np.repeat(np.arange(n), cnt)
    start = np.cumsum(cnt) - cnt
    cols_sorted = np.arange(cnt.sum()) - np.repeat(start, cnt) + np.repeat(lo, cnt)
    t = (xs[cols_sorted] - xs[rows_sorted]) / h
    vals = 0.75 * (1 - t ** 2)
    return sparse.csr_matrix((vals, (order[rows_sorted], order[cols_sorted])), shape=(n, n))


def _remove_window(W, b):
    """Zero all entries of W with |i - j| <= b, where i and j are positions in the original order of the data.

    Parameters:
    W (scipy.sparse matrix): Kernel weight matrix.
    b (int): Half-width of the window to remove (b = 0 removes the diagonal only).

    Returns:
    scipy.sparse.csr_matrix: The matrix with the window removed.
    """
    W = W.tocoo()
    keep = np.abs(W.row - W.col) > b
    return sparse.csr_matrix((W.data[keep], (W.row[keep], W.col[keep])), shape=W.shape)


def _refine(M, eta):
    """Arg-max of each row of M over the eta grid, with parabolic refinement around the grid maximum.

    A maximum at the first or last grid point is returned unrefined and flagged as a boundary estimate.

    Parameters:
    M (np.ndarray): (n, m) kernel-weighted log-likelihood over the grid.
    eta (np.ndarray): Equally spaced grid, shape (m,).

    Returns:
    tuple: (eta_hat, at_boundary). eta_hat is (n,) and at_boundary is a boolean mask of length n.
    """
    k = np.argmax(M, axis=1)
    n, m = M.shape
    idx = np.arange(n)
    kk = np.clip(k, 1, m - 2)
    y0, y1, y2 = M[idx, kk - 1], M[idx, kk], M[idx, kk + 1]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(den < 0, 0.5 * (y0 - y2) / den, 0.0)
    step = eta[1] - eta[0]
    est = eta[kk] + np.clip(delta, -1, 1) * step
    boundary = (k == 0) | (k == m - 1)
    est = np.where(boundary, eta[k], est)
    return est, boundary


class LocalConstant:
    """Local-constant likelihood estimator of the calibration function at every X_i, vectorised over an eta grid.

    The log-likelihood of each observation is evaluated once on the eta grid (Lmat). The kernel-weighted sum
    W @ Lmat then gives the local log-likelihood of every X_i at every grid value.
    """

    def __init__(self, u1, u2, x, family, fixed=None, m=121):
        """Evaluate the log-likelihood of every observation on the eta grid.

        Parameters:
        u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
        u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
        x (np.ndarray): Covariate values, shape (n,).
        family (str): Family key.
        fixed (list): Fixed extra parameters (the Student-t degrees of freedom). Optional.
        m (int): Number of eta grid points. Default 121.
        """
        self.u1, self.u2, self.x = np.asarray(u1, float), np.asarray(u2, float), np.asarray(x, float)
        self.family = family
        self.eta = _eta_grid(family, m)
        self.fixed = fixed
        self.Lmat = _loglik_matrix(self.u1, self.u2, family, fixed, self.eta)         # (n, m)
        self.n = len(self.x)
        self.range = float(self.x.max() - self.x.min())
        self.g_inv = FAMILY_INFO[family]["link_inv"]

    def _loglik_at(self, eta_i):
        """Log density of every observation at its own value of eta.

        Parameters:
        eta_i (np.ndarray): One calibration-scale value per observation, shape (n,).

        Returns:
        np.ndarray: (n,) log densities.
        """
        theta = self.g_inv(np.asarray(eta_i, float))
        return log_density(self.u1, self.u2, theta, self.family, self.fixed or None)

    def fit(self, h, exclude=None):
        """Estimate eta_hat_h(X_i) for every i.

        Parameters:
        h (float): Bandwidth.
        exclude (int): None keeps observation i in its own local fit (used for the statistic). An integer b >= 0 leaves
            out the window |j - i| <= b (used for cross-validation). Optional.

        Returns:
        tuple: (est, boundary, wsum). est holds the estimates, boundary flags those at the edge of the eta grid, and wsum
            holds the sum of the kernel weights of each observation.
        """
        W = _kernel_matrix(self.x, h)
        if exclude is not None:
            W = _remove_window(W, int(exclude))
        M = W @ self.Lmat
        est, boundary = _refine(M, self.eta)
        wsum = np.asarray(W.sum(axis=1)).ravel()
        return est, boundary, wsum

    def cv_score(self, h, b=0):
        """Leave-one-out (b = 0) or leave-block-out (b > 0) cross-validated log-likelihood B(h) of Acar et al. (2013).

        Observations with no kernel weight are skipped. If they are more than 10% of the sample, -inf is returned.

        Parameters:
        h (float): Bandwidth.
        b (int): Half-width of the window left out. Default 0.

        Returns:
        float: The cross-validated log-likelihood.
        """
        est, boundary, wsum = self.fit(h, exclude=b)
        ok = wsum > 1e-8
        ll = self._loglik_at(est)
        return float(np.sum(ll[ok])) if ok.mean() > 0.9 else -np.inf


def _global_fit(u1, u2, family):
    """Constant-calibration maximum-likelihood fit (the fit under H0).

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    family (str): Family key.

    Returns:
    tuple: (primary, fixed, ll). primary is the signed parameter, fixed the fixed extra parameters (Student-t degrees
        of freedom) and ll the log-likelihood L_n(H0).
    """
    primary, fixed, _ = fit_global(u1, u2, family)
    ll = float(np.sum(log_density(u1, u2, primary, family, fixed or None)))
    return primary, fixed, ll


def select_bandwidth(lc, fractions=PILOT_FRACTIONS, b=0):
    """Choose the bandwidth over the pilot grid by cross-validated likelihood (h = fraction * |X|).

    Parameters:
    lc (LocalConstant): The estimator for the sample.
    fractions (np.ndarray): Candidate bandwidths as fractions of |X|. Default PILOT_FRACTIONS.
    b (int): Half-width of the window left out in the cross-validation. Default 0.

    Returns:
    tuple: (h, rows). h is the bandwidth with the highest cross-validated likelihood and rows is the list of
        (h, score) pairs.
    """
    rows = [(f * lc.range, lc.cv_score(f * lc.range, b)) for f in fractions]
    h = max(rows, key=lambda r: r[1])[0]
    return h, rows


# ── the test ────────────────────────────────────────────────────────────────
def glrt_constant(u1, u2, x, family, h=None, cv_exclude=0, fractions=PILOT_FRACTIONS, lc=None):
    """GLRT statistic lambda_n(h) for H0: eta constant, against a local-constant alternative.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    x (np.ndarray): Covariate values, shape (n,).
    family (str): Family key.
    h (float): Bandwidth. If None, it is selected by cross-validated likelihood. Optional.
    cv_exclude (int): Half-width of the window left out in the cross-validation. Default 0.
    fractions (np.ndarray): Candidate bandwidths as fractions of |X|. Default PILOT_FRACTIONS.
    lc (LocalConstant): Precomputed estimator, to reuse the likelihood grid. Optional.

    Returns:
    dict: Keys family, n, lam, L1, L0, h, h_source, x_range, abs_X_over_h, r_K, df, p_asymptotic (NaN when |X| / h < 5),
        boundary_fraction, eta_tilde, notes (warnings that must accompany the number) and cv_table.
    """
    u1 = np.asarray(u1, float); u2 = np.asarray(u2, float); x = np.asarray(x, float)
    primary, fixed, L0 = _global_fit(u1, u2, family)
    lc = lc or LocalConstant(u1, u2, x, family, fixed)
    if h is None:
        h, cv_rows = select_bandwidth(lc, fractions, cv_exclude); src = f"leave-{2 * cv_exclude + 1}-out CV likelihood"
    else:
        cv_rows, src = None, "fixed"
    est, boundary, wsum = lc.fit(h)
    L1 = float(np.sum(lc._loglik_at(est)))
    lam = L1 - L0
    p, ref = p_asymptotic(max(lam, 0.0), h, lc.range)
    notes = []
    if lam < -1e-8:
        notes.append("lambda_n < 0 (numerical: local fit below the global MLE)")
    if boundary.mean() > 0.02:
        notes.append(f"{boundary.mean():.1%} of local estimates sit at the edge of the eta grid (parameter-space boundary)")
    if not ref["ok"]:
        notes.append(ref["note"])
    if family == "student":
        notes.append("Student-t: df held at the global MLE in both models (estimated, not part of the test)")
    return dict(family=family, n=len(x), lam=lam, L1=L1, L0=L0, h=float(h), h_source=src, x_range=lc.range,
                abs_X_over_h=lc.range / h, r_K=ref["r_K"], df=ref["df"], p_asymptotic=p if ref["ok"] else np.nan,
                boundary_fraction=float(boundary.mean()), eta_tilde=float(FAMILY_INFO[family]["link_fwd"](primary)),
                notes=notes, cv_table=cv_rows)


# ── calibrated null distributions ───────────────────────────────────────────
def _family_key(family):
    """Map the family name used here to the key of copula_families ("student" becomes "t").

    Parameters:
    family (str): Family name.

    Returns:
    str: Key for copula_families.make_bicop_signed.
    """
    return "t" if family == "student" else family


def simulate_null_pairs(family, primary, fixed, n, rng):
    """Draw n iid pairs from the fitted constant-copula null model.

    Parameters:
    family (str): Family key.
    primary (float): Signed primary parameter under H0.
    fixed (list or None): Fixed extra parameters (the Student-t degrees of freedom).
    n (int): Number of pairs.
    rng (np.random.Generator): Source of the seed of the sampler.

    Returns:
    np.ndarray: (n, 2) pairs.
    """
    from copula_families import make_bicop_signed, sample_bicop
    params = [primary] + list(fixed or [])
    cop = make_bicop_signed(_family_key(family), params)
    return np.asarray(sample_bicop(cop, n, seed=int(rng.integers(1, 2 ** 31 - 1))), float)


def null_parametric(u1, u2, x, family, B=199, h=None, reselect_h=True, cv_exclude=0, seed=0, fractions=PILOT_FRACTIONS,
                    reps=None):
    """Null distribution of lambda_n by parametric bootstrap under H0, with the covariate path X held fixed.

    Iid pairs are simulated from the fitted constant copula and lambda_n is recomputed. With reselect_h the bandwidth
    is re-selected by cross-validation in every replicate, so the data-driven bandwidth is part of the distribution
    of the statistic. The null is valid when the pairs are iid given X (checked by glrt_size_study).

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    x (np.ndarray): Covariate values, shape (n,).
    family (str): Family key.
    B (int): Number of bootstrap replicates. Default 199.
    h (float): Bandwidth used when reselect_h is False. Optional.
    reselect_h (bool): Re-select the bandwidth in every replicate. Default True.
    cv_exclude (int): Half-width of the window left out in the cross-validation. Default 0.
    seed (int): Seed. Default 0.
    fractions (np.ndarray): Candidate bandwidths as fractions of |X|. Default PILOT_FRACTIONS.
    reps (sequence of int): Replicate numbers to compute. If given, replicate r uses its own generator
        SeedSequence([seed, r]), so the value of a replicate does not depend on how the replicates are split over
        workers, and B is ignored. If None, one generator seeded with seed serves all B replicates in order.
        Optional.

    Returns:
    np.ndarray: Simulated values of lambda_n, one per replicate.
    """
    rng = np.random.default_rng(seed)
    primary, fixed, _ = _global_fit(u1, u2, family)
    out = []
    for r in (range(B) if reps is None else reps):
        if reps is not None:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(r)]))
        s = simulate_null_pairs(family, primary, fixed, len(x), rng)
        r = glrt_constant(s[:, 0], s[:, 1], x, family, h=None if reselect_h else h, cv_exclude=cv_exclude, fractions=fractions)
        out.append(r["lam"])
    return np.asarray(out)


def p_monte_carlo(lam_obs, lam_null):
    """Monte Carlo p-value (1 + #{lambda* >= lambda}) / (B + 1).

    Parameters:
    lam_obs (float): Observed statistic.
    lam_null (np.ndarray): Simulated null values, shape (B,).

    Returns:
    float: The p-value.
    """
    lam_null = np.asarray(lam_null, float)
    return float((1 + np.sum(lam_null >= lam_obs)) / (len(lam_null) + 1))


# ── size and power study (synthetic data) ───────────────────────────────────
def simulate_conditional(n, family, eta_fn, x, rng):
    """Simulate n pairs with calibration function eta_fn(x), iid given x, using the family's closed-form link.

    Parameters:
    n (int): Number of pairs.
    family (str): Family key (the Student-t degrees of freedom are fixed at 6).
    eta_fn (callable): Calibration function eta(x).
    x (np.ndarray): Covariate values, shape (n,).
    rng (np.random.Generator): Source of the seeds of the sampler.

    Returns:
    np.ndarray: (n, 2) pairs, clipped to [1e-6, 1 - 1e-6].
    """
    from copula_families import make_bicop_signed, sample_bicop
    g_inv = FAMILY_INFO[family]["link_inv"]
    out = np.empty((n, 2))
    for i in range(n):
        th = float(g_inv(eta_fn(x[i])))
        cop = make_bicop_signed(_family_key(family), [th, 6.0] if family == "student" else [th])
        out[i] = sample_bicop(cop, 1, seed=int(rng.integers(1, 2 ** 31 - 1)))[0]
    return np.clip(out, 1e-6, 1 - 1e-6)


def glrt_size_study(n=500, family="gaussian", R=200, alphas=(0.10, 0.05, 0.01), eta_fn=None, x_gen=None, seed=0, h=None):
    """Monte Carlo rejection rates of the GLRT with the asymptotic scaled chi-square reference.

    With a constant eta_fn this is the type I error (size), with a non-constant one the power. Replicates for which the
    reference does not apply (|X| / h < 5, p is NaN) are left out of the rates and reported in valid_fraction.

    Parameters:
    n (int): Sample size. Default 500.
    family (str): Family key. Default "gaussian".
    R (int): Number of replicates. Default 200.
    alphas (tuple): Significance levels. Default (0.10, 0.05, 0.01).
    eta_fn (callable): Calibration function eta(x). Default constant 0.5.
    x_gen (callable): x_gen(rng, n) draws the covariate path. Default iid U(0, 1).
    seed (int): Seed. Default 0.
    h (float): Fixed bandwidth on the scale of the covariate. If None, cross-validated. Optional.

    Returns:
    dict: Keys n, R, family, rates (alpha -> rejection rate), valid_fraction, mean_lambda, p_values and lam.
    """
    rng = np.random.default_rng(seed)
    eta_fn = eta_fn or (lambda x: 0.5)
    x_gen = x_gen or (lambda rng, n: rng.uniform(0, 1, n))
    lam, p = [], []
    for _ in range(R):
        x = x_gen(rng, n)
        s = simulate_conditional(n, family, eta_fn, x, rng)
        r = glrt_constant(s[:, 0], s[:, 1], x, family, h=h)
        lam.append(r["lam"]); p.append(r["p_asymptotic"])
    p = np.asarray(p, float)
    valid = np.isfinite(p)                              # NaN when |X|/h < 5: not counted as a non-rejection
    rates = {a: (float(np.mean(p[valid] <= a)) if valid.any() else float("nan")) for a in alphas}
    return dict(n=n, R=R, family=family, rates=rates, valid_fraction=float(valid.mean()), mean_lambda=float(np.mean(lam)),
                p_values=p, lam=np.asarray(lam))


# ── real-data wrapper ───────────────────────────────────────────────────────
CAVEATS = ("estimated PITs; serial dependence; data-driven bandwidth; folds overlap in time; "
           "size pilot includes iid pairs and an AR(1)-pair stress test; the iid parametric null does not correct serial dependence")


def _par_chunk(args):
    """Run null_parametric on one chunk of replicates (worker function of _parallel).

    Parameters:
    args (tuple): (u1, u2, x, family, reps, seed).

    Returns:
    np.ndarray: Simulated values of lambda_n for the replicates in reps.
    """
    u1, u2, x, family, reps, seed = args
    return null_parametric(u1, u2, x, family, seed=seed, reps=reps)


def _parallel(u1, u2, x, family, B, seed, workers):
    """Split the B bootstrap replicates of null_parametric over worker processes and concatenate the results.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    x (np.ndarray): Covariate values, shape (n,).
    family (str): Family key.
    B (int): Total number of replicates.
    seed (int): Base seed. Replicate r is drawn from SeedSequence([seed, r]), whatever the number of workers.
    workers (int): Maximum number of worker processes.

    Returns:
    np.ndarray: (B,) simulated values of lambda_n.
    """
    workers = resolve_pool_workers(n_tasks=B, max_workers=workers,
                                   worker_gib=float(os.getenv('THESIS_GLRT_WORKER_GIB', '2.0')))
    chunks = [c for c in np.array_split(np.arange(B), workers) if len(c)]
    jobs = [(u1, u2, x, family, [int(r) for r in c], seed) for c in chunks]
    if len(jobs) == 1:
        return np.asarray(_par_chunk(jobs[0]))
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        return np.concatenate(list(ex.map(_par_chunk, jobs)))


def holm(p):
    """Holm step-down adjustment of p-values (Holm, 1979).

    Parameters:
    p (array-like): Raw p-values.

    Returns:
    np.ndarray: Adjusted p-values, in the order of p.
    """
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def glrt_one(u1, u2, w, family, B=199, seed=0, workers=None, alpha=0.05):
    """Run the GLRT for one family: the statistic, the asymptotic reference and the parametric-bootstrap p-value.

    The verdict reads "covariate effect supported by the calibrated null" if the bootstrap p-value is at most alpha.
    Otherwise it reads "formal significance not established (not evidence of no effect)".

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    w (np.ndarray): Covariate values, shape (n,).
    family (str): Family key.
    B (int): Bootstrap replicates. With B = 0 the bootstrap is skipped. Default 199.
    seed (int): Seed. Default 0.
    workers (int): Optional upper limit; default resolves CPU, task count and RAM.
    alpha (float): Significance level of the verdict. Default 0.05.

    Returns:
    dict: Keys family, n, lam, h, abs_X_over_h, p_asymptotic_reference, boundary_fraction, notes, p_param_boot,
        B and verdict.
    """
    obs = glrt_constant(u1, u2, w, family)
    out = dict(family=family, n=obs["n"], lam=obs["lam"], h=obs["h"], abs_X_over_h=obs["abs_X_over_h"],
               p_asymptotic_reference=obs["p_asymptotic"], boundary_fraction=obs["boundary_fraction"],
               notes="; ".join(obs["notes"]))
    if B and B > 0:
        lp = _parallel(u1, u2, w, family, B, seed, workers)
        out.update(p_param_boot=p_monte_carlo(obs["lam"], lp), B=len(lp))
    else:
        out.update(p_param_boot=np.nan, B=0)
    ok = np.isfinite(out["p_param_boot"])
    out["verdict"] = ("covariate effect supported by the calibrated null" if ok and out["p_param_boot"] <= alpha
                      else "formal significance not established (not evidence of no effect)")
    return out


def run_glrt_real(u1, u2, w, primary_family, exploratory_families=(), B=199, seed=0, workers=None, alpha=0.05):
    """GLRT on real data for a primary family and exploratory families.

    The first row is the primary test. The other rows are exploratory, with a Holm-adjusted bootstrap p-value (column
    p_holm_exploratory).

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    w (np.ndarray): Covariate values, shape (n,).
    primary_family (str): Family of the primary test.
    exploratory_families (tuple): Families of the exploratory tests. Default ().
    B (int): Bootstrap replicates per family. Default 199.
    seed (int): Seed. Default 0.
    workers (int): Optional upper limit; default resolves CPU, task count and RAM.
    alpha (float): Significance level of the verdicts. Default 0.05.

    Returns:
    pd.DataFrame: One row per family, with the columns of glrt_one plus role and p_holm_exploratory. The caveats
        are in df.attrs["caveats"].
    """
    u1 = np.asarray(u1, float); u2 = np.asarray(u2, float); w = np.asarray(w, float)
    rows = [dict(role="primary", **glrt_one(u1, u2, w, primary_family, B, seed, workers, alpha))]
    for k, f in enumerate(exploratory_families):
        rows.append(dict(role="exploratory", **glrt_one(u1, u2, w, f, B, seed + 10 + k, workers, alpha)))
    df = pd.DataFrame(rows)
    ex = df.role == "exploratory"
    df["p_holm_exploratory"] = np.nan
    if ex.sum() and df.loc[ex, "p_param_boot"].notna().all():
        df.loc[ex, "p_holm_exploratory"] = holm(df.loc[ex, "p_param_boot"].values)
    df.attrs["caveats"] = CAVEATS
    return df


def aic_primary_family(u1, u2, families):
    """Pick the primary family of the GLRT by AIC of the unconditional parametric fit on the given (training) rows.

    AIC = 2k - 2 log L with k = 1 (Student-t: 2) and the better of the two rotation branches for the signed
    families (copula_families.fit_family). Only the rows passed in are used, so no held-out data enter the choice. A
    family that cannot be fitted is listed with an infinite AIC and an error message.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).
    families (list): Candidate family names.

    Returns:
    tuple: (best_family, table). table is a DataFrame of AIC by family, sorted ascending.
    """
    import copula_families as CF
    data = np.column_stack([np.asarray(u1, float), np.asarray(u2, float)])
    rows = []
    for fam in families:
        key = CF.resolve(fam).key
        try:
            cop = CF.fit_family(key, data)
            ll = float(cop.loglik(data)); k = CF.resolve(key).n_params
            rows.append(dict(family=fam, loglik=ll, k=k, aic=2 * k - 2 * ll))
        except Exception as e:                        # a family that cannot be fitted is listed, never silently dropped
            rows.append(dict(family=fam, loglik=np.nan, k=np.nan, aic=np.inf, error=str(e)[:80]))
    tab = pd.DataFrame(rows).sort_values('aic').reset_index(drop=True)
    if not np.isfinite(tab.aic.iloc[0]):
        raise RuntimeError("aic_primary_family: no family could be fitted")
    return tab.family.iloc[0], tab


# ── pilot simulation ────────────────────────────────────────────────────────
def gaussian_ar1_pairs(rng, n, rho, phi):
    """Gaussian-copula pairs with constant correlation rho and serial AR(1) persistence phi.

    Each latent coordinate follows a stationary AR(1), using correlated innovations whose
    contemporaneous correlation is rho. The marginal PIT preserves both the copula and the serial dependence.

    Parameters:
    rng (np.random.Generator): Random number generator.
    n (int): Number of time steps.
    rho (float): Contemporaneous correlation of the two coordinates.
    phi (float): AR(1) coefficient of each coordinate.

    Returns:
    np.ndarray: Pairs on uniform margins, shape (n, 2), clipped to [1e-6, 1 - 1e-6].
    """
    z = np.empty((n, 2))
    z[0] = rng.standard_normal(2)
    z[0, 1] = rho * z[0, 0] + np.sqrt(1 - rho ** 2) * z[0, 1]
    for t in range(1, n):
        eps = rng.standard_normal(2)
        eps[1] = rho * eps[0] + np.sqrt(1 - rho ** 2) * eps[1]
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * eps
    return np.clip(norm.cdf(z), 1e-6, 1 - 1e-6)


sys.path.insert(0, str(Path(__file__).resolve().parent))


def run(out=None, quick=False):
    """Pilot simulation of the GLRT: size, power, bandwidth and AR(1) dependence in both the covariate and pairs.

    1. Size with constant calibration, by family (n = 500, cross-validated h).
    2. Size by sample size (Gaussian).
    3. Size by fixed bandwidth (Gaussian, n = 500).
    4. Power against linear, quadratic and sinusoidal calibration functions (Gaussian, n = 500).
    5. Size under the constant Gaussian-copula null with AR(1) dependence in both the covariate and
       the pair time series (phi = 0.95, n = 300). This is a diagnostic, not a calibrated bootstrap.

    Parameters:
    out (str): Path of a JSON file for the results. Optional.
    quick (bool): Fewer replicates (a smoke test). Default False.

    Returns:
    dict: Results keyed by design.
    """
    R = 40 if quick else 200
    res = {}
    t0 = time.time()
    # 1. size (constant calibration), by family, n = 500, CV bandwidth
    for fam, eta in (("gaussian", 0.5), ("student", 0.5), ("clayton", 1.0), ("frank", 3.0)):
        r = glrt_size_study(n=500, family=fam, R=R, eta_fn=lambda x, e=eta: e, seed=1)
        res[f"size|{fam}|n=500|CV h"] = dict(rates=r["rates"], valid_fraction=r["valid_fraction"], mean_lambda=r["mean_lambda"], R=R)
        print(f"size {fam:9s} n=500 CV-h  rates {r['rates']}  valid {r['valid_fraction']:.2f}  mean lambda {r['mean_lambda']:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    # 2. size, sample size (gaussian)
    for n in (250, 1000):
        r = glrt_size_study(n=n, family="gaussian", R=R, eta_fn=lambda x: 0.5, seed=2)
        res[f"size|gaussian|n={n}|CV h"] = dict(rates=r["rates"], valid_fraction=r["valid_fraction"], mean_lambda=r["mean_lambda"], R=R)
        print(f"size gaussian  n={n} CV-h  rates {r['rates']}  valid {r['valid_fraction']:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    # 3. size, bandwidth sensitivity (fixed fractions of |X|, gaussian, n = 500)
    for frac in (0.15, 0.30, 0.50):
        r = glrt_size_study(n=500, family="gaussian", R=R, eta_fn=lambda x: 0.5, seed=3, h=frac)
        res[f"size|gaussian|n=500|h={frac}"] = dict(rates=r["rates"], valid_fraction=r["valid_fraction"], mean_lambda=r["mean_lambda"], R=R)
        print(f"size gaussian  n=500 h={frac}|X|  rates {r['rates']}  valid {r['valid_fraction']:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    # 4. power (gaussian, n = 500, CV h)
    alts = {"linear": lambda x: -0.5 + 1.5 * x, "quadratic": lambda x: 1.0 - 3.0 * (x - 0.5) ** 2 * 4 * 0.5,
            "sinusoidal": lambda x: 0.5 + 0.8 * np.sin(2 * np.pi * x)}
    for nm, f in alts.items():
        r = glrt_size_study(n=500, family="gaussian", R=max(R // 2, 20), eta_fn=f, seed=4)
        res[f"power|gaussian|n=500|{nm}"] = dict(rates=r["rates"], R=max(R // 2, 20))
        print(f"power {nm:10s} n=500  rejection rates {r['rates']}  valid {r['valid_fraction']:.2f}  [{time.time() - t0:.0f}s]", flush=True)
    # 5. serially dependent covariate and bivariate pairs: stress-test nominal size under the constant-copula null
    def ar1(rng, n, phi=0.95):
        """Stationary AR(1) series with unit variance.

        Parameters:
        rng (np.random.Generator): Random number generator.
        n (int): Length of the series.
        phi (float): AR(1) coefficient. Default 0.95.

        Returns:
        np.ndarray: The series, length n.
        """
        z = np.empty(n); z[0] = rng.standard_normal()
        for i in range(1, n):
            z[i] = phi * z[i - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal()
        return z
    Rb, B = (12, 19) if quick else (60, 49)
    rng = np.random.default_rng(5); rej = {"asymptotic": 0, "parametric bootstrap (X fixed)": 0}
    for _ in range(Rb):
        x = ar1(rng, 300)
        s = gaussian_ar1_pairs(rng, 300, rho=float(FAMILY_INFO["gaussian"]["link_inv"](0.5)), phi=0.95)
        r = glrt_constant(s[:, 0], s[:, 1], x, "gaussian")
        rej["asymptotic"] += (r["p_asymptotic"] <= 0.10)
        ln = null_parametric(s[:, 0], s[:, 1], x, "gaussian", B=B, seed=int(rng.integers(1e6)))
        rej["parametric bootstrap (X fixed)"] += (p_monte_carlo(r["lam"], ln) <= 0.10)
    res["size|gaussian|n=300|AR(1) x and pairs|alpha=0.10"] = {k: v / Rb for k, v in rej.items()} | {"R": Rb, "B": B}
    print("AR(1) covariate and pairs, alpha=0.10:", res["size|gaussian|n=300|AR(1) x and pairs|alpha=0.10"], f"[{time.time() - t0:.0f}s]", flush=True)
    if out:
        Path(out).write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None, quick=("--quick" in sys.argv))
