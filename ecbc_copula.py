"""ecbc_copula.py

Empirical checkerboard Bernstein copula (ECBC) estimator of a conditional bivariate copula, after Lu and Ghosh (2024), used as a
nonparametric benchmark on the synthetic designs and on the real-data folds.

Estimator (covariate x, pseudo-observations (U1, U2), training sample (X_i, U1i, U2i), v = F_n(x) with F_n(x) = #{X_i <= x} / (n + 1)):
  1. conditional margins   F^_{j|x}(y) = d C_j(w, v) / dv at (w, v) = (F_nY(y), F_n(x)), with C_j the two-dimensional ECBC
                           of (Y_j, X) (Lu & Ghosh, 2024, eqs. 6-11); the covariate-adjusted points are U^_ij = F^_{j|X_i}(U_ji)
  2. three-dimensional ECBC C#(u1, u2, v) of (U^_1, U^_2, V) (the empirical checkerboard copula smoothed by Bernstein polynomials of
     degrees l1, l2 in the two copula arguments and m in the covariate)
  3. conditional copula    C#(1)(u1, u2 | v) = d C#(u1, u2, v) / dv (eq. 12), whose margins F1(u1 | v) and F2(u2 | v) (eqs. 14-15) are
                           exactly uniform only asymptotically; the copula is therefore C#(u1, u2 | v) = C#(1)(F1^{-1}(u1 | v),
                           F2^{-1}(u2 | v) | v) (eq. 16), which is a copula for every v.

The coefficient array of C#(1)(. | v) is a mixture, with weights that are nonnegative for every v, of the cell masses of the empirical
checkerboard copula, so C#(1)(. | v) is 2-increasing and F1, F2 are nondecreasing. The copula is represented by its K x K cell masses, as in
gijbels_copula, so both estimators are scored by the same code.

Implementation:
  * Degrees: the hierarchical shifted-Poisson prior of Lu and Ghosh (2024, Remark 1): l_j | alpha_j ~ Poisson(n^alpha_j) + 1, alpha_j ~
    Uniform(1/3, 2/3), j = 1, 2, and m | alpha ~ Poisson(n^alpha) + 2, alpha ~ Uniform(1/3, 2/3). The paper states the prior for the asymptotic
    theory (expectations of the error over the prior). The estimator here is the average of the conditional copulas over N_PRIOR_DRAWS
    draws of (l1, l2, m) from this prior, a Monte Carlo approximation of the prior average with stratified exponents and a fixed seed, which
    keeps the cost at that of N_PRIOR_DRAWS fits of the 3-D ECBC. The Monte Carlo error of the average is not zero with few draws.
  * The empirical checkerboard copula of n points with ordinal ranks R_ij spreads the mass 1/n of point i uniformly over the cell
    prod_j [(R_ij - 1) / n, R_ij / n]. Its coefficients are theta_{h1,h2,k} = C_n(h1 / l1, h2 / l2, k / m).
  * Ties among the covariate-adjusted points are broken by position (stable ordinal ranks).
  * A covariate value outside the training range is handled through v = F_n(x), which saturates at 1 / (n + 1) and n / (n + 1).
  * The Bernstein copula of Sancetta and Satchell (2004) and the composite Bernstein copulas of Yang et al. (2015) use the same
    polynomial basis; the ECBC differs in taking the coefficients from the checkerboard copula of the ranks.
  * Zero tail dependence: at every finite degree the Bernstein smoothing gives lower and upper tail dependence coefficients of zero, so the
    estimator can approximate but never reproduce the limit of a copula with tail dependence (Sancetta & Satchell, 2004).

Lu, Y., & Ghosh, S. (2024). Nonparametric conditional copula estimation based on smoothed checkerboard Bernstein sieves. Mathematics,
12(8), 1135.
Sancetta, A., & Satchell, S. (2004). The Bernstein copula and its applications to modeling and approximations of multivariate
distributions. Econometric Theory, 20(3), 535-562.
Yang, J., Chen, Z., Wang, F., & Wang, R. (2015). Composite Bernstein copulas. ASTIN Bulletin, 45(2), 445-475.

The two-dimensional building block (Bernstein polynomial of the empirical copula on the grid h / degree, for a degree that divides n) agrees
with OpenTURNS EmpiricalBernsteinCopula called with the normalised rank sample, to machine precision.
"""
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from threadpoolctl import threadpool_limits
from run_support import resolve_pool_workers

import ces_score as CES
import gijbels_copula as GJ

U_FINE = np.linspace(0.0, 1.0, 4001)       # grid on which the conditional margins are inverted
N_PRIOR_DRAWS = 5                 # number of degree triples drawn from the hierarchical prior and averaged
PRIOR_SEED = 0                    # fixed, so the estimator is reproducible
LABEL = "ECBC (Lu and Ghosh 2024)"


def sample_prior_degrees(n, n_draws=N_PRIOR_DRAWS, seed=PRIOR_SEED):
    """Draws of the degrees (l1, l2, m) from the hierarchical shifted-Poisson prior of Lu and Ghosh (2024, Remark 1).

    l_j | alpha_j ~ Poisson(n^alpha_j) + 1 with alpha_j ~ Uniform(1/3, 2/3), j = 1, 2, and m | alpha ~ Poisson(n^alpha) + 2 with alpha ~ Uniform(1/3, 2/3);
    alpha_1, alpha_2 and alpha are independent. The exponents are stratified (Latin hypercube): each of the three is drawn once from each of
    n_draws equal bins of (1/3, 2/3), with an independent random order of the bins, so that a few draws cover the prior evenly. The
    Poisson variates are then drawn given the exponents. The seed is fixed, so the result is reproducible and is the same for every call.

    Parameters:
    n (int): Sample size.
    n_draws (int): Number of draws.
    seed (int): Seed of the generator.

    Returns:
    list: n_draws tuples (l1, l2, m).
    """
    rng = np.random.default_rng(seed)
    alpha = np.column_stack([(rng.permutation(n_draws) + rng.uniform(size=n_draws)) / n_draws for _ in range(3)]) / 3.0 + 1.0 / 3.0
    lam = float(n) ** alpha
    l1 = rng.poisson(lam[:, 0]) + 1
    l2 = rng.poisson(lam[:, 1]) + 1
    m = rng.poisson(lam[:, 2]) + 2
    return [(int(a), int(b), int(c)) for a, b, c in zip(l1, l2, m)]


def bernstein_basis(deg, x):
    """Bernstein basis polynomials B_{h,deg}(x) = C(deg, h) x^h (1 - x)^(deg - h), h = 0, ..., deg.

    Parameters:
    deg (int): Degree.
    x (array-like): Evaluation points in [0, 1].

    Returns:
    np.ndarray: Matrix of shape (len(x), deg + 1).
    """
    x = np.atleast_1d(np.asarray(x, float))
    return binom.pmf(np.arange(deg + 1)[None, :], deg, x[:, None])


def _ranks(a):
    """Ordinal ranks 1..n of a vector (ties broken by position).

    Parameters:
    a (np.ndarray): Vector.

    Returns:
    np.ndarray: Integer ranks.
    """
    r = np.empty(len(a), int)
    r[np.argsort(a, kind="stable")] = np.arange(1, len(a) + 1)
    return r


def _checkerboard_theta_blocked(R, degrees, max_entries=2_000_000):
    """Coefficients theta = C_n(h_1 / d_1, ..., h_p / d_p) of the empirical checkerboard copula on the Bernstein grid.

    Accumulated over blocks of points to bound memory (the tensor product has n x prod(d + 1) entries).

    Parameters:
    R (np.ndarray): Ordinal ranks 1..n of the n points in each of p coordinates, shape (n, p).
    degrees (sequence of int): Degree d_s of each coordinate.
    max_entries (int): Largest number of entries of one block of the tensor product.

    Returns:
    np.ndarray: Array of shape (d_1 + 1, ..., d_p + 1).
    """
    n, p = R.shape
    shape = [d + 1 for d in degrees]
    tot = np.zeros(int(np.prod(shape)))
    block = max(1, int(max_entries // tot.size))
    for a in range(0, n, block):
        Rb = R[a:a + block]
        fac = [np.clip(np.arange(degrees[s] + 1)[None, :] * (n / degrees[s]) - (Rb[:, [s]] - 1), 0.0, 1.0) for s in range(p)]
        out = fac[0]
        for s in range(1, p):
            out = (out[:, :, None] * fac[s][:, None, :]).reshape(len(Rb), -1)
        tot += out.sum(0)
    return (tot / n).reshape(shape)


def conditional_margin_pits(X, Y, V, deg_w, deg_v):
    """Covariate-adjusted points U^_i = F^_{Y|X_i}(Y_i) from the two-dimensional ECBC of (Y, X) (Lu & Ghosh, 2024, eqs. 6-11).

    Parameters:
    X (np.ndarray): Covariate of the training sample.
    Y (np.ndarray): One margin of the training sample.
    V (np.ndarray): Covariate pseudo-observations F_n(X_i), in (0, 1).
    deg_w (int): Degree in the margin argument (g_j in the paper).
    deg_v (int): Degree in the covariate (m_j in the paper).

    Returns:
    np.ndarray: The conditional cdf of Y given X = X_i evaluated at Y_i.
    """
    n = len(Y)
    R = np.column_stack([_ranks(Y), _ranks(X)])
    theta = _checkerboard_theta_blocked(R, (deg_w, deg_v))                 # (g + 1, m + 1)
    lam = np.diff(theta, axis=1)                                           # lambda_{h,k}, (g + 1, m)
    W = R[:, 0] / (n + 1.0)
    eta = deg_v * (bernstein_basis(deg_v - 1, V) @ lam.T)                  # (n, g + 1): eta_h(v_i)
    return np.sum(bernstein_basis(deg_w, W) * eta, axis=1)


class _ECBCComponent:
    """ECBC with one set of degrees: the conditional copula C#(. | v) of Lu and Ghosh (2024) for a given (l1, l2, m).

    Parameters:
    X (np.ndarray): Covariate values of the training sample.
    U (np.ndarray): Pseudo-observations of the training sample, shape (n, 2).
    K (int): Number of cells per margin.
    degrees (tuple): (l1, l2, m) of the three-dimensional ECBC.
    degrees_margin (sequence): (g, m) of the two-dimensional ECBC of step 1, one pair per margin.
    """

    def __init__(self, X, U, K, degrees, degrees_margin):
        """Fit the component on the training sample.

        Parameters:
        X (np.ndarray): Covariate values of the training sample.
        U (np.ndarray): Pseudo-observations of the training sample, shape (n, 2).
        K (int): Number of cells per margin.
        degrees (tuple): (l1, l2, m) of the three-dimensional ECBC.
        degrees_margin (sequence): (g, m) of the two-dimensional ECBC of step 1, one pair per margin.
        """
        self.X, self.K, self.n = X, int(K), len(X)
        self.l1, self.l2, self.m = degrees
        self.xs = np.sort(X)
        V = _ranks(X) / (self.n + 1.0)
        Uh = np.column_stack([conditional_margin_pits(X, U[:, j], V, *degrees_margin[j]) for j in range(2)])
        R = np.column_stack([_ranks(Uh[:, 0]), _ranks(Uh[:, 1]), _ranks(X)])
        theta = _checkerboard_theta_blocked(R, (self.l1, self.l2, self.m))
        self.gamma = np.diff(theta, axis=2)                                # (l1 + 1, l2 + 1, m)
        self.margin_range = 0.0

    def v_of(self, x0):
        """Covariate pseudo-observation F_n(x0) = #{X_i <= x0} / (n + 1), clipped to [1 / (n + 1), n / (n + 1)].

        Parameters:
        x0 (float): Covariate value.

        Returns:
        float: The pseudo-observation; a value outside the training range is evaluated at the nearest interior quantile.
        """
        v = np.searchsorted(self.xs, x0, side="right") / (self.n + 1.0)
        return float(np.clip(v, 1.0 / (self.n + 1.0), self.n / (self.n + 1.0)))

    def eta(self, x0):
        """Coefficients eta_{h1,h2|v} of C#(1)(. | v) at v = F_n(x0) (eq. 20).

        Parameters:
        x0 (float): Covariate value.

        Returns:
        np.ndarray: Array of shape (l1 + 1, l2 + 1).
        """
        pm = bernstein_basis(self.m - 1, [self.v_of(x0)])[0]
        return self.m * (self.gamma @ pm)

    def cdf_grid(self, x0):
        """Cdf of the genuine conditional copula (eq. 16) on the grid edges i / K.

        Parameters:
        x0 (float): Covariate value.

        Returns:
        np.ndarray: Array of shape (K + 1, K + 1).
        """
        eta = self.eta(x0)
        Bf1, Bf2 = bernstein_basis(self.l1, U_FINE), bernstein_basis(self.l2, U_FINE)
        F1 = np.maximum.accumulate(Bf1 @ eta[:, -1])                       # F1(u | v) = C#(1)(u, 1 | v)
        F2 = np.maximum.accumulate(Bf2 @ eta[-1, :])
        levels = np.arange(self.K + 1) / self.K
        x1 = np.interp(levels, F1 / F1[-1], U_FINE); x2 = np.interp(levels, F2 / F2[-1], U_FINE)
        self.margin_range = max(self.margin_range, abs(F1[-1] - 1.0), abs(F2[-1] - 1.0))
        return bernstein_basis(self.l1, x1) @ eta @ bernstein_basis(self.l2, x2).T


class ECBCFit:
    """Training-only fit of the conditional ECBC, averaged over draws from the prior of the degrees, evaluated through K x K cell masses.

    The estimator is the average of the conditional copulas of the components, one per draw (l1, l2, m) from the hierarchical shifted-Poisson
    prior of Lu and Ghosh (2024, Remark 1); see sample_prior_degrees. An average of copulas is a copula, so the uniform margins are kept.

    Parameters:
    X (array-like): Covariate values of the training sample.
    U (array-like): Pseudo-observations of the training sample, shape (n, 2).
    K (int): Number of cells per margin. Default ces_score.DEFAULT_K.
    n_draws (int): Number of prior draws. Default N_PRIOR_DRAWS.
    seed (int): Seed of the prior draws. Default PRIOR_SEED.
    """

    def __init__(self, X, U, K=CES.DEFAULT_K, n_draws=N_PRIOR_DRAWS, seed=PRIOR_SEED):
        """Fit one component per draw of the degrees from the prior.

        Parameters:
        X (array-like): Covariate values of the training sample.
        U (array-like): Pseudo-observations of the training sample, shape (n, 2).
        K (int): Number of cells per margin. Default ces_score.DEFAULT_K.
        n_draws (int): Number of prior draws. Default N_PRIOR_DRAWS.
        seed (int): Seed of the prior draws. Default PRIOR_SEED.
        """
        self.X = np.asarray(X, float).copy()
        U = np.asarray(U, float)
        self.K, self.n = int(K), len(self.X)
        self.parts = [_ECBCComponent(self.X, U, self.K, (l1, l2, m), ((l1, m), (l2, m)))
                      for l1, l2, m in sample_prior_degrees(self.n, n_draws, seed)]
        self.log = {}

    @property
    def degrees(self):
        """list: (l1, l2, m) of each component."""
        return [(c.l1, c.l2, c.m) for c in self.parts]

    def cdf_grid(self, x0):
        """Average over the components of the conditional copula cdf on the grid edges i / K.

        Parameters:
        x0 (float): Covariate value.

        Returns:
        np.ndarray: Array of shape (K + 1, K + 1).
        """
        cdf = np.mean([c.cdf_grid(x0) for c in self.parts], axis=0)
        self.log["margin_range"] = max(c.margin_range for c in self.parts)
        return cdf

    def masses(self, x0):
        """Cell masses of the estimated conditional copula at covariate value x0.

        Parameters:
        x0 (float): Covariate value.

        Returns:
        np.ndarray: The K * K masses, summing to one, in row-major order (u1 by u2). Masses below zero by rounding are clipped.
        """
        raw = GJ.masses_from_cdf_grid(self.cdf_grid(x0))
        self.log["clipped_mass"] = max(self.log.get("clipped_mass", 0.0), float(-raw[raw < 0].sum()))
        m = np.clip(raw, 0.0, None)
        return (m / m.sum()).ravel()

    def kendall_tau(self, x0):
        """Kendall's tau of the estimated conditional copula at covariate value x0 (from the cell masses).

        Parameters:
        x0 (float): Covariate value.

        Returns:
        float: Kendall's tau.
        """
        return GJ.tau_from_masses(self.masses(x0), self.K)


def ecbc_synth_job(bundle, K=CES.DEFAULT_K, G=CES.DEFAULT_GRID_POINTS, w_eval=None):
    """Fit the ECBC on one synthetic training bundle and score it on the bundle's out-of-sample draw.

    Nothing from the out-of-sample draw enters the fit. Same outputs and discretisation as gijbels_copula.gijbels_synth_job.

    Parameters:
    bundle (dict): Synthetic bundle with 'w', 'u1', 'u2' (training), 'ow', 'o1', 'o2' (out of sample), 'spec', and optionally 'rep'.
    K (int): Number of cells per margin. Default ces_score.DEFAULT_K.
    G (int): Number of covariate grid points for the pointwise scores. Default ces_score.DEFAULT_GRID_POINTS.
    w_eval (np.ndarray or None): Covariate grid of the tau and tail curves. Default synthetic_study.W_EVAL.

    Returns:
    dict: As gijbels_synth_job, with the degrees ('degrees') in place of the bandwidths.
    """
    import synthetic_study as SS        # imported here: synthetic_study pulls in fold_pipeline (and with it torch)
    t0 = time.time()
    w_eval = SS.W_EVAL if w_eval is None else w_eval
    X = np.asarray(bundle["w"], float); U = np.column_stack([bundle["u1"], bundle["u2"]]).astype(float)
    fit = ECBCFit(X, U, K)
    ow = np.asarray(bundle["ow"], float)
    o1, o2 = np.asarray(bundle["o1"], float), np.asarray(bundle["o2"], float)
    gw = np.linspace(float(ow.min()), float(ow.max()), G)
    outside = int(np.sum((ow < X.min()) | (ow > X.max())))
    raw, cal = GJ.pointwise_on_grid_both(fit.masses, ow, o1, o2, gw, K)
    ms_eval = [fit.masses(float(x)) for x in w_eval]
    tau = np.array([GJ.tau_from_masses(m, K) for m in ms_eval])
    chi = {c: np.array([GJ.corner_chi_from_masses(m, K)[c] for m in ms_eval]) for c in GJ.CORNERS}
    tau_plot = None
    if int(bundle.get("rep", 0)) == 0:
        tau_plot = np.array([fit.kendall_tau(float(x)) for x in SS.W_PLOT])
    mg = [CES.margin_error(fit.masses(float(x)), K) for x in np.linspace(X.min(), X.max(), 9)]
    return dict(ces=float(np.sum(raw["es"])), ces_cal=float(np.sum(cal["es"])), n_oos=len(o1),
                cvpe_disc=float(np.mean(raw["loss"])), loss_oos=raw["loss"], tau_curve=tau, chi=chi, tau_plot=tau_plot,
                h=np.nan, g=[np.nan], label=LABEL, info={}, degrees=fit.degrees,
                clipped_mass=float(fit.log.get("clipped_mass", 0.0)),
                margin_error_max=float(np.max(mg)), neff_min=float(fit.n), bad_pit=0,
                oos_outside_support=outside, runtime=time.time() - t0)


def ecbc_row(bundle, res, orc=None):
    """Summary row of one ECBC fit, with the columns of gijbels_copula.gijbels_row.

    Parameters:
    bundle (dict): Synthetic bundle.
    res (dict): Result of ecbc_synth_job.
    orc (dict or None): Result of gijbels_copula.oracle_discretised. Default None.

    Returns:
    dict: Summary row; 'h' and 'g' (bandwidths of the kernel estimator) are NaN.
    """
    return GJ.gijbels_row(bundle, res, orc)


def _fingerprint(bundle, K, G):
    """Hash of the code (this module, the Gijbels scoring module, ces_score, synthetic_study) and of the bundle data and settings.

    Parameters:
    bundle (dict): Synthetic bundle.
    K (int): Number of cells per margin.
    G (int): Number of covariate grid points.

    Returns:
    str: Hexadecimal SHA-256 digest.
    """
    import hashlib
    import synthetic_study as SS
    h = hashlib.sha256()
    for module in (sys.modules[__name__], GJ, CES, SS):
        h.update(GJ._code_digest(module).encode())
    for k in ("u1", "u2", "w", "o1", "o2", "ow"):
        h.update(np.ascontiguousarray(bundle[k]).tobytes())
    h.update(f"{K}|{G}".encode())
    return h.hexdigest()


def _one(args):
    """Worker of run_ecbc: fit and score one bundle in a single-threaded process.

    Parameters:
    args (tuple): (bundle, K, G).

    Returns:
    dict: {'status': 'ok', 'row': summary row, 'res': job result without the per-observation loss}, or
        {'status': 'failed', 'error': repr of the exception}.
    """
    bundle, K, G = args
    try:
        with threadpool_limits(limits=1):
            res = ecbc_synth_job(bundle, K=K, G=G)
            orc = GJ.oracle_discretised(bundle, K=K, G=G)
        return dict(status="ok", row=ecbc_row(bundle, res, orc), res={k: v for k, v in res.items() if k != "loss_oos"})
    except Exception as exc:
        return dict(status="failed", error=repr(exc))


def run_ecbc(bundles, cache_dir, max_workers=None, K=CES.DEFAULT_K, G=CES.DEFAULT_GRID_POINTS):
    """Run the ECBC for each bundle of one design, or load the cached result (one pickle per bundle, code and data fingerprint).

    Parameters:
    bundles (list): Synthetic bundles of one design; each needs the keys of ecbc_synth_job plus 'dgp' and 'rep'.
    cache_dir (str or pathlib.Path): Folder of the cached results; created if missing.
    max_workers (int): Optional upper limit; default uses CPU, task count and RAM budget.
    K (int): Number of cells per margin. Default ces_score.DEFAULT_K.
    G (int): Number of covariate grid points. Default ces_score.DEFAULT_GRID_POINTS.

    Returns:
    tuple: (pandas.DataFrame with one row per successful bundle, list of (key, error) for the failed ones). The frame's
        attrs['tau_plot'] is the fine-grid tau(w) curve of the first data draw.

    Raises:
    ValueError: If the bundles belong to more than one design.
    """
    if len({b["dgp"] for b in bundles}) > 1:
        raise ValueError("run_ecbc takes the bundles of one design per call.")
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for b in bundles:
        f = cache / f"ecbc_{b['dgp']}_rep{b['rep']}.pkl"; fp = _fingerprint(b, K, G)
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
    df.attrs["tau_plot"] = next((v["res"]["tau_plot"] for _, v in sorted(out.items())
                                 if v["status"] == "ok" and v["res"].get("tau_plot") is not None), None)
    return df, fails
