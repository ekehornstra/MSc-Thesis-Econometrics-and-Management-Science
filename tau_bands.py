"""Bands for the fold-averaged Kendall's tau(w) curves of the nonparametric estimators and the networks.

The Acar curves get their band from conditional_copula.fold_averaged_curve. This module gives the other models of the
Kendall's tau figure a band of the same kind where one can be computed:

* ECBC and Gijbels: the same shared stationary-bootstrap draws as the Acar bands (same seed and block length), each fold refits the
  estimator on the rows of the draw that lie in its training set, the fold curves of a replicate are averaged, and the 2.5% and
  97.5% percentiles are taken over the replicates.
* Networks: a bootstrap band would need every network to be retrained per replicate, so the band is the range over the training
  seeds of the fold-averaged curve. It measures training variability, not sampling uncertainty.
"""
import hashlib
import pickle
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from data_split import stationary_bootstrap_indices


def _tau_curve(X, U, kind, x_eval, K):
    """Kendall's tau(w) of one nonparametric fit on a grid of covariate values.

    Parameters:
    X (np.ndarray): Covariate values of the training sample, shape (n,).
    U (np.ndarray): Pseudo-observations of the training sample, shape (n, 2).
    kind (str): 'ecbc' or 'gijbels'.
    x_eval (np.ndarray): Grid of covariate values; values outside the range of X are clipped to it.
    K (int): Number of cells per margin of the conditional copula.

    Returns:
    np.ndarray: tau on x_eval; NaN everywhere if the fit fails.
    """
    import ecbc_copula as ECBC
    import gijbels_copula as GIJ
    try:
        if kind == "gijbels":
            g1, _ = GIJ.plugin_g(X, U[:, 0])
            g2, _ = GIJ.plugin_g(X, U[:, 1])
            Ut = GIJ.conditional_pits(X, U, (g1, g2))
            ok = np.all(np.isfinite(Ut), axis=1)
            h, _ = GIJ.plugin_h(X[ok], Ut[ok])
            fit = GIJ.GijbelsFit(X, U, h, (g1, g2), K)
        else:
            fit = ECBC.ECBCFit(X, U, K)
        lo, hi = float(X.min()), float(X.max())
        return np.array([GIJ.tau_from_masses(fit.masses(float(np.clip(x, lo, hi))), K) for x in x_eval])
    except Exception:
        return np.full(len(x_eval), np.nan)


def _replicate(X, U, idx, kind, x_eval, K):
    """tau(w) of the fit on one bootstrap sample of the training rows.

    Parameters:
    X (np.ndarray): Covariate values of the training rows.
    U (np.ndarray): Pseudo-observations of the training rows.
    idx (np.ndarray): Positions of the bootstrap sample among the training rows.
    kind (str): 'ecbc' or 'gijbels'.
    x_eval (np.ndarray): Grid of covariate values.
    K (int): Number of cells per margin.

    Returns:
    np.ndarray: tau on x_eval.
    """
    return _tau_curve(X[idx], U[idx], kind, x_eval, K)


def _digest(ARR, folds, kind, x_eval, ci_boot, seed):
    """Cache key: fitted data, settings and the code of this module and of the two estimators.

    Parameters:
    ARR (dict): Fold name -> fold arrays.
    folds (list): Fold names.
    kind (str): 'ecbc' or 'gijbels'.
    x_eval (np.ndarray): Grid of covariate values.
    ci_boot (int): Number of bootstrap replicates.
    seed (int): Seed of the bootstrap draws.

    Returns:
    str: Sixteen hexadecimal characters.
    """
    h = hashlib.sha256()
    for name in (__file__, "ecbc_copula.py", "gijbels_copula.py"):
        p = Path(name) if Path(name).exists() else Path(__file__).with_name(Path(name).name)
        h.update(p.read_bytes())
    h.update(repr((kind, ci_boot, seed, tuple(folds), np.round(x_eval, 10).tolist())).encode())
    for f in folds:
        for key in ("u1", "u2", "w", "train"):
            a = np.ascontiguousarray(np.asarray(ARR[f][key]))
            h.update(repr((f, key, a.shape, str(a.dtype))).encode())
            h.update(a.tobytes())
        h.update(repr(float(ARR[f]["b"])).encode())
    return h.hexdigest()[:16]


def nonparam_boot_band(ARR, folds, kind, wg, n_eval=15, ci_boot=200, n_jobs=1, seed=42, K=None, cache_dir="."):
    """95% bootstrap band of the fold-averaged tau(w) of ECBC or Gijbels, on the grid wg.

    Replicate r is one stationary bootstrap draw over the whole series, shared by all folds and identical to the draws of
    conditional_copula.fold_averaged_curve (same seed, same mean block length). Fold k refits on the rows of the draw that lie in
    its training set; the fold curves of the replicate are averaged. tau is computed on n_eval points and interpolated to wg. The
    result is cached in cache_dir after the computation, keyed by the data, the settings and the code.

    Parameters:
    ARR (dict): Fold name -> fold arrays with the keys u1, u2, w, train and b.
    folds (iterable): Fold names.
    kind (str): 'ecbc' or 'gijbels'.
    wg (np.ndarray): Grid on which the figure draws the curve; the band is interpolated to it.
    n_eval (int): Number of grid points at which the replicates are evaluated. Default 15.
    ci_boot (int): Bootstrap replicates. Default 200.
    n_jobs (int): Worker processes. Default 1.
    seed (int): Seed of the shared draws. Default 42.
    K (int or None): Cells per margin; None takes ces_score.DEFAULT_K.
    cache_dir (str): Folder of the cache file. Default the working directory.

    Returns:
    dict: lo and hi (np.ndarray on wg), x_eval, replicates (ci_boot x n_eval array of fold means) and n_failed (fits that failed).
    """
    import ces_score as CES
    K = CES.DEFAULT_K if K is None else int(K)
    folds = list(folds)
    wg = np.asarray(wg, float)
    x_eval = np.linspace(wg.min(), wg.max(), n_eval)
    cache = Path(cache_dir) / f"tau_bands_{kind}_{ci_boot}_{_digest(ARR, folds, kind, x_eval, ci_boot, seed)}.pkl"
    if cache.exists():
        res = pickle.load(open(cache, "rb"))
    else:
        n = {len(ARR[f]["w"]) for f in folds}
        if len(n) != 1:
            raise ValueError(f"the folds are over series of different lengths: {sorted(n)}")
        n = n.pop()
        rng = np.random.default_rng(seed)
        b = float(np.mean([ARR[f]["b"] for f in folds]))
        draws = [stationary_bootstrap_indices(n, b, rng) for _ in range(ci_boot)]
        tasks = []
        for f in folds:
            A = ARR[f]
            tr = np.asarray(A["train"], bool)
            pos = np.cumsum(tr) - 1
            X = np.asarray(A["w"], float)[tr]
            U = np.column_stack([np.asarray(A["u1"], float)[tr], np.asarray(A["u2"], float)[tr]])
            tasks += [(X, U, pos[d[tr[d]]]) for d in draws]
        out = Parallel(n_jobs=n_jobs)(delayed(_replicate)(X, U, idx, kind, x_eval, K) for X, U, idx in tasks)
        curves = np.stack(out).reshape(len(folds), ci_boot, n_eval)
        res = dict(replicates=np.nanmean(curves, axis=0), x_eval=x_eval, n_failed=int(np.isnan(curves).all(axis=2).sum()))
        pickle.dump(res, open(cache, "wb"))
    avg = res["replicates"]
    lo, hi = np.nanpercentile(avg, [2.5, 97.5], axis=0)
    return dict(lo=np.interp(wg, x_eval, lo), hi=np.interp(wg, x_eval, hi), x_eval=x_eval, replicates=avg, n_failed=res["n_failed"])


def seed_range_band(tau_seeds, tau, folds, model, wg):
    """Range over the training seeds of the fold-averaged tau(w) of one network.

    Parameters:
    tau_seeds (dict): Fold -> model -> {seed: tau curve on that fold's w grid} (the artifact key tau_curves_seeds).
    tau (dict): Fold -> {'w_grid': grid, model: seed-mean curve} (the artifact key tau_curves).
    folds (iterable): Fold names.
    model (str): Network name.
    wg (np.ndarray): Common grid of the figure.

    Returns:
    tuple: (lo, hi), the minimum and maximum over the seeds of the fold-averaged curve, on wg.
    """
    folds = list(folds)
    seeds = sorted(set.intersection(*[set(tau_seeds[f][model]) for f in folds]))
    per_seed = np.stack([np.mean([np.interp(wg, tau[f]["w_grid"], tau_seeds[f][model][s]) for f in folds], axis=0) for s in seeds])
    return per_seed.min(axis=0), per_seed.max(axis=0)
