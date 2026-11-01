"""fold_pipeline.py

Four-fold x three-seed evaluation pipeline: blocked outer folds, admissibility gates, pooled CVPE and the Model
Confidence Set.

The module keeps the notebooks thin. Its main entry points are:

  * load_fold, fold_arrays   -- per-fold arrays written by notebook 00 (fails loudly if the artifact is stale)
  * parametric_fold_job      -- 7 unconditional and 14 regime fits, 49 low/high pairs by indexing, marginal KS
                                distances and joint-exceedance probabilities
  * acar_job                 -- one Acar family on one fold, with the bandwidth tuned on that fold only
  * neural_job               -- one network for (fold, seed): unconditional CODINE, regime CODINE (low and high),
                                3 C-CODINE and 6 C-CopFlow = 12 networks per (fold, seed)
  * build_fold_jobs, run_batch -- the 144-task grid with resume files and an explicit worker cap
  * gate_table, assemble_store -- gates, pool and loss tensor (n_steps, models, folds); the gate rules and their
                                constants live in gate_config.py
  * per_step_loss, mcs_table, joint_top_cvpe -- pooled CVPE and the MCS

The loss of a model at a test step is the CVPE integrand (u1 - E[U1 | U2])^2 + (u2 - E[U2 | U1])^2 (Acar et al.,
2011). Seeds are averaged on the loss, never on the predictions, and folds are averaged with np.nanmean over the
fold axis. Everything stored per job is small (losses, e1 and e2, joint-exceedance probabilities, diagnostics), so
the batch artifact stays light. The argument smoke=True of build_fold_jobs gives tiny settings for a quick test.

The candidate set has 76 members: (7+1) unconditional, (49+1) binary, (7+3+6) continuous and 2 nonparametric
(gate_config.N_CANDIDATES). gate_table lists all of them and asserts the count.

References:
Acar, E. F., Craiu, R. V., & Yao, F. (2011). Dependence calibration in conditional copulas: A nonparametric
    approach. Biometrics, 67(2), 445-453.
Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline flows. Advances in Neural
    Information Processing Systems, 32.
Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.
Letizia, N. A., Novello, N., & Tonello, A. M. (2025). Copula density neural estimation. IEEE Transactions on
    Neural Networks and Learning Systems, 36(10), 19452-19459.
Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. arXiv:1711.05101.
Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked autoregressive flow for density estimation.
    Advances in Neural Information Processing Systems, 30.
Patton, A., Politis, D. N., & White, H. (2009). Correction to "Automatic block-length selection for the dependent
    bootstrap" by D. Politis and H. White. Econometric Reviews, 28(4), 372-375.
Politis, D. N., & White, H. (2004). Automatic block-length selection for the dependent bootstrap. Econometric
    Reviews, 23(1), 53-70.
"""
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import ecbc_copula as ECBC
import gate_config as GC
import gijbels_copula as GIJ
from copula_families import (DEFAULT_FAMILIES, FAMILY_DISPLAY, FAMILY_KEYS, QUAD_N, branch_of, get_tail_deps, interp_theta_col,
                             safe_make_bicop)
from run_support import (ARTIFACT_DIR, SHARD, get_logger, load_artifact, resolve_max_workers, resolve_pool_workers,
                         resolve_memory_workers)


class _JobAccess:
    """Read-only dict-style access to a job dataclass.

    Supports job["x"], job.get("x", default) and "x" in job. A field that is None counts as missing, so get returns the
    default for it and the membership test is False.
    """

    def __getitem__(self, k):
        """Field access as job[k].

        Parameters:
        k (str): Field name.

        Returns:
        object: The value of the field.

        Raises:
        KeyError: If the job has no such field.
        """
        try:
            return getattr(self, k)
        except AttributeError:
            raise KeyError(k) from None

    def get(self, k, default=None):
        """Field access as job.get(k, default).

        Parameters:
        k (str): Field name.
        default (object): Value returned if the field is missing or None. Default None.

        Returns:
        object: The value of the field, or default.
        """
        v = getattr(self, k, None)
        return default if v is None else v

    def __contains__(self, k):
        """Membership test as k in job.

        Parameters:
        k (str): Field name.

        Returns:
        bool: True if the field exists and is not None.
        """
        return getattr(self, k, None) is not None


@dataclass(eq=False)
class FoldJob(_JobAccess):
    """One unit of work on one blocked fold.

    The field kind selects what runs: 'param' (parametric fits), 'acar' (one Acar family), a nonparametric kind ('gijbels' or
    'ecbc'), or a neural kind ('codine_uncond', 'codine_regime', 'ccodine', 'ccopflow'). The weight-decay sensitivity jobs use
    the kinds 'ccodine' and 'ccopflow'.

    Parameters:
    fold (str): Fold name, one of FOLDS.
    arrays (dict): Fold arrays from fold_arrays.
    kind (str): Job kind, see above.
    model (str): Model name (neural kinds).
    seed (int): Network seed. Default 0.
    wd (float): Weight decay of the conditional networks. Default None.
    regime (str): 'low' or 'high' (codine_regime only). Default None.
    hyper (dict): Overrides of default_hyper (for the nonparametric kinds: 'K' and 'G', the number of cells per margin and of
        covariate grid points).
    keep_net (bool): Keep the fitted network in the result. Default False.
    pit (str): 'kernel' (pseudo-observations from the kernel-weighted PIT) or 'uncond' (unconditional PIT), for the
        parametric jobs. Default 'kernel'.
    families (tuple): Parametric families to fit; default copula_families.DEFAULT_FAMILIES.
    family (str): Acar family key (acar jobs).
    h (float): Acar bandwidth (acar jobs).
    n_eval (int): Number of evaluation points of the Acar calibration curve. Default gate_config.ACAR_N_EVAL.
    """
    fold: str
    arrays: dict
    kind: str = ""
    model: str = ""
    seed: int = 0
    wd: float = None
    regime: str = None
    hyper: dict = field(default_factory=dict)
    keep_net: bool = False
    pit: str = "kernel"
    families: tuple = None
    family: str = None
    h: float = None
    n_eval: int = GC.ACAR_N_EVAL


FOLDS = ("A1", "A2", "B1", "B2")
SEEDS = GC.SEEDS
Q_GRID = GC.Q_GRID                          # upper-tail thresholds for chi(q) and the joint-exceedance probability
assert len(FOLDS) == GC.N_FOLDS
N_BATCH = 32                                # batch size of the final training of the conditional networks (n_train <= 3000)

CCODINE_VARIANTS = {
    "C-CODINE (w only)": "w_only",
    "C-CODINE (spline basis)": "spline_basis",
    "C-CODINE (FiLM)": "film",
}
CCOPFLOW_VARIANTS = {
    "C-CopFlow (w only, affine)": ("w_only", "affine"),
    "C-CopFlow (w only, spline coupling)": ("w_only", "spline"),
    "C-CopFlow (spline basis, affine)": ("spline_basis", "affine"),
    "C-CopFlow (spline basis, spline coupling)": ("spline_basis", "spline"),
    "C-CopFlow (FiLM, affine)": ("film", "affine"),
    "C-CopFlow (FiLM, spline coupling)": ("film", "spline"),
}
# 12 networks per (fold, seed): the 9 conditional ones below, the unconditional CODINE and the two regime CODINE networks
NEURAL_MODELS = list(CCODINE_VARIANTS) + list(CCOPFLOW_VARIANTS)
GIJBELS_MODEL = "Nonparametric: Gijbels et al. (2011)"
ECBC_MODEL = "Nonparametric: ECBC (Lu and Ghosh 2024)"
NONPARAMETRIC_MODELS = [GIJBELS_MODEL, ECBC_MODEL]
NONPARAMETRIC_KIND = {GIJBELS_MODEL: "gijbels", ECBC_MODEL: "ecbc"}
UNCOND_CODINE = "Unconditional: CODINE"
SPLIT_CODINE = "Divergence split: CODINE"

REQUIRED_FOLD_KEYS = ("u1", "u2", "u1_uncond", "u2_uncond", "x_std", "train_mask", "test_mask",
                      "high_divergence_mask", "low_divergence_mask", "b", "purge_final", "val_purge", "kernel_h")


# ── loading of the fold arrays ─────────────────────────────────────────────
def load_fold(name):
    """Load one fold's artifact from notebook 00 and fail loudly when it is stale.

    A KeyError such as 'test_mask' means the artifact was written by an older notebook 00 (single split). Re-run the
    last two cells of 00_preprocessing.ipynb (the preprocess_fold loop and the save_artifact("preprocessing_fold_...")
    cell) before any fold notebook.

    Parameters:
    name (str): Fold name, one of FOLDS.

    Returns:
    dict: The artifact contents, containing at least REQUIRED_FOLD_KEYS.

    Raises:
    FileNotFoundError: If the artifact does not exist.
    RuntimeError: If keys are missing.
    """
    a = load_artifact(f"preprocessing_fold_{name}")
    if not a:
        raise FileNotFoundError(
            f"artifacts/preprocessing_fold_{name}_* not found. Run the per-fold cells at the end "
            f"of 00_preprocessing.ipynb first.")
    missing = [k for k in REQUIRED_FOLD_KEYS if k not in a]
    if missing:
        raise RuntimeError(
            f"preprocessing_fold_{name} is stale (missing keys {missing}). Re-run the last two "
            f"cells of 00_preprocessing.ipynb to rewrite the per-fold artifacts.")
    return a


def fold_arrays(a):
    """Plain arrays used by the jobs.

    Parameters:
    a (dict): Fold artifact from load_fold.

    Returns:
    dict: u1, u2 (float64 pseudo-observations), u1_un, u2_un (unconditional PIT), w (float32 covariate), train, test,
        high (bool masks), and the integers b, purge_final and val_purge.
    """
    return dict(
        u1=np.asarray(a["u1"], float), u2=np.asarray(a["u2"], float),
        u1_un=np.asarray(a["u1_uncond"], float), u2_un=np.asarray(a["u2_uncond"], float),
        w=np.asarray(a["x_std"], np.float32),
        train=np.asarray(a["train_mask"], bool), test=np.asarray(a["test_mask"], bool),
        high=np.asarray(a["high_divergence_mask"], bool),
        b=int(a["b"]), purge_final=int(a["purge_final"]), val_purge=int(a["val_purge"]),
    )


def load_all_folds(folds=FOLDS):
    """Load the arrays of several folds.

    Parameters:
    folds (sequence of str): Fold names. Default FOLDS.

    Returns:
    dict: {fold: fold_arrays(load_fold(fold))}.
    """
    return {f: fold_arrays(load_fold(f)) for f in folds}


# ── joint-exceedance helpers ───────────────────────────────────────────────
@lru_cache(maxsize=32)
def _exceedance_midpoint_grid(q, n_cells):
    """Cached midpoint grid of the square [q, 1]^2.

    Parameters:
    q (float): Lower threshold, in [0, 1).
    n_cells (int): Number of cells per dimension.

    Returns:
    np.ndarray: (n_cells ** 2, 2) float32 array of cell midpoints, read-only.

    Raises:
    ValueError: If q is outside [0, 1) or n_cells is not positive.
    """
    if not (0.0 <= q < 1.0) or n_cells < 1:
        raise ValueError("q must be in [0,1) and n_cells must be positive")
    mid = q + (1.0 - q) * (np.arange(n_cells) + 0.5) / n_cells
    g1, g2 = np.meshgrid(mid, mid, indexing="ij")
    pts = np.column_stack([g1.ravel(), g2.ravel()]).astype(np.float32)
    pts.setflags(write=False)
    return pts


def exceed_prob_density(density_fn, q, w_value=None, n_cells=100):
    """Joint-exceedance probability P(U1 > q, U2 > q) as the integral of a copula density over [q, 1]^2.

    The integral is a midpoint rule on n_cells x n_cells cells. Negative density values are set to zero.

    Parameters:
    density_fn (callable): density_fn(pts, w) -> density at the points, for example
        lambda pts, w: eval_conditional_density(net, pts, w). For an unconditional density use w_value=None.
    q (float): Threshold, in [0, 1).
    w_value (float): Covariate value passed to density_fn. Default None.
    n_cells (int): Cells per dimension. Default 100.

    Returns:
    float: The probability.
    """
    pts = _exceedance_midpoint_grid(float(q), int(n_cells))
    dens = np.asarray(density_fn(pts, w_value), float)
    return float(np.sum(np.maximum(dens, 0.0)) * ((1.0 - q) / n_cells) ** 2)


def exceed_step_from_grid(w_step, w_grid, p_grid):
    """Interpolate a probability curve p(w) given on a grid at the covariate value of each step.

    Values outside the grid are clamped to the end points (np.interp).

    Parameters:
    w_step (array-like): Covariate value of each step.
    w_grid (array-like): Increasing grid of covariate values.
    p_grid (array-like): Probability at each grid value.

    Returns:
    np.ndarray: Probability at each step.
    """
    return np.interp(np.asarray(w_step, float), np.asarray(w_grid, float), np.asarray(p_grid, float))


def _loss_from_pred(u1, u2, e1, e2):
    """CVPE integrand of a step: (u1 - e1)^2 + (u2 - e2)^2.

    Parameters:
    u1 (array-like): Observed first pseudo-observation.
    u2 (array-like): Observed second pseudo-observation.
    e1 (array-like): Predicted E[U1 | U2].
    e2 (array-like): Predicted E[U2 | U1].

    Returns:
    np.ndarray: Per-step loss.
    """
    return (np.asarray(u1) - np.asarray(e1)) ** 2 + (np.asarray(u2) - np.asarray(e2)) ** 2


def _nan_full(n):
    """Array of n NaN values.

    Parameters:
    n (int): Length.

    Returns:
    np.ndarray: NaN array of length n.
    """
    return np.full(n, np.nan)


# ── parametric fits: one job per fold and PIT variant ──────────────────────
def _strip_rows(rows):
    """Drop pyvinecopulib objects from fitted-model rows, keeping the fitted parameters and tail coefficients.

    The copula objects are not safely picklable. Each row keeps its parameters, rotation, branch and the asymptotic tail
    coefficients (lambda_L, lambda_U from get_tail_deps; NaN if they cannot be computed).

    Parameters:
    rows (list): Rows of fit_parametric_fold (dicts with 'cop', 'fam' and 'name').

    Returns:
    list: One dict per row, without 'cop' and 'fam', with the added keys params, rotation, branch, lambda_L, lambda_U.
    """
    out = []
    for r in rows:
        d = {k: v for k, v in r.items() if k not in ("cop", "fam")}
        d["params"] = np.asarray(r["cop"].parameters, float).ravel().tolist()
        d["rotation"] = int(r["cop"].rotation)
        d["branch"] = branch_of(r["cop"])                 # own / turned rotation (signed families), "-" otherwise
        try:
            d["lambda_L"], d["lambda_U"] = get_tail_deps(r["cop"], r["name"])
        except Exception:
            d["lambda_L"] = d["lambda_U"] = np.nan
        out.append(d)
    return out


def joint_exceed_prob_cop(cop, q):
    """Joint-exceedance probability P(U1 > q, U2 > q) of a fitted copula.

    By inclusion-exclusion, P(U1 > q, U2 > q) = 1 - 2q + C(q, q).

    Parameters:
    cop (pyvinecopulib.Bicop): Fitted bivariate copula.
    q (float): Threshold.

    Returns:
    float: The probability, clipped to [0, 1].
    """
    c = float(np.asarray(cop.cdf(np.array([[q, q]])))[0])
    return float(np.clip(1 - 2 * q + c, 0.0, 1.0))


def joint_exceed_probs_parametric(fp, high_mask, test, q):
    """Per-step joint-exceedance probabilities of all parametric models of one fold.

    Parameters:
    fp (dict): Output of fold_models.fit_parametric_fold.
    high_mask (np.ndarray): Boolean mask of the high-divergence steps.
    test (np.ndarray): Boolean mask of the test steps.
    q (float): Threshold.

    Returns:
    dict: {model name: (n,) array}, NaN outside the test steps. Names are 'Uncond:<Fam>' and 'Binary:<Low>|<High>'.
    """
    n = len(high_mask); high = np.asarray(high_mask, bool); test = np.asarray(test, bool)
    out = {}
    for r in fp['rows_uncond']:
        p = np.full(n, np.nan); p[test] = joint_exceed_prob_cop(r['cop'], q)
        out[f"Uncond:{r['name']}"] = p
    lo = {r['name']: joint_exceed_prob_cop(r['cop'], q) for r in fp['rows_regime']['low']}
    hi = {r['name']: joint_exceed_prob_cop(r['cop'], q) for r in fp['rows_regime']['high']}
    for fl, pl in lo.items():
        for fh, ph in hi.items():
            p = np.full(n, np.nan)
            p[test & ~high] = pl; p[test & high] = ph
            out[f"Binary:{fl}|{fh}"] = p
    return out


def parametric_fold_job(job):
    """Fit all parametric candidates on one fold.

    Runs 7 unconditional and 14 regime fits, forms the 49 low/high pairs by indexing, and computes the marginal KS
    distances and the joint-exceedance probabilities.

    Parameters:
    job (FoldJob or dict): Keys fold, arrays (from fold_arrays), pit ('kernel' or 'uncond') and families (optional).

    Returns:
    dict: fold, pit, loss {model: (n,)}, p_step {q: {model: (n,)}}, aic_uncond, aic_regime, aic_pair, selected_pair,
        ks {('uncond' | 'low' | 'high', family): D_max}, rows_uncond, rows_regime and runtime. Model names are
        'Uncond:<Fam>' and 'Binary:<FamLow>|<FamHigh>', as in fold_models.
    """
    from fold_models import fit_parametric_fold

    t0 = time.time()
    A = job["arrays"]
    pit = job.get("pit", "kernel")
    u1, u2 = (A["u1"], A["u2"]) if pit == "kernel" else (A["u1_un"], A["u2_un"])
    families = job.get("families") or DEFAULT_FAMILIES
    fp = fit_parametric_fold(u1, u2, A["train"], A["test"], A["high"], families=families)

    p_step = {q: joint_exceed_probs_parametric(fp, A["high"], A["test"], q) for q in Q_GRID}

    ks = {}
    for scope, mask in (("uncond", A["test"]), ("low", A["test"] & ~A["high"]), ("high", A["test"] & A["high"])):
        if not np.any(mask):
            continue
        for item in (fp["rows_uncond"] if scope == "uncond" else fp["rows_regime"][scope]):
            cop = item["cop"]
            ks[(scope, item["name"])] = _cop_ks(cop, seed=107)
    return dict(fold=job["fold"], pit=pit, loss=fp["loss"], p_step=p_step,
                aic_uncond=fp["aic_uncond"], aic_regime=fp["aic_regime"], aic_pair=fp["aic_pair"],
                selected_pair=fp["selected_pair"], ks=ks,
                rows_uncond=_strip_rows(fp["rows_uncond"]),
                rows_regime={k: _strip_rows(v) for k, v in fp["rows_regime"].items()},
                runtime=time.time() - t0)


# ── Acar (continuous local-likelihood) per fold ────────────────────────────
def _cop_ks(cop, seed=0):
    """Largest one-sample KS distance of the two margins of a simulated copula.

    Parameters:
    cop (pyvinecopulib.Bicop): Copula to simulate from.
    seed (int): Seed of the simulation. Default 0.

    Returns:
    float: max over the two margins of the KS distance to the uniform distribution (gate_config.KS_N_SAMPLES draws), or
        NaN if the simulation fails or is not finite.
    """
    from scipy.stats import kstest
    draws = np.asarray(cop.simulate(GC.KS_N_SAMPLES, seeds=[int(seed), int(seed)+1]))
    if draws.ndim != 2 or draws.shape[1] != 2 or not np.isfinite(draws).all():
        return np.nan
    return max(float(kstest(draws[:, j], "uniform").statistic) for j in range(2))


def _acar_cop_at(fam, curve, fixed, wv):
    """Copula of an Acar family, with its rotation, at one covariate value.

    The parameter is interpolated from the calibration curve in the excess scale. For the Student-t family the degrees of
    freedom are the fixed training value, clipped to [2.001, 49].

    Parameters:
    fam (str): Acar family key.
    curve (pd.DataFrame): Calibration curve with columns 'x' and 'theta'.
    fixed (sequence): Fixed (not interpolated) parameters, for example the Student-t degrees of freedom.
    wv (float): Covariate value.

    Returns:
    pyvinecopulib.Bicop: The copula at wv.
    """
    theta = float(interp_theta_col(fam, wv, curve["x"].values, curve["theta"].values))   # signed, interpolated in the excess scale
    if fam == "student":
        return safe_make_bicop(fam, [theta, float(np.clip(fixed[0], 2.001, 49.0))])
    return safe_make_bicop(fam, [theta])


def acar_job(job):
    """One Acar family on one fold.

    Estimates the calibration curve on the fold's training rows with the bandwidth tuned on this fold only, then
    computes the per-step loss on the test rows, the per-step joint-exceedance probability and the marginal KS distance
    (largest over the training-range percentiles of the covariate).

    Parameters:
    job (FoldJob or dict): Keys fold, arrays, family (key of conditional_copula.FAMILY_INFO, one of ACAR_FAMILIES), h
        and n_eval.

    Returns:
    dict: fold, family, h, loss (n,), p_step {q: (n,)}, ks, curve, fixed and runtime. Joint-exceedance probabilities that
        cannot be computed are NaN and counted in a warning.
    """
    from conditional_copula import estimate_calibration_curve, fit_global
    from fold_models import make_acar_predictor

    t0 = time.time()
    A, fam, h = job["arrays"], job["family"], float(job["h"])
    tr, te = A["train"], A["test"]
    u1, u2, w = A["u1"], A["u2"], A["w"].astype(float)
    curve = estimate_calibration_curve(u1[tr], u2[tr], w[tr], family=fam, h=h,
                                       n_eval=int(job.get("n_eval", GC.ACAR_N_EVAL)), ci_boot=0)
    _, fixed, _ = fit_global(u1[tr], u2[tr], fam)
    pred = make_acar_predictor(fam, {fam: curve}, {fam: fixed}, w[te])
    e1, e2 = pred(u1[te], u2[te])
    loss = _nan_full(len(u1)); loss[te] = _loss_from_pred(u1[te], u2[te], e1, e2)
    p_step = {}
    for q in Q_GRID:
        p = _nan_full(len(u1)); te_idx = np.flatnonzero(te)
        for i in te_idx:
            try:
                cop = _acar_cop_at(fam, curve, fixed, w[i])
                c = float(np.asarray(cop.cdf(np.array([[q, q]])))[0])
                p[i] = float(np.clip(1 - 2 * q + c, 0.0, 1.0))
            except Exception:                       # recorded as NaN; counted below
                p[i] = np.nan
        p_step[q] = p
        n_nan = int(np.isnan(p[te_idx]).sum())
        if n_nan:
            warnings.warn(f"Acar {fam}, fold {job['fold']}, q={q}: {n_nan}/{len(te_idx)} joint-exceedance "
                          f"probabilities could not be computed (NaN)")
    ks = max(_cop_ks(_acar_cop_at(fam, curve, fixed, float(v)), seed=137 + j)
             for j, v in enumerate(np.quantile(w[tr], GC.KS_PERCENTILES)))
    return dict(fold=job["fold"], family=fam, h=h, loss=loss, p_step=p_step, ks=ks,
                curve=curve, fixed=fixed, runtime=time.time() - t0)


# ── nonparametric estimators (Gijbels et al., 2011; Lu & Ghosh, 2024) per fold ───────────────────────────
def nonparametric_job(job):
    """One nonparametric conditional-copula estimator on one fold.

    The estimator (kind 'gijbels': Gijbels, Veraverbeke and Omelka, 2011, with the plug-in bandwidths of gijbels_copula; kind 'ecbc':
    the empirical checkerboard Bernstein copula of Lu and Ghosh, 2024) is fitted on the fold's training rows. It returns the cell masses
    of the conditional copula at a covariate value, so the test rows are scored as in ces_score: masses on a covariate grid spanning the test
    covariate values (clipped to the training range) and linear interpolation between the two bracketing grid points.
    CVPE interpolates the conditional means before squaring the prediction errors; CES interpolates scores. The conditional means
    are taken from the cell masses merged to gijbels_copula.LOSS_CELLS cells per margin, so it approximates conditional means differently from the exact-quadrature loss of the other models (the Copula Energy Score
    on the K x K cells is). The marginal-uniformity check simulates from the masses at the seven training-range percentiles of the covariate.

    Parameters:
    job (FoldJob or dict): Keys fold, arrays, kind ('gijbels' or 'ecbc') and hyper (optional 'K' and 'G', default ces_score.DEFAULT_K
        and DEFAULT_GRID_POINTS).

    Returns:
    dict: fold, kind, model, loss (n,), p_step {q: (n,)}, es_raw and es_cal (n,; Copula Energy Score of the raw and the margin-calibrated
        masses), ks (D_max over the percentiles and both margins), ks_pass, w_grid and tau_grid (Kendall's tau on the covariate grid),
        info (bandwidths or degrees), clipped_mass, margin_error_max and runtime. Rows outside the test window are NaN.
    """
    import ces_score as CES
    from scipy.stats import kstest
    t0 = time.time()
    A, kind = job["arrays"], job["kind"]
    hp = job.get("hyper") or {}
    K, G = int(hp.get("K", CES.DEFAULT_K)), int(hp.get("G", CES.DEFAULT_GRID_POINTS))
    tr, te = A["train"], A["test"]
    u1, u2, w = A["u1"], A["u2"], np.asarray(A["w"], float)
    n = len(u1)
    X, U = w[tr], np.column_stack([u1[tr], u2[tr]])
    if kind == "gijbels":
        g1, _ = GIJ.plugin_g(X, U[:, 0]); g2, _ = GIJ.plugin_g(X, U[:, 1])
        Ut = GIJ.conditional_pits(X, U, (g1, g2)); ok = np.all(np.isfinite(Ut), axis=1)
        h, _ = GIJ.plugin_h(X[ok], Ut[ok])
        fit = GIJ.GijbelsFit(X, U, h, (g1, g2), K)
        info = dict(h=float(h), g=[float(g1), float(g2)], bad_pit=fit.bad_pit)
    elif kind == "ecbc":
        fit = ECBC.ECBCFit(X, U, K)
        info = dict(degrees=fit.degrees)
    else:
        raise ValueError(f"unknown nonparametric kind {kind!r}")
    lo, hi = float(X.min()), float(X.max())
    cache = {}

    def masses(wv):
        """Cell masses at the covariate value wv, clipped to the training range (memoised)."""
        wv = float(np.clip(wv, lo, hi))
        if wv not in cache:
            cache[wv] = fit.masses(wv)
        return cache[wv]

    w_te = np.clip(w[te], lo, hi)
    grid = np.linspace(float(w_te.min()), float(w_te.max()), G)
    raw, cal = GIJ.pointwise_on_grid_both(masses, w_te, u1[te], u2[te], grid, K)
    out = dict(fold=job["fold"], kind=kind, model=GIJBELS_MODEL if kind == "gijbels" else ECBC_MODEL,
               loss=_nan_full(n), es_raw=_nan_full(n), es_cal=_nan_full(n), p_step={q: _nan_full(n) for q in Q_GRID})
    out["loss"][te], out["es_raw"][te], out["es_cal"][te] = raw["loss"], raw["es"], cal["es"]
    for q in Q_GRID:
        pg = [GIJ.joint_exceed_from_masses(masses(x), K, q) for x in grid]
        out["p_step"][q][te] = exceed_step_from_grid(w_te, grid, pg)
    ks = []
    for j, v in enumerate(np.quantile(X, GC.KS_PERCENTILES)):
        d = GIJ.sample_from_masses(masses(v), K, GC.KS_N_SAMPLES, seed=137 + j)
        ks.append(max(float(kstest(d[:, c], "uniform").statistic) for c in range(2)))
    out.update(ks=max(ks), ks_pass=bool(GC.ks_seed_pass(max(ks), 2 * len(GC.KS_PERCENTILES), False)),
               w_grid=grid, tau_grid=np.array([GIJ.tau_from_masses(masses(x), K) for x in grid]), info=info,
               clipped_mass=float(fit.log.get("clipped_mass", 0.0)),
               margin_error_max=float(max(CES.margin_error(masses(x), K) for x in grid[:: max(1, G // 9)])),
               n_test_outside_support=int(np.sum((w[te] < lo) | (w[te] > hi))), runtime=time.time() - t0)
    return out


# ── weight decay ───────────────────────────────────────────────────────────
# Three values, one per architecture group, chosen by notebook 03 on synthetic data (not on the real data):
#   * WD_CCODINE: the three C-CODINE networks.
#   * WD_CCOPFLOW_AFFINE: the three C-CopFlow networks with affine coupling.
#   * WD_CCOPFLOW_SPLINE: the three C-CopFlow networks with rational-quadratic spline coupling.
# Notebook 03 trains the w-only variant of each group on ten synthetic designs (fresh draws, other sample sizes than the penalty stage) over
# WD_GRID, with the penalty off, and takes the lower median over the ten cells of the weight decay with the lowest excess CVPE. The
# values are read from margin_penalty_calibrated.json (the same file as the penalty weights).
# The unconditional CODINE and the two regime CODINE networks are not conditional networks and keep WD_CODINE_UNCOND = 0, the
# setting of the CODINE paper (Letizia et al., 2025), which trains with Adam and adds no regularisation term.
# wd_tuning_job, build_wd_jobs and select_wd serve only the sensitivity check on fold A1 (RUN_WD_SENSITIVITY in notebook 04); they never
# choose a value used in the study.
# The optimiser is Adam with coupled L2 (torch.optim.Adam(weight_decay=...)). Under Adam, L2 regularisation and weight
# decay are not equivalent (Loshchilov & Hutter, 2017), so the value is an L2 coefficient.
WD_CODINE_UNCOND = 0.0
WD_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
WD_FALLBACK = (0.0, 1e-6, 1e-6)      # (C-CODINE, C-CopFlow affine, C-CopFlow spline): only for uncalibrated imports and smoke runs
GRAD_CLIP_NORM = 5.0          # gradient-norm clip of all 12 networks (the conditional trainers default to the same 5.0)
# Weights of the marginal-uniformity penalty of the two conditional network families (codine.margin_uniformity_penalty; Aich & Aich, 2025,
# eq. 1). 0 switches it off. One weight per group, as for the weight decay: C-CODINE, C-CopFlow with affine coupling, C-CopFlow with spline
# coupling. Order of precedence: the environment variable THESIS_MARGIN_PENALTY (one value for all groups); the values chosen by notebook
# 03_marginal_uniformity_penalty_calibration (margin_penalty_calibrated.json in the artifact folder). The weights and the weight decays are
# tuned once on the ten synthetic designs and reused unchanged on every synthetic design and on the real data.
# The file is mandatory for production network jobs; notebook 03 and smoke runs explicitly allow an uncalibrated import.


def _read_calibration(path):
    """Read and validate the calibration file written by notebook 03.

    Parameters:
    path (Path): margin_penalty_calibrated.json.

    Returns:
    dict: The file content.

    Raises:
    ValueError: If the format is outdated, the file comes from a smoke run in a production run, or a value is invalid.
    """
    import json
    with open(path) as fh:
        d = json.load(fh)
    if d.get("schema_version") != 4:
        raise ValueError(f"Outdated calibration format in {path}; rerun notebook 03")
    if d.get("smoke") and os.getenv("THESIS_SMOKE") != "1":
        raise ValueError(f"Smoke-run calibration cannot be used in production: {path}")
    pen = (float(d["penalty_codine"]), float(d["penalty_flow_affine"]), float(d["penalty_flow_spline"]))
    wd = (float(d["wd_codine"]), float(d["wd_flow_affine"]), float(d["wd_flow_spline"]))
    if not all(np.isfinite(v) and v >= 0 for v in pen + wd):
        raise ValueError(f"Invalid calibrated weights in {path}")
    if not d.get("smoke") and d.get("real_binned_ks") is None and not d.get("delta_override"):
        raise ValueError(f"Calibration did not use real folds or an explicit delta override: {path}")
    return d


def _load_margin_penalties():
    """Penalty weights (C-CODINE, C-CopFlow affine, C-CopFlow spline coupling) from the environment or the mandatory calibration file.

    Returns:
    tuple: The three weights.
    """
    env = os.environ.get("THESIS_MARGIN_PENALTY")
    if env is not None:
        weight = float(env)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("THESIS_MARGIN_PENALTY must be finite and nonnegative")
        return weight, weight, weight
    if os.getenv("THESIS_ALLOW_UNCALIBRATED_IMPORT") == "1":
        return 0.0, 0.0, 0.0
    path = Path(ARTIFACT_DIR) / "margin_penalty_calibrated.json"
    if path.exists():
        d = _read_calibration(path)
        return float(d["penalty_codine"]), float(d["penalty_flow_affine"]), float(d["penalty_flow_spline"])
    if os.getenv("THESIS_SMOKE") == "1":
        return 0.0, 0.0, 0.0
    raise FileNotFoundError(f"Missing {path}: run notebook 03 before importing fold_pipeline for production jobs. "
                            "To switch off the penalty explicitly set THESIS_MARGIN_PENALTY=0.")


def _load_weight_decays():
    """Weight decays (C-CODINE, C-CopFlow affine, C-CopFlow spline coupling) from the mandatory calibration file.

    Returns:
    tuple: The three weight decays; WD_FALLBACK in uncalibrated imports (notebooks 01 to 03) and in smoke runs without a file.
    """
    if os.getenv("THESIS_ALLOW_UNCALIBRATED_IMPORT") == "1":
        return WD_FALLBACK
    path = Path(ARTIFACT_DIR) / "margin_penalty_calibrated.json"
    if path.exists():
        d = _read_calibration(path)
        return float(d["wd_codine"]), float(d["wd_flow_affine"]), float(d["wd_flow_spline"])
    if os.getenv("THESIS_SMOKE") == "1":
        return WD_FALLBACK
    raise FileNotFoundError(f"Missing {path}: run notebook 03 before importing fold_pipeline for production jobs.")


MARGIN_PENALTY_CODINE, MARGIN_PENALTY_FLOW_AFFINE, MARGIN_PENALTY_FLOW_SPLINE = _load_margin_penalties()
WD_CCODINE, WD_CCOPFLOW_AFFINE, WD_CCOPFLOW_SPLINE = _load_weight_decays()
MARGIN_N_W = 2                 # covariate values of a minibatch on which the penalty is evaluated
MARGIN_GRID = 12               # grid points per coordinate of the penalty
MARGIN_EVERY = 4               # the penalty is added on every 4th minibatch, with 4 times the weight


def wd_of_model(model, wd_codine=None, wd_affine=None, wd_spline=None):
    """Weight decay of one conditional network: by its group (C-CODINE, C-CopFlow affine, C-CopFlow spline coupling).

    Parameters:
    model (str): Name of a conditional network.
    wd_codine (float or None): Value of the C-CODINE group. Default None (WD_CCODINE).
    wd_affine (float or None): Value of the affine C-CopFlow group. Default None (WD_CCOPFLOW_AFFINE).
    wd_spline (float or None): Value of the spline-coupling C-CopFlow group. Default None (WD_CCOPFLOW_SPLINE).

    Returns:
    float: Weight decay.
    """
    if model in CCODINE_VARIANTS:
        return float(WD_CCODINE if wd_codine is None else wd_codine)
    kind = CCOPFLOW_VARIANTS[model][1]
    if kind == "spline":
        return float(WD_CCOPFLOW_SPLINE if wd_spline is None else wd_spline)
    return float(WD_CCOPFLOW_AFFINE if wd_affine is None else wd_affine)


def fixed_wd_by_model(value=None):
    """Weight decay of each conditional network.

    Parameters:
    value (float): If given, one common value for all nine networks (sensitivity use only). Default None: the calibrated value
        of the network's group (see wd_of_model).

    Returns:
    dict: {model name: weight decay}.
    """
    if value is not None:
        return {m: float(value) for m in list(CCODINE_VARIANTS) + list(CCOPFLOW_VARIANTS)}
    return {m: wd_of_model(m) for m in list(CCODINE_VARIANTS) + list(CCOPFLOW_VARIANTS)}


def wd_key(j):
    """Resume key of a weight-decay sensitivity job.

    Parameters:
    j (FoldJob or dict): Job with fold, model and wd.

    Returns:
    str: Key of the form wd__<fold>__<model>__<wd>.
    """
    return f"wd__{j['fold']}__{''.join(ch if ch.isalnum() else '_' for ch in j['model'])}__{j['wd']:g}"


def wd_tuning_job(job):
    """Train one conditional network at one weight decay and return its best validation CVPE.

    Uses the same trainer, batch size, patience and validation purge as the production runs (train_net), on the
    training rows of one fold.

    Parameters:
    job (FoldJob or dict): Job with fold, arrays, kind, model, wd and optionally seed and hyper.

    Returns:
    dict: fold, model, wd, best_val_cvpe, best_epoch and runtime.

    Raises:
    RuntimeError: If no finite validation CVPE was obtained (the run diverged).
    """
    t0 = time.time()
    A = job["arrays"]; hp = _hyper(job)
    tr = A["train"]
    out = train_net(job["kind"], job["model"], job.get("seed", 0), A["u1"][tr], A["u2"][tr],
                    A["w"][tr], None, hp, job["wd"], A["val_purge"])
    res = out["res"]
    if res["best_epoch"] is None:
        raise RuntimeError(f"weight-decay tuning: no finite validation CVPE for {job['model']} at wd={job['wd']:g} "
                           f"(fold {job['fold']}); the run diverged")
    return dict(fold=job["fold"], model=job["model"], wd=float(job["wd"]),
                best_val_cvpe=float(res["best_val_cvpe"]), best_epoch=int(res["best_epoch"]),
                runtime=time.time() - t0)


def build_wd_jobs(arrays_by_fold, fold="A1", grid=WD_GRID, seed=0, smoke=False):
    """One job per conditional network variant and weight-decay candidate (9 variants x len(grid)).

    Parameters:
    arrays_by_fold (dict): {fold: fold_arrays}.
    fold (str): Fold to run on. Default 'A1'.
    grid (sequence of float): Candidate weight decays. Default WD_GRID.
    seed (int): Network seed. Default 0.
    smoke (bool): Shrink the training settings for a quick test. Default False.

    Returns:
    list: FoldJob objects.
    """
    hyper = default_hyper()
    if smoke:
        hyper.update(max_epochs=60, checkpoint_every=20, patience=2)
    jobs = []
    for v in CCODINE_VARIANTS:
        for wd in grid:
            jobs.append(FoldJob(fold=fold, model=v, kind="ccodine", wd=wd, seed=seed, arrays=arrays_by_fold[fold], hyper=hyper))
    for v in CCOPFLOW_VARIANTS:
        for wd in grid:
            jobs.append(FoldJob(fold=fold, model=v, kind="ccopflow", wd=wd, seed=seed, arrays=arrays_by_fold[fold], hyper=hyper))
    return jobs


def select_wd(results):
    """Weight decay with the lowest best validation CVPE for each variant.

    Parameters:
    results (dict): {key: result of wd_tuning_job}.

    Returns:
    tuple: (dict {model: weight decay}, DataFrame of all candidates sorted by model and weight decay).
    """
    tab = pd.DataFrame([{k: v for k, v in r.items()} for r in results.values()])
    best = tab.sort_values("best_val_cvpe").groupby("model").head(1)
    return {m: float(w) for m, w in zip(best.model, best.wd)}, tab.sort_values(["model", "wd"]).reset_index(drop=True)


# ── neural networks: one per (fold, seed, model) ───────────────────────────
def net_key(fold, seed, model):
    """Resume key of a network job.

    Parameters:
    fold (str): Fold name.
    seed (int): Network seed.
    model (str): Model name.

    Returns:
    str: Key of the form <fold>__s<seed>__<model with non-alphanumeric characters replaced by '_'>.
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in model)
    return f"{fold}__s{seed}__{safe}"


def default_hyper():
    """Hyper-parameters of the network jobs.

    All 12 networks share one training recipe except the weight decay (one value per group, see wd_of_model): Adam with learning rate
    1e-3, batch size 32, gradient clipping at 5, validation every 100 epochs and patience of 5 checkpoints. Gate-related
    values come from gate_config. build_fold_jobs copies this dict into every job, so the resume fingerprint of a job
    changes whenever one of these values changes.

    Returns:
    dict: The hyper-parameters.
    """
    return dict(max_epochs=8000, checkpoint_every=100, patience=5, codine_checkpoint_every=100,
                codine_weight_decay=WD_CODINE_UNCOND, codine_grad_clip=GRAD_CLIP_NORM,
                val_frac=0.25, n_cells=100, n_w_grid=50, ks_n_samples=GC.KS_N_SAMPLES, grid_n=QUAD_N,
                margin_penalty_codine=MARGIN_PENALTY_CODINE, margin_penalty_flow_affine=MARGIN_PENALTY_FLOW_AFFINE,
                margin_penalty_flow_spline=MARGIN_PENALTY_FLOW_SPLINE, margin_n_w=MARGIN_N_W, margin_grid=MARGIN_GRID, margin_every=MARGIN_EVERY)


def _hyper(job):
    """Hyper-parameters of a job: the defaults updated with the job's own overrides.

    Parameters:
    job (FoldJob or dict): Job with an optional hyper dict.

    Returns:
    dict: Hyper-parameters.
    """
    h = default_hyper()
    h.update(job.get("hyper", {}))
    return h


def train_net(kind, model, seed, u1_tr, u2_tr, w_tr, regime_mask_tr, hp, wd, val_purge):
    """Train one network on the given training rows.

    Shared by the real-data fold batch and the synthetic study.

    Parameters:
    kind (str): 'codine_uncond', 'codine_regime' (regime_mask_tr marks the rows of the regime), 'ccodine' or
        'ccopflow'.
    model (str): Model name, a key of CCODINE_VARIANTS or CCOPFLOW_VARIANTS (ignored by the CODINE kinds).
    seed (int): Network seed.
    u1_tr (np.ndarray): First pseudo-observation of the training rows.
    u2_tr (np.ndarray): Second pseudo-observation of the training rows.
    w_tr (np.ndarray): Covariate of the training rows (not used by the CODINE kinds).
    regime_mask_tr (np.ndarray): Boolean regime mask of the training rows (codine_regime only), else None.
    hp (dict): Hyper-parameters, as default_hyper.
    wd (float): Weight decay of the conditional networks (the CODINE kinds use hp['codine_weight_decay']).
    val_purge (int): Purge between the training and validation parts of the early-stopping split, already clamped
        (1 for i.i.d. synthetic data).

    Returns:
    dict: obj (the fitted CopulaNet, ConditionalCopulaNet or ConditionalCopulaFlow), res (early-stopping result),
        dens (dens(pts, w) -> density; w is ignored by the unconditional kinds) and sample_fn (sample_fn(w, n, seed) ->
        (n, 2) samples; None for the unconditional kinds).

    Raises:
    ValueError: If kind is unknown.
    """
    u1_tr = np.asarray(u1_tr); u2_tr = np.asarray(u2_tr); w_tr = np.asarray(w_tr)
    if kind in ("codine_uncond", "codine_regime"):
        # verbose=True also switches the tqdm epoch bar off here (train_codine_early_stop sets show_progress = not verbose);
        # the conditional trainers have no bar by default, so there verbose=False only skips the one-line summary.
        from codine import train_codine_early_stop, eval_density
        kw = dict(b=val_purge, max_epochs=hp["max_epochs"], checkpoint_every=hp["codine_checkpoint_every"],
                  patience=hp["patience"], val_frac=hp["val_frac"], seed=seed, verbose=True, calib_seed=seed,
                  weight_decay=float(hp["codine_weight_decay"]), grad_clip_norm=hp["codine_grad_clip"])
        if kind == "codine_uncond":
            res = train_codine_early_stop(u1_tr.astype(np.float32), u2_tr.astype(np.float32),
                                          split_mode="chronological", **kw)
        else:
            res = train_codine_early_stop(u1_tr.astype(np.float32), u2_tr.astype(np.float32),
                                          split_mode="regime_chronological", regime_mask=regime_mask_tr, **kw)
        net = res["net"]
        return dict(obj=net, res=res, dens=(lambda pts, wv, o=net: eval_density(o, pts)), sample_fn=None)
    if kind == "ccodine":
        from codine import (train_conditional_codine_early_stop, eval_conditional_density,
                                        make_codine_sample_fn)
        res = train_conditional_codine_early_stop(
            u1_tr, u2_tr, w_tr, b=val_purge, val_frac=hp["val_frac"], max_epochs=hp["max_epochs"],
            checkpoint_every=hp["checkpoint_every"], seed=seed, patience=hp["patience"], verbose=False,
            batch_size=N_BATCH, weight_decay=float(wd), conditioning_mode=CCODINE_VARIANTS[model],
            calib_seed=seed, margin_penalty=float(hp.get("margin_penalty", hp.get("margin_penalty_codine", 0.0))), margin_n_w=int(hp.get("margin_n_w", MARGIN_N_W)),
            margin_grid=int(hp.get("margin_grid", MARGIN_GRID)), margin_every=int(hp.get("margin_every", MARGIN_EVERY)))
        obj = res["net"]
        return dict(obj=obj, res=res, dens=(lambda pts, wv, o=obj: eval_conditional_density(o, pts, wv)),
                    sample_fn=make_codine_sample_fn(obj))
    if kind == "ccopflow":
        from conditional_copula_flow import (train_conditional_copula_flow_early_stop, eval_flow_density,
                                             make_flow_sample_fn)
        mode, coupling = CCOPFLOW_VARIANTS[model]
        res = train_conditional_copula_flow_early_stop(
            u1_tr, u2_tr, w_tr, b=val_purge, val_frac=hp["val_frac"], max_epochs=hp["max_epochs"],
            checkpoint_every=hp["checkpoint_every"], seed=seed, patience=hp["patience"], verbose=False,
            n_layers=8, hidden=64, scale_bound=3.5, batch_size=N_BATCH, weight_decay=float(wd),
            conditioning_mode=mode, coupling_type=coupling, margin_penalty=float(hp.get("margin_penalty", hp.get(f"margin_penalty_flow_{coupling}", 0.0))),
            margin_n_w=int(hp.get("margin_n_w", MARGIN_N_W)), margin_grid=int(hp.get("margin_grid", MARGIN_GRID)), margin_every=int(hp.get("margin_every", MARGIN_EVERY)))
        obj = res["flow"]
        return dict(obj=obj, res=res, dens=(lambda pts, wv, o=obj: eval_flow_density(o, pts, wv)),
                    sample_fn=make_flow_sample_fn(obj))
    raise ValueError(f"unknown neural job kind {kind!r}")


def neural_job(job):
    """Train one network for a real-data fold and score it on the test rows.

    Top-level and picklable, so it can go to a ProcessPoolExecutor. Returns the per-step test loss, the predictions, the
    joint-exceedance probabilities and the diagnostics. For the conditional networks the marginal-uniformity check is made
    on the fold's training covariate range (no test information) and the joint-exceedance probability is computed on a
    covariate grid and interpolated at the test steps.

    Parameters:
    job (FoldJob or dict): Keys fold, seed, model, kind (one of the neural kinds), arrays, regime ('low' or 'high', for
        codine_regime), wd, hyper (overrides) and keep_net.

    Returns:
    dict: fold, seed, model, kind, regime, loss, e1, e2 (all (n,), NaN outside the scored rows), p_step {q: (n,)}, ks (D_max
        of the marginal check), ks_pass, ks_rejection_rate (conditional networks), best_epoch, best_val_cvpe,
        stopped_epoch (last epoch at which the validation CVPE was evaluated), net (if keep_net, else None), losses (if
        keep_net) and runtime.
    """
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    hp = _hyper(job)
    A, kind, seed, model = job["arrays"], job["kind"], int(job["seed"]), job["model"]
    u1, u2, w = A["u1"], A["u2"], A["w"]
    tr, te, hi = A["train"], A["test"], A["high"]
    n = len(u1)
    out = dict(fold=job["fold"], seed=seed, model=model, kind=kind, regime=job.get("regime"),
               loss=_nan_full(n), e1=_nan_full(n), e2=_nan_full(n),
               p_step={q: _nan_full(n) for q in Q_GRID}, ks=np.nan, ks_pass=None,
               net=None)
    rmask = np.ones(n, bool)
    if kind == "codine_regime":
        rmask = hi if job["regime"] == "high" else ~hi
    T = train_net(kind, model, seed, u1[tr], u2[tr], w[tr], rmask[tr], hp, job.get("wd", 0.0), A["val_purge"])
    obj, res, dens = T["obj"], T["res"], T["dens"]

    if kind in ("codine_uncond", "codine_regime"):
        from fold_models import make_codine_predictor
        sel = te & rmask
        e1, e2 = make_codine_predictor(obj, grid_size=hp["grid_n"])(u1[sel].astype(np.float32),
                                                                    u2[sel].astype(np.float32))
        out["loss"][sel] = _loss_from_pred(u1[sel], u2[sel], e1, e2)
        out["e1"][sel], out["e2"][sel] = e1, e2
        for q in Q_GRID:
            out["p_step"][q][sel] = exceed_prob_density(dens, q, None, hp["n_cells"])
        from model_evaluation import marginal_uniformity_check, ks_gate_summary
        # Unconditional CODINE gets the same sample-based marginal KS check. A missing or failed sampler fails closed
        # in the gate.
        try:
            from codine import sample_from_codine
            sample_fn = lambda _wv, count, sd: sample_from_codine(obj, n_samples=count, seed=sd)
            chk = marginal_uniformity_check(sample_fn, [0.5], n_samples=hp["ks_n_samples"], seed=100 * seed)
            gate = ks_gate_summary(chk, grid=True)        # CODINE uses the grid sampler
            out["ks"], out["ks_pass"] = gate["d_max"], gate["pass"]
        except Exception as exc:
            warnings.warn(f"CODINE marginal KS failed ({model}, fold {job['fold']}): {exc}")
            out["ks"], out["ks_pass"] = np.nan, False
        out.update(best_epoch=res["best_epoch"], best_val_cvpe=res["best_val_cvpe"],
                   stopped_epoch=max(res["val_cvpe"].keys()))
    else:
        from model_evaluation import (neural_conditional_predictions, marginal_uniformity_check,
                                      marginal_uniformity_summary, ks_gate_summary)
        e1, e2, _ = neural_conditional_predictions(dens, u1[te], u2[te], w[te].astype(float),
                                                   grid_n=hp["grid_n"])
        out["loss"][te] = _loss_from_pred(u1[te], u2[te], e1, e2)
        out["e1"][te], out["e2"][te] = e1, e2
        # Marginal-uniformity gate on the training covariate range of the fold (no test information): D_max over the
        # 7 percentile values x 2 coordinates, with hp["ks_n_samples"] draws at each value. The draws use a seed that
        # depends on the network seed. out["ks"] is D_max; the Bonferroni rejection rate is stored as a diagnostic only.
        w_chk = np.quantile(w[tr], GC.KS_PERCENTILES)
        chk = marginal_uniformity_check(T["sample_fn"], w_chk, n_samples=hp["ks_n_samples"], seed=100 * seed)
        gate = ks_gate_summary(chk, grid=(model in CCODINE_VARIANTS))     # C-CODINE: grid sampler; flows: exact
        out["ks"] = gate["d_max"]
        out["ks_pass"] = gate["pass"]
        out["ks_rejection_rate"] = marginal_uniformity_summary(chk)["rejection_rate"]
        # Joint-exceedance probability on a w grid, interpolated at the test steps.
        w_grid = np.linspace(np.quantile(w[tr], 0.01), np.quantile(w[tr], 0.99), hp["n_w_grid"])
        for q in Q_GRID:
            pg = np.array([exceed_prob_density(dens, q, float(wg), hp["n_cells"]) for wg in w_grid])
            out["p_step"][q][te] = exceed_step_from_grid(w[te], w_grid, pg)
        out.update(best_epoch=res["best_epoch"], best_val_cvpe=res["best_val_cvpe"],
                   stopped_epoch=max(res["val_cvpe"].keys()))
    if job.get("keep_net"):
        out["net"] = obj; out["losses"] = res["losses"]
    out["runtime"] = time.time() - t0
    return out


def neural_job_persist(job, state_dir=None):
    """Run neural_job and store the fitted best-checkpoint network of every job.

    Runs in the worker. The network is written to <batch_dir>/states/<job key>.pt by fold_models.save_state (weights,
    configuration, normalisation, checksum), and is dropped from the returned result unless the job asked for it, so the
    parent process does not accumulate fitted models. This function is not part of the fitting digest (_STAGE_DEPS), and
    the resume fingerprint is computed on the unmodified job, so finished results stay valid.

    Parameters:
    job (FoldJob or dict): Job as for neural_job.
    state_dir (str or Path): The 'states' folder of the batch directory (<batch_dir>/states). If None or empty, nothing is
        stored.

    Returns:
    dict: Result of neural_job, with 'state' (file, sha256, version, or the error) when a network was stored.
    """
    import dataclasses
    j = dataclasses.replace(job, keep_net=True) if dataclasses.is_dataclass(job) else {**job, "keep_net": True}
    out = neural_job(j)
    net = out.get("net")
    if net is not None and state_dir:
        import fold_models as MS
        key = net_key(job["fold"], job["seed"], job["model"])
        try:
            side = MS.save_state(net, MS.state_path(Path(state_dir).parent, key), meta=dict(
                key=key, fold=job["fold"], seed=int(job["seed"]), model=job["model"], kind=job["kind"],
                regime=job.get("regime"), wd=job.get("wd"), hyper=_hyper(job),
                code_version=_STAGE_CODE_VERSIONS.get("neural"), best_epoch=out.get("best_epoch"),
                best_val_cvpe=out.get("best_val_cvpe")))
            out["state"] = dict(file=f"{MS.STATE_SUBDIR}/{key}.pt", sha256=side["sha256"], version=MS.VERSION)
        except Exception as exc:                       # never lose a finished training to a storage problem
            out["state"] = dict(error=repr(exc))
            warnings.warn(f"state not stored for {key}: {exc!r}")
    if not job.get("keep_net"):
        out["net"] = None
        out.pop("losses", None)
    return out


# ── batch grid, resume files, worker cap ───────────────────────────────────
BATCH_DIR = ARTIFACT_DIR / "fold_batch"
log = get_logger()


def build_fold_jobs(arrays_by_fold, wd_by_model, folds=FOLDS, seeds=SEEDS, keep_net_seed=0,
                    smoke=False, kinds=None):
    """The full neural task grid: len(folds) x len(seeds) x 12 networks (144 for 4 x 3, gate_config.N_NETWORK_TRAININGS).

    A subset of kinds gives jobs identical to those of the full grid, so notebooks 01, 02 and 04 share the same resume
    files. The three CODINE networks take their weight decay from hyper['codine_weight_decay'].

    Parameters:
    arrays_by_fold (dict): {fold: fold_arrays}.
    wd_by_model (dict): {model name: weight decay} for the 9 conditional networks (see fixed_wd_by_model). May be None when
        no conditional network is requested.
    folds (sequence of str): Folds. Default FOLDS.
    seeds (sequence of int): Seeds. Default SEEDS.
    keep_net_seed (int): Seed whose jobs keep the fitted network in the result. Default 0.
    smoke (bool): Shrink epochs and Monte Carlo sizes so that the grid runs in minutes (testing only). Default False.
    kinds (set of str): Optional subset of {'codine_uncond', 'codine_regime', 'ccodine', 'ccopflow'}. Default None (all).

    Returns:
    list: FoldJob objects.
    """
    hyper = default_hyper()
    if smoke:
        hyper.update(max_epochs=60, checkpoint_every=20, codine_checkpoint_every=20, patience=2,
                     n_cells=20, n_w_grid=5, ks_n_samples=200, grid_n=30)
    jobs = []
    for f in folds:
        A = arrays_by_fold[f]
        for s in seeds:
            base = dict(fold=f, seed=s, arrays=A, hyper=hyper, keep_net=(s == keep_net_seed))
            if kinds is None or "codine_uncond" in kinds:
                jobs.append(FoldJob(**base, model=UNCOND_CODINE, kind="codine_uncond"))
            if kinds is None or "codine_regime" in kinds:
                for reg in ("low", "high"):
                    jobs.append(FoldJob(**base, model=f"{SPLIT_CODINE}|{reg}", kind="codine_regime", regime=reg))
            if kinds is None or "ccodine" in kinds:
                for v in CCODINE_VARIANTS:
                    jobs.append(FoldJob(**base, model=v, kind="ccodine", wd=wd_by_model[v]))
            if kinds is None or "ccopflow" in kinds:
                for v in CCOPFLOW_VARIANTS:
                    jobs.append(FoldJob(**base, model=v, kind="ccopflow", wd=wd_by_model[v]))
    return jobs


def _job_key(j):
    """Resume key of a neural job.

    Parameters:
    j (FoldJob or dict): Job with fold, seed and model.

    Returns:
    str: net_key of the job.
    """
    return net_key(j["fold"], j["seed"], j["model"])


def _worker_init():
    """Initialiser of the worker processes.

    Sends detailed log messages to <THESIS_LOG_DIR>/worker_<pid>.log when that variable is set, and limits the BLAS, OpenMP
    and torch thread counts to one per worker.
    """
    _ld = os.environ.get("THESIS_LOG_DIR")
    if _ld:
        try:
            import logging
            Path(_ld).mkdir(parents=True, exist_ok=True)
            lg = get_logger()
            for h in list(lg.handlers):
                lg.removeHandler(h)
            fh = logging.FileHandler(Path(_ld) / f"worker_{os.getpid()}.log")
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            lg.addHandler(fh)
        except Exception:
            pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _job_cost(j):
    """Rough relative cost of a job, used only to order submissions (longest first).

    Parameters:
    j (FoldJob or dict): Job.

    Returns:
    float: Cost proxy.
    """
    kind = j.get("kind")
    c = {"ccopflow": 5.0, "ccodine": 3.0, "codine_regime": 2.0, "codine_uncond": 2.0}.get(kind, 1.0)
    if "family" in j:                              # Acar local-likelihood job
        c += 3.0
    if kind in ("gijbels", "ecbc"):
        c += 4.0
    return c


def _scalar_row(r, key):
    """Scalar diagnostics of a finished job as a flat dict, for the CSV run log.

    Parameters:
    r (dict): Result of a job.
    key (str): Job key.

    Returns:
    dict: 'key' plus every scalar entry of r.
    """
    row = {"key": key}
    for k, v in r.items():
        if isinstance(v, (int, float, str, bool, np.integer, np.floating, np.bool_)) or v is None:
            row[k] = v
    return row


def _step_frame(r):
    """Per-step test results of a finished job.

    Parameters:
    r (dict): Result of a job.

    Returns:
    pd.DataFrame: Columns pos, loss and, where present, e1, e2 and p_q<q> (or one loss column per model for the parametric
        jobs), restricted to rows with a finite loss. None if the result has no loss.
    """
    loss = r.get("loss")
    cols = {}
    if isinstance(loss, np.ndarray):
        cols["loss"] = loss
        for nm in ("e1", "e2"):
            if isinstance(r.get(nm), np.ndarray):
                cols[nm] = r[nm]
        if isinstance(r.get("p_step"), dict):
            for q, v in r["p_step"].items():
                cols[f"p_q{q:g}"] = np.asarray(v, float)
        ref = loss
    elif isinstance(loss, dict) and loss:
        for m, v in loss.items():
            cols[m] = np.asarray(v, float)
        ref = None
    else:
        return None
    df = pd.DataFrame(cols)
    df.insert(0, "pos", np.arange(len(df)))
    keep = df["loss"].notna() if ref is not None else df.drop(columns="pos").notna().any(axis=1)
    return df[keep]


def persist_job_csv(rows, r, key, csv_dir, log_name="run_log"):
    """Write one finished job to CSV, so nothing is lost if the kernel dies.

    The scalar run log is rewritten atomically after every job (it stays small). The per-step results go to
    step_<key>.csv.

    Parameters:
    rows (dict): {key: scalar row}, updated in place.
    r (dict): Result of the job.
    key (str): Job key.
    csv_dir (str or Path): Output directory.
    log_name (str): Name of the run log without extension. Default 'run_log'.
    """
    csv_dir = Path(csv_dir); csv_dir.mkdir(parents=True, exist_ok=True)
    rows[key] = _scalar_row(r, key)
    tmp = csv_dir / f"{log_name}.csv.tmp"
    pd.DataFrame(list(rows.values())).to_csv(tmp, index=False)
    os.replace(tmp, csv_dir / f"{log_name}.csv")
    sf = _step_frame(r)
    if sf is not None:
        sf.to_csv(csv_dir / f"step_{key}.csv", index=False)


_NUMERIC_MODULES = ("fold_pipeline", "model_evaluation", "codine", "conditional_copula_flow", "conditional_copula",
                    "copula_families", "fold_models", "data_split", "synthetic_study")
# gate_config is deliberately not part of this digest: the gate decisions (the D_max thresholds, in every fold, seed and
# regime) are applied after the jobs, in gate_table. Every gate-related value that changes a job's result (KS sample
# size, covariate percentiles) is copied into the job itself (default_hyper, job fields), so it enters the job
# fingerprint through the job.


def _source_digest(path):
    """Digest of one module's code that ignores comments, docstrings and layout.

    The digest is the AST of the module with the docstrings removed. If the file cannot be parsed, the raw bytes are
    returned instead.

    Parameters:
    path (str or Path): Source file.

    Returns:
    bytes: The AST dump, or the raw file content.
    """
    import ast
    raw = Path(path).read_bytes()
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return raw
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=True, include_attributes=False).encode()


def code_version(root=None, modules=None):
    """SHA-1 over the code of every module that changes numerical results.

    Resume files written by an older version of the code are therefore recomputed instead of silently reused. Comment,
    docstring and layout edits do not change the digest. Any change to executable code does, including the family table
    and the rotation handling in copula_families.py.

    Parameters:
    root (path): Directory holding the modules. Default: the directory of this file.
    modules (sequence of str): Module names. Default _NUMERIC_MODULES.

    Returns:
    str: The first 12 hex characters of the digest.
    """
    import hashlib
    h = hashlib.sha1()
    here = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    for m in (modules or _NUMERIC_MODULES):
        p = here / f"{m}.py"
        h.update(m.encode())
        if p.exists():
            h.update(_source_digest(p))
        else:
            h.update(b"<missing>")
    return h.hexdigest()[:12]


# Source dependencies of the fits, not of the downstream MCS and tables. A change to result-producing code
# invalidates the jobs of its stage; a change to tables, reporting or simulations does not.
_STAGE_DEPS = {
    "param": (("copula_families", "model_evaluation", "fold_models", "data_split", "gate_config"),
              ("parametric_fold_job", "_cop_ks", "joint_exceed_probs_parametric", "_loss_from_pred", "_strip_rows",
               "exceed_prob_density", "_exceedance_midpoint_grid", "exceed_step_from_grid", "joint_exceed_prob_cop")),
    "acar": (("conditional_copula", "copula_families", "model_evaluation",
              "fold_models", "data_split", "gate_config"),
             ("acar_job", "_acar_cop_at", "_cop_ks", "_loss_from_pred", "_nan_full")),
    "nonparam": (("gijbels_copula", "ecbc_copula", "ces_score", "model_evaluation", "gate_config"),
                 ("nonparametric_job", "_nan_full", "exceed_step_from_grid")),
    "neural": (("codine", "conditional_copula_flow", "copula_families",
                "model_evaluation", "fold_models", "data_split", "gate_config"),
               ("train_net", "neural_job", "_cop_ks", "_loss_from_pred", "_nan_full", "exceed_prob_density",
                "_exceedance_midpoint_grid", "exceed_step_from_grid", "default_hyper", "_hyper")),
}
# The unconditional and regime C-CODINE jobs do not use the marginal-uniformity penalty, so their fingerprints must not depend on
# it (notebooks 01 and 02 may run before the penalty weights exist).
_STAGE_DEPS["neural_uncond"] = _STAGE_DEPS["neural"]

# Module constants that enter the results of a stage without being part of a job (the job fingerprint covers the rest).
_STAGE_CONSTANTS = {
    "param": ("Q_GRID",),
    "acar": ("Q_GRID",),
    "nonparam": ("Q_GRID",),
    "neural": ("Q_GRID", "N_BATCH", "CCODINE_VARIANTS", "CCOPFLOW_VARIANTS", "GRAD_CLIP_NORM", "WD_CCODINE", "WD_CCOPFLOW_AFFINE", "WD_CCOPFLOW_SPLINE",
               "MARGIN_PENALTY_CODINE", "MARGIN_PENALTY_FLOW_AFFINE", "MARGIN_PENALTY_FLOW_SPLINE", "MARGIN_N_W", "MARGIN_GRID", "MARGIN_EVERY"),
    "neural_uncond": ("Q_GRID", "GRAD_CLIP_NORM", "WD_CODINE_UNCOND"),
}


def _fitting_digest(kind):
    """Digest of the code that produces the fits of one stage.

    Covers the AST of the stage's modules (_STAGE_DEPS), the AST of its fitting functions in this file and the values of the
    module constants it reads (_STAGE_CONSTANTS), so editing mcs_table or run_batch does not invalidate an already computed
    fit, while editing a fitting function or a constant such as N_BATCH or Q_GRID does.

    Parameters:
    kind (str): Stage, a key of _STAGE_DEPS. An unknown kind uses the 'neural' stage.

    Returns:
    str: The first 12 hex characters of the digest.
    """
    import hashlib
    modules, symbols = _STAGE_DEPS.get(kind, _STAGE_DEPS["neural"])
    h = hashlib.sha1()
    root = Path(__file__).resolve().parent
    for name in modules:
        h.update(name.encode())
        h.update(_source_digest(root / (name + ".py")))
    for name in _STAGE_CONSTANTS.get(kind, _STAGE_CONSTANTS["neural"]):
        h.update(f"{name}={globals()[name]!r}".encode())
    import ast
    # Only the AST of the fitting functions: editing mcs_table or run_batch does not invalidate a computed fit.
    tree = ast.parse((root / "fold_pipeline.py").read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in symbols:
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                    getattr(node.body[0], "value", None), ast.Constant) and isinstance(
                    node.body[0].value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
            h.update(ast.dump(node, annotate_fields=True, include_attributes=False).encode())
    return h.hexdigest()[:12]


_STAGE_CODE_VERSIONS = {k: _fitting_digest(k) for k in _STAGE_DEPS}
_CODE_VERSION = code_version()      # fallback digest for job kinds without a stage of their own (the synthetic study)


def _fingerprint(obj):
    """SHA-1 over everything a job's result depends on.

    Covers the hyper-parameters, seeds, flags, the content of every input array and the code version of the job's stage, so
    resume files from a different configuration (smoke versus full, another bandwidth, re-run preprocessing) are
    recomputed, not silently reused.

    Parameters:
    obj (FoldJob or dict): The job.

    Returns:
    str: Hex digest.
    """
    import hashlib
    h = hashlib.sha1()

    def feed(o):
        """Add the content of o to the digest, recursively.

        Parameters:
        o: Job, dict, list, tuple, array or scalar to hash.
        """
        if is_dataclass(o):
            o = {f.name: getattr(o, f.name) for f in fields(o)}
        if isinstance(o, dict):
            for k in sorted(o, key=str):
                h.update(str(k).encode()); feed(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(b"[")
            for v in o:
                feed(v)
            h.update(b"]")
        elif isinstance(o, np.ndarray):
            h.update(str(o.dtype).encode()); h.update(str(o.shape).encode())
            h.update(np.ascontiguousarray(o).tobytes())
        else:
            h.update(repr(o).encode())
    kind = obj.get("kind", "") if isinstance(obj, dict) else getattr(obj, "kind", "")
    if kind in ("codine_uncond", "codine_regime"):
        obj = {f.name: getattr(obj, f.name) for f in fields(obj)} if is_dataclass(obj) else dict(obj)
        obj["hyper"] = {k: v for k, v in (obj.get("hyper") or {}).items() if not k.startswith("margin_")}
    feed(obj)
    stage = ("neural" if kind in ("ccodine", "ccopflow")
             else "neural_uncond" if kind in ("codine_uncond", "codine_regime")
             else "nonparam" if kind in ("gijbels", "ecbc") else kind)
    h.update(_STAGE_CODE_VERSIONS.get(stage, _CODE_VERSION).encode())
    return h.hexdigest()


class IncompleteBatch(RuntimeError):
    """Raised after a sharded run: this shard's jobs are done but results of other shards are still missing."""


def _record_failure(batch_dir, key):
    """Save the traceback of the exception being handled under <batch_dir>/failures/<key>.txt.

    The caller re-raises, so behaviour is unchanged, but the reason survives a dead kernel. The file is read by
    reporting_tables.py.

    Parameters:
    batch_dir (str or Path): Batch directory.
    key (str): Job key.
    """
    import traceback
    try:
        d = Path(batch_dir) / "failures"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc())
    except OSError:
        pass


def _clear_failure(batch_dir, key):
    """Remove the saved traceback of a job that has now succeeded.

    Parameters:
    batch_dir (str or Path): Batch directory.
    key (str): Job key.
    """
    try:
        (Path(batch_dir) / "failures" / f"{key}.txt").unlink(missing_ok=True)
    except OSError:
        pass


def run_batch(jobs, max_workers=None, batch_dir=BATCH_DIR, fn=neural_job, key_fn=_job_key,
              desc="fold x seed batch", csv_dir=None, log_name="run_log", shard=None):
    """Run jobs in a process pool and write one pickle per finished job.

    Resume: a job is skipped if its pickle exists and the stored fingerprint matches. A file written under other
    settings, inputs or code is recomputed. Jobs are submitted longest first (a cost proxy: neural before parametric), so
    the last workers do not idle on a long tail.

    Parameters:
    jobs (list): FoldJob objects (or dicts) with unique keys.
    max_workers (int): Number of workers. None calls resolve_max_workers here; pass an integer on a shared node.
    batch_dir (str or Path): Directory of the resume files. Default BATCH_DIR.
    fn (callable): Job function. Default neural_job. neural_job is run through neural_job_persist, which stores the
        fitted networks, unless THESIS_PERSIST_STATE=0.
    key_fn (callable): key_fn(job) -> resume key. Default _job_key.
    desc (str): Label of the progress bar and log messages.
    csv_dir (str or Path): If given, every finished job (and every job recovered from the resume files) is also written
        to CSV there at once. Default None.
    log_name (str): Name of the CSV run log. Default 'run_log'.
    shard (str): "i/n" (1-based; default: the THESIS_SHARD environment variable). Runs only every n-th job of the
        longest-first list, starting at the i-th, so n cluster tasks can share one batch through the resume files.

    Returns:
    dict: {key: result}.

    Raises:
    ValueError: If the job keys are not unique or the shard string is invalid.
    RuntimeError: If THESIS_RESUME_ONLY=1 and jobs are absent or stale, or if a job fails.
    IncompleteBatch: If a sharded call cannot return every result after its slice is done. An unsharded call
        afterwards assembles the finished files.
    """
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from tqdm.auto import tqdm
    shard = shard or SHARD
    if shard:
        import re
        match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", str(shard))
        if not match:
            raise ValueError(f"Invalid THESIS_SHARD={shard!r}: expected i/n with 1 <= i <= n")
        i, n = map(int, match.groups())
        if i > n:
            raise ValueError(f"Invalid THESIS_SHARD={shard!r}: i must not exceed n")
        # A separate log per shard avoids temporary-file collisions on shared storage.
        log_name = f"{log_name}_shard_{i}_of_{n}"
    batch_dir = Path(batch_dir); batch_dir.mkdir(parents=True, exist_ok=True)
    results, todo, csv_rows = {}, [], {}
    n_stale = 0
    fps = {}
    keyed_jobs = [(key_fn(j), j) for j in jobs]
    if len({k for k, _ in keyed_jobs}) != len(keyed_jobs):
        raise ValueError("run_batch received duplicate job keys")
    # The shard assignment is computed on the full job list before resume filtering. Ties are sorted by key, so it
    # is deterministic regardless of completion order.
    keyed_jobs.sort(key=lambda kj: (-_job_cost(kj[1]), kj[0]))
    assigned = set(k for k, _ in (keyed_jobs[i - 1::n] if shard else keyed_jobs))
    for k, j in keyed_jobs:
        p = batch_dir / f"{k}.pkl"
        fps[k] = _fingerprint(j)
        fp_file = batch_dir / f"{k}.fp"
        valid = p.exists() and fp_file.exists() and fp_file.read_text() == fps[k]
        if valid:
            try:
                with p.open("rb") as fh:
                    results[k] = pickle.load(fh)
            except (EOFError, pickle.UnpicklingError, OSError):
                valid = False
                results.pop(k, None)
        if valid:
            if csv_dir is not None and k in assigned:
                if not (Path(csv_dir) / f"step_{k}.csv").exists():
                    persist_job_csv(csv_rows, results[k], k, csv_dir, log_name)
                else:
                    csv_rows[k] = _scalar_row(results[k], k)
        elif k in assigned:
            if p.exists():
                n_stale += 1
            todo.append((k, j))
        # A stale file that belongs to another shard is never deleted, because that shard may be working on it.
    n_all = len(keyed_jobs)
    log.info(f"{desc}: {len(results)} valid resume results, {len(todo)} assigned jobs to run"
             + (f" (shard {shard} of {n_all} total)" if shard else "")
             + (f" ({n_stale} stale/corrupt assigned resume files)" if n_stale else ""))

    if os.environ.get("THESIS_RESUME_ONLY") == "1" and todo:
        raise RuntimeError(f"Resume-only assembly: {len(todo)} required {desc} jobs are absent or stale: "
                           + ", ".join(k for k, _ in todo[:12]))

    def finish(k, r):
        """Write the result of job k and its fingerprint to the batch directory, then log it.

        Parameters:
        k (str): Job key.
        r (dict): Result of the job.
        """
        import uuid
        unique = uuid.uuid4().hex
        p = batch_dir / f"{k}.pkl"
        tmp_p = batch_dir / f".{k}.{unique}.pkl.tmp"
        tmp_f = batch_dir / f".{k}.{unique}.fp.tmp"
        try:
            with tmp_p.open("wb") as fh:
                pickle.dump(r, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush(); os.fsync(fh.fileno())
            tmp_f.write_text(fps[k])
            # The result is committed first and the fingerprint last, because the fingerprint is the validity marker.
            os.replace(tmp_p, p)
            os.replace(tmp_f, batch_dir / f"{k}.fp")
        finally:
            tmp_p.unlink(missing_ok=True)
            tmp_f.unlink(missing_ok=True)
        results[k] = r
        if csv_dir is not None:
            persist_job_csv(csv_rows, r, k, csv_dir, log_name)

    if todo:
        # All independent pools obey BOTH CPU/task limits and a cgroup RAM budget.
        # These are conservative initial per-worker estimates, not observed peak RSS.
        is_neural = fn in (neural_job, wd_tuning_job) or getattr(fn, "_ram_heavy", False)
        if is_neural:
            worker_gib = float(os.getenv('THESIS_NEURAL_WORKER_GIB', '1.5'))
        elif fn is nonparametric_job:
            worker_gib = float(os.getenv('THESIS_NONPARAM_WORKER_GIB', '2.5'))
        else:
            worker_gib = float(os.getenv('THESIS_OTHER_WORKER_GIB', '1.5'))
        cpu_nw = resolve_max_workers(n_tasks=len(todo))
        nw = resolve_pool_workers(n_tasks=len(todo), max_workers=max_workers, worker_gib=worker_gib)
        log.info(f"{desc}: {nw} workers (CPU/task limit {cpu_nw}, estimated {worker_gib:g} GiB each; "
                 "RAM headroom checked at pool creation)")
        run_fn = fn
        if fn is neural_job and os.environ.get("THESIS_PERSIST_STATE", "1") != "0":
            import functools
            run_fn = functools.partial(neural_job_persist, state_dir=str(batch_dir / "states"))
        if getattr(fn, "_persist_state", False) and os.environ.get("THESIS_PERSIST_STATE", "1") != "0":
            import functools
            run_fn = functools.partial(fn, state_dir=str(batch_dir / "states"))
        if nw <= 1:
            for k, j in tqdm(todo, desc=desc):
                try:
                    r_ = run_fn(j)
                except Exception:
                    _record_failure(batch_dir, k); raise
                _clear_failure(batch_dir, k)
                finish(k, r_)
        else:
            # Only a small window of jobs is submitted, rather than keeping every job and future in memory.
            os.environ.setdefault("THESIS_LOG_DIR", str(batch_dir / "logs"))
            with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as ex:
                iterator = iter(todo)
                pending = {}
                def fill():
                    """Submit jobs until 2 x workers are pending."""
                    while len(pending) < nw * 2:
                        try:
                            k, j = next(iterator)
                        except StopIteration:
                            break
                        pending[ex.submit(run_fn, j)] = k
                fill()
                with tqdm(total=len(todo), desc=desc) as progress:
                    while pending:
                        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                        for fut in done:
                            k = pending.pop(fut)
                            try:
                                r_ = fut.result()
                            except Exception:
                                _record_failure(batch_dir, k); raise
                            _clear_failure(batch_dir, k)
                            finish(k, r_)
                            progress.update(1)
                        fill()
    if shard and len(results) < len(jobs):
        raise IncompleteBatch(f"{desc}: shard {shard} finished its fixed assignment; "
                              f"{len(jobs) - len(results)} of {len(jobs)} results not visible yet. "
                              "Run resume-only assembly when ALL shards have finished.")
    return results


# ── gates and the loss store ───────────────────────────────────────────────
def display_binary_name(low, high):
    """Display name of a binary (low/high regime) candidate.

    Parameters:
    low (str): Display name of the low-regime family.
    high (str): Display name of the high-regime family.

    Returns:
    str: 'Divergence split: <low> (low) / <high> (high)'.
    """
    return f"Divergence split: {low} (low) / {high} (high)"


# Canonical names of the 76 candidates. Parametric rows use the display names of copula_families
# (r['name'] of fit_and_evaluate), Acar rows the FAMILY_INFO key of conditional_copula.
ACAR_FAMILIES = tuple("student" if k == "t" else k for k in FAMILY_KEYS)       # FAMILY_INFO keys


def unconditional_name(display):
    """Display name of an unconditional parametric candidate.

    Parameters:
    display (str): Display name of the family.

    Returns:
    str: 'Unconditional: <display>'.
    """
    return f"Unconditional: {display}"


def acar_name(fam):
    """Display name of an Acar candidate.

    Parameters:
    fam (str): Acar family key.

    Returns:
    str: 'Acar (continuous): <fam>'.
    """
    return f"Acar (continuous): {fam}"


def candidate_table(nonparametric=True):
    """The 76 candidates in table order (74 without the nonparametric ones). Pure bookkeeping, no results needed.

    Parameters:
    nonparametric (bool): Include the two nonparametric estimators. Default True.

    Returns:
    pd.DataFrame: Columns model and group ('Unconditional', 'Binary', 'Acar', 'Nonparametric' or 'Neural').
    """
    rows = [(unconditional_name(FAMILY_DISPLAY[k]), "Unconditional") for k in FAMILY_KEYS]
    rows.append((UNCOND_CODINE, "Unconditional"))
    rows += [(display_binary_name(FAMILY_DISPLAY[a], FAMILY_DISPLAY[b]), "Binary") for a in FAMILY_KEYS for b in FAMILY_KEYS]
    rows.append((SPLIT_CODINE, "Binary"))
    rows += [(acar_name(f), "Acar") for f in ACAR_FAMILIES]
    if nonparametric:
        rows += [(m, "Nonparametric") for m in NONPARAMETRIC_MODELS]
    rows += [(m, "Neural") for m in NEURAL_MODELS]
    df = pd.DataFrame(rows, columns=["model", "group"])
    expected = GC.N_CANDIDATES if nonparametric else GC.N_CANDIDATES_SYNTHETIC
    assert len(df) == expected and df.model.is_unique, (len(df), expected)
    return df


def _seed_mean(arrs):
    """Mean over seeds of per-step loss arrays.

    Rows outside the test window are NaN in every seed and stay NaN.

    Parameters:
    arrs (list): Equal-length arrays, one per seed.

    Returns:
    np.ndarray: Mean per step, or None if the list is empty.
    """
    if not len(arrs):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # rows outside the test window are NaN in every seed
        return np.nanmean(np.vstack(arrs), axis=0)


def neural_results_to_losses(neural, folds=FOLDS, seeds=SEEDS):
    """Seed-averaged loss of each neural model per fold, with gate bookkeeping.

    All available seeds enter each average. The marginal-uniformity distance of each seed feeds the gate in gate_table,
    which excludes a model entirely if any seed or fold fails, but no individual seed is dropped from the average. The
    divergence-split CODINE loss takes the low-regime network's loss where it is finite and the high-regime network's
    elsewhere.

    Parameters:
    neural (dict): {key: neural_job result}.
    folds (sequence of str): Folds. Default FOLDS.
    seeds (sequence of int): Seeds. Default SEEDS.

    Returns:
    tuple: (loss, gate_rows, seed_means). loss[model][fold] is a (n,) array or None, gate_rows is a DataFrame (model,
        fold, n_seeds, n_pass, admissible, passing_seeds), and seed_means[model][fold] is the list of per-seed mean test
        losses (for the seed SD).
    """
    loss, rows, seed_means = {}, [], {}
    models = [UNCOND_CODINE, SPLIT_CODINE] + NEURAL_MODELS
    for m in models:
        loss[m] = {}; seed_means[m] = {}
        for f in folds:
            per_seed, used = [], []
            for s in seeds:
                if m == SPLIT_CODINE:
                    lo = neural.get(net_key(f, s, f"{SPLIT_CODINE}|low")); hi = neural.get(net_key(f, s, f"{SPLIT_CODINE}|high"))
                    if lo is None or hi is None:
                        continue
                    L = np.where(np.isfinite(lo["loss"]), lo["loss"], hi["loss"])
                    per_seed.append(L); used.append(s)
                else:
                    r = neural.get(net_key(f, s, m))
                    if r is not None:
                        per_seed.append(r["loss"]); used.append(s)
            # All available seeds contribute to the average. No goodness-of-fit diagnostic excludes a seed.
            passing = sorted(used)
            rows.append(dict(model=m, fold=f, n_seeds=len(seeds), n_pass=len(passing),
                             admissible=bool(passing), passing_seeds=tuple(passing)))
            loss[m][f] = _seed_mean(per_seed) if per_seed else None
            seed_means[m][f] = [float(np.nanmean(x)) for x in per_seed]
    return loss, pd.DataFrame(rows), seed_means


def gate_table(param, acar, neural, folds=FOLDS, seeds=SEEDS, nonparam=None):
    """Marginal-uniformity gate of all 76 candidates.

    Every candidate must satisfy the marginal KS tolerance in each fold (for the neural models in each seed and regime).
    Parametric and Acar copulas have analytically uniform margins, so their simulation check tests the implementation.
    Missing or non-finite KS values fail closed. The threshold depends on the check: n_tests = 2 for one simulated copula
    (two margins), 14 for the seven covariate percentiles of the Acar and conditional networks, and grid=True for the
    CODINE grid sampler. gate_config.ks_threshold aligns them (0.07 for the two-test exact check).

    Parameters:
    param (dict): {fold: parametric job result}.
    acar (dict): {(fold, family): Acar job result}.
    neural (dict): {key: neural_job result}.
    folds (sequence of str): Folds. Default FOLDS.
    seeds (sequence of int): Network seeds that must pass. Default SEEDS.
    nonparam (dict or None): {(fold, kind): nonparametric job result}. Default None: the table then has the 74 candidates without the
        nonparametric ones.

    Returns:
    pd.DataFrame: Indexed by model, with group, threshold, max_ks, has_missing_ks, in_pool, reason and ks_<fold>.

    Raises:
    AssertionError: If the number of candidates is not gate_config.N_CANDIDATES (76; 74 without nonparam) or the names are not unique.
    """
    rows = []
    def row(model, group, diagnostics, n_tests=2, grid=False):
        """Gate row of one candidate from its {label: D_max} diagnostics.

        Parameters:
        model (str): Model name.
        group (str): Model group.
        diagnostics (dict): D_max of each check, keyed by label.
        n_tests (int): Number of KS tests behind each D_max. Default 2.
        grid (bool): True if the margins were checked with the grid sampler. Default False.

        Returns:
        dict: The gate row (threshold, largest D_max, pool membership, reason and D_max per fold).
        """
        thr = GC.ks_threshold(n_tests, grid)
        failures = [(f, d) for f, d in diagnostics.items() if not GC.ks_seed_pass(d, n_tests, grid)]
        out = dict(model=model, group=group, threshold=thr,
                   max_ks=max((float(v) for v in diagnostics.values() if v is not None and np.isfinite(v)), default=np.nan),
                   has_missing_ks=any(v is None or not np.isfinite(v) for v in diagnostics.values()),
                   in_pool=not failures,
                   reason="; ".join(f"{key}: KS missing or > {thr:.4f}" for key, _ in failures))
        out.update({f"ks_{f}": max((float(v) for k, v in diagnostics.items() if k.startswith(f + '/') and v is not None and np.isfinite(v)), default=np.nan) for f in folds})
        return out
    for key in FAMILY_KEYS:
        d = FAMILY_DISPLAY[key]
        rows.append(row(unconditional_name(d), "Unconditional", {
            f + '/param': param.get(f, {}).get('ks', {}).get(('uncond', d)) for f in folds}))
    def nks(fold, seed, model):
        """D_max of one network, or None if its result is missing.

        Parameters:
        fold (str): Fold name.
        seed (int): Training seed.
        model (str): Model name.

        Returns:
        float: The stored D_max, or None.
        """
        result = neural.get(net_key(fold, seed, model))
        return None if result is None else result.get('ks')
    def neural_row(model, group, regimes=(None,)):
        """Gate row of a neural candidate over all folds, seeds and regimes.

        Parameters:
        model (str): Model name.
        group (str): Model group.
        regimes (tuple): Regimes to include; None stands for the model without a regime. Default (None,).

        Returns:
        dict: The gate row, see row.
        """
        conditional = model in NEURAL_MODELS
        return row(model, group, {
            f'{f}/{reg or "all"}/seed{s}': nks(f, s, model if reg is None else f'{model}|{reg}')
            for f in folds for s in seeds for reg in regimes},
            n_tests=(2 * len(GC.KS_PERCENTILES) if conditional else 2),
            grid=(model in CCODINE_VARIANTS or model in (UNCOND_CODINE, SPLIT_CODINE)))
    rows.append(neural_row(UNCOND_CODINE, "Unconditional"))
    for kl in FAMILY_KEYS:
        for kh in FAMILY_KEYS:
            dl, dh = FAMILY_DISPLAY[kl], FAMILY_DISPLAY[kh]
            rows.append(row(display_binary_name(dl, dh), "Binary", {
                f + '/' + reg: param.get(f, {}).get('ks', {}).get((reg, d))
                for f in folds for reg, d in [('low', dl), ('high', dh)]}))
    rows.append(neural_row(SPLIT_CODINE, "Binary", regimes=('low', 'high')))
    for fam in ACAR_FAMILIES:
        rows.append(row(acar_name(fam), "Acar", {
            f + '/acar': (acar.get((f, fam)) or {}).get('ks') for f in folds},
            n_tests=2 * len(GC.KS_PERCENTILES)))
    if nonparam is not None:
        for model in NONPARAMETRIC_MODELS:
            rows.append(row(model, "Nonparametric", {
                f + '/nonparametric': (nonparam.get((f, NONPARAMETRIC_KIND[model])) or {}).get('ks') for f in folds},
                n_tests=2 * len(GC.KS_PERCENTILES)))
    for model in NEURAL_MODELS:
        rows.append(neural_row(model, "Neural"))
    gt = pd.DataFrame(rows)
    expected = GC.N_CANDIDATES if nonparam is not None else GC.N_CANDIDATES_SYNTHETIC
    if len(gt) != expected or not gt.model.is_unique:
        raise AssertionError(f"Expected {expected} distinct candidates; got {len(gt)}")
    return gt.set_index('model')


def pool_at_threshold(gt, d_max):
    """Candidates that would pass the marginal-uniformity gate if the reference threshold were d_max.

    The applied threshold of a row is KS_D_MAX plus its own alignment term (14 tests, grid sampler). The alternative
    threshold keeps that term: thr_alt = threshold - KS_D_MAX + d_max. A candidate whose largest D_max is missing fails
    closed, and a candidate with any missing required fold, seed or regime check stays excluded at every threshold.

    Parameters:
    gt (pd.DataFrame): Output of gate_table.
    d_max (float): Alternative reference threshold.

    Returns:
    list: Names of the candidates that pass.
    """
    thr = gt["threshold"].to_numpy(float) - GC.KS_D_MAX + float(d_max)
    ok = (np.isfinite(gt["max_ks"].to_numpy(float))
          & (gt["max_ks"].to_numpy(float) <= thr)
          & ~gt["has_missing_ks"].to_numpy(bool))
    return list(gt.index[ok])


def pool_size_table(gt, levels=None):
    """Pool size by candidate group at each reference threshold.

    Parameters:
    gt (pd.DataFrame): Output of gate_table.
    levels (sequence of float): Reference thresholds. Default gate_config.KS_SENSITIVITY_LEVELS (0.05, 0.07, 0.10).

    Returns:
    pd.DataFrame: Rows are groups plus 'Total', columns are the thresholds and 'candidates'.
    """
    levels = tuple(levels or GC.KS_SENSITIVITY_LEVELS)
    out = {}
    for d in levels:
        keep = set(pool_at_threshold(gt, d))
        out[f"D_max <= {d:.2f}"] = gt.assign(_in=gt.index.isin(keep)).groupby("group")["_in"].sum().astype(int)
    tab = pd.DataFrame(out).fillna(0).astype(int)
    tab.loc["Total"] = tab.sum()
    tab["candidates"] = list(gt.groupby("group").size().reindex(tab.index[:-1]).astype(int)) + [len(gt)]
    return tab


def training_stability_table(neural, max_epochs=None):
    """Stability check of every trained network.

    A network is stable if its best validation CVPE is finite and training stopped before the epoch cap. The test CVPE is
    never used. If any CODINE network (unconditional or per-regime) is unstable under the shared recipe, the comparison
    of CODINE with the conditional networks is confounded by the training recipe and should not be read as a
    conditioning effect.

    Parameters:
    neural (dict): {key: neural_job result}.
    max_epochs (int): Epoch cap. Default default_hyper()['max_epochs'].

    Returns:
    pd.DataFrame: One row per network with key, model, fold, seed, regime, best_epoch, stopped_epoch, best_val_cvpe and
        stable.
    """
    cap = int(max_epochs or default_hyper()["max_epochs"])
    rows = []
    for key, r in neural.items():
        if r is None or "best_epoch" not in r:
            continue
        bv = r.get("best_val_cvpe")
        stopped = r.get("stopped_epoch")
        ok = (bv is not None and np.isfinite(bv) and r.get("best_epoch") is not None
              and stopped is not None and int(stopped) < cap)
        rows.append(dict(key=key, model=r.get("model"), fold=r.get("fold"), seed=r.get("seed"),
                         regime=r.get("regime"), best_epoch=r.get("best_epoch"), stopped_epoch=stopped,
                         best_val_cvpe=bv, stable=bool(ok)))
    return pd.DataFrame(rows)


def assemble_store(param, acar, neural, folds=FOLDS, seeds=SEEDS, n=None, gates=None, nonparam=None):
    """Collect every candidate's per-fold per-step loss into one tensor.

    Models are listed whether or not they pass the gates; in_pool says which enter the MCS.

    Parameters:
    param (dict): {fold: parametric job result}.
    acar (dict): {(fold, family): Acar job result}.
    neural (dict): {key: neural_job result}.
    folds (sequence of str): Folds. Default FOLDS.
    seeds (sequence of int): Seeds. Default SEEDS.
    n (int): Number of steps. Default: taken from the parametric results.
    gates (pd.DataFrame): Precomputed gate_table. Default: computed here.
    nonparam (dict or None): {(fold, kind): nonparametric job result}. Default None (74 candidates).

    Returns:
    dict: tensor (n, M, F), models, folds, group {model: group}, in_pool {model: bool}, gate_table, neural_gate_rows,
        seed_sd {model: {fold: SD over seeds of the mean test loss}}, selected_pair {fold: (low, high)} and no_loss
        {model: reason} for candidates without a loss in every fold.

    Raises:
    RuntimeError: If a candidate passes the gates but has no loss in some fold, since that would silently shrink the pool.
    """
    n = n or len(next(iter(next(iter(param.values()))["loss"].values())))
    cols = {}                                   # model -> {fold: (n,) array}
    group = {}
    for f in folds:
        for name, L in param[f]["loss"].items():
            if name.startswith("Uncond:"):
                m = unconditional_name(name.split(":", 1)[1]); group[m] = "Unconditional"
            else:
                lo, hi = name.split(":", 1)[1].split("|"); m = display_binary_name(lo, hi); group[m] = "Binary"
            cols.setdefault(m, {})[f] = L
    for (f, fam), r in acar.items():
        m = acar_name(fam); group[m] = "Acar"
        if r.get("loss") is not None:
            cols.setdefault(m, {})[f] = r["loss"]
    for (f, kind), r in (nonparam or {}).items():
        m = r["model"]; group[m] = "Nonparametric"
        if r.get("loss") is not None:
            cols.setdefault(m, {})[f] = r["loss"]
    n_loss, n_gate_rows, seed_means = neural_results_to_losses(neural, folds, seeds)
    for m, per_fold in n_loss.items():
        group[m] = ("Unconditional" if m == UNCOND_CODINE else "Binary" if m == SPLIT_CODINE else "Neural")
        for f, L in per_fold.items():
            if L is not None:
                cols.setdefault(m, {})[f] = L
    gt = gates if gates is not None else gate_table(param, acar, neural, folds, seeds, nonparam=nonparam)
    models = [m for m in cols if all(f in cols[m] for f in folds)]
    no_loss = {m: "no loss in fold(s) " + ",".join(f for f in folds if f not in cols.get(m, {}))
               for m in gt.index if m not in models}
    in_pool = {m: bool(gt.loc[m, "in_pool"]) if m in gt.index else False for m in models}
    stray = [m for m in gt.index if bool(gt.loc[m, "in_pool"]) and m not in models]
    if stray:
        raise RuntimeError(f"{len(stray)} candidates pass every gate but have no loss in every fold: {stray[:5]}")
    tensor = np.full((n, len(models), len(folds)), np.nan)
    for j, m in enumerate(models):
        for k, f in enumerate(folds):
            tensor[:, j, k] = cols[m][f]
    seed_sd = {m: {f: (float(np.std(v, ddof=1)) if len(v) > 1 else np.nan) for f, v in seed_means[m].items()}
               for m in seed_means}
    return dict(tensor=tensor, models=models, folds=list(folds), group={m: group[m] for m in models},
                in_pool=in_pool, gate_table=gt, neural_gate_rows=n_gate_rows, seed_sd=seed_sd,
                selected_pair={f: param[f]["selected_pair"] for f in folds}, no_loss=no_loss)


# ── per-step series, CVPE tables, MCS, joint upper region ──────────────────
def per_step_loss(store, only_pool=True, models=None):
    """Per-step loss, averaged over the folds that test the step.

    Parameters:
    store (dict): Output of assemble_store.
    only_pool (bool): Restrict to the models in the pool. Default True.
    models (list): Explicit model list (overrides only_pool). Default None.

    Returns:
    tuple: (L, models), with L of shape (n, M) as the nanmean over the fold axis.

    Raises:
    ValueError: If any step is untested for any model, which would silently misalign the MCS.
    """
    ms = models or [m for m in store["models"] if (store["in_pool"][m] or not only_pool)]
    idx = [store["models"].index(m) for m in ms]
    _assert_same_nan_pattern(store["tensor"][:, idx, :], ms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        L = np.nanmean(store["tensor"][:, idx, :], axis=2)
    if not np.isfinite(L).all():
        raise ValueError("per-step loss has NaN: some steps are untested for some models")
    return L, ms


def _assert_same_nan_pattern(T, ms):
    """Check that every model is tested by the same folds at every step.

    Each fold scores a step on its own PIT scale, so a model whose loss at a step comes from one fold while the others'
    comes from two would be compared across scales.

    Parameters:
    T (np.ndarray): (n, M, F) slice of the loss tensor.
    ms (list): Model names, one per column.

    Raises:
    ValueError: Listing the models whose NaN pattern differs from the first model's.
    """
    pat = np.isnan(T)
    bad = [ms[j] for j in range(1, pat.shape[1]) if not np.array_equal(pat[:, j, :], pat[:, 0, :])]
    if bad:
        raise ValueError(f"{len(bad)} model(s) have a different tested-fold pattern than {ms[0]!r} "
                         f"(NaN losses inside the test windows): {bad[:5]}")


def per_fold_loss(store, fold, models=None, only_pool=True):
    """Per-step loss of one fold on its own test window.

    One PIT scale, nothing averaged across folds and no stitching. Usable only when the fold's tested steps form one
    contiguous stretch (A1, A2, B1). B2 tests two separated outer quarters, and joining them would put an artificial
    junction into the bootstrap.

    Parameters:
    store (dict): Output of assemble_store.
    fold (str): Fold name.
    models (list): Explicit model list. Default: the pool (or all models if only_pool is False).
    only_pool (bool): Restrict to the models in the pool. Default True.

    Returns:
    tuple: (L of shape (n_test, M), models, positions of the test steps).

    Raises:
    ValueError: If the fold has no tested step or its test steps are not contiguous.
    """
    ms = models or [m for m in store["models"] if (store["in_pool"][m] or not only_pool)]
    idx = [store["models"].index(m) for m in ms]
    k = store["folds"].index(fold)
    T = store["tensor"][:, idx, k]
    _assert_same_nan_pattern(T[:, :, None], ms)
    keep = np.isfinite(T).all(axis=1)
    pos = np.flatnonzero(keep)
    if not len(pos):
        raise ValueError(f"fold {fold}: no tested step")
    if not np.array_equal(pos, np.arange(pos[0], pos[-1] + 1)):
        raise ValueError(f"fold {fold}: the test steps are not one contiguous stretch")
    return T[keep], ms, pos


def fold_cvpe_table(store, models=None):
    """Descriptive per-fold CVPE (mean over each fold's test steps) for every model.

    The spread across folds is not a standard error, because the steps in the overlap of two test blocks are tested twice.

    Parameters:
    store (dict): Output of assemble_store.
    models (list): Models to include. Default all.

    Returns:
    pd.DataFrame: One row per model, with a column per fold, 'pooled' (mean over steps of the fold-averaged loss), 'group'
        and 'in_pool'.
    """
    ms = models or store["models"]
    rows = {}
    for m in ms:
        j = store["models"].index(m)
        rows[m] = {f: float(np.nanmean(store["tensor"][:, j, k])) for k, f in enumerate(store["folds"])}
    df = pd.DataFrame(rows).T
    df["pooled"] = [float(np.nanmean(np.nanmean(store["tensor"][:, store["models"].index(m), :], axis=1))) for m in df.index]
    df["group"] = [store["group"][m] for m in df.index]
    df["in_pool"] = [store["in_pool"][m] for m in df.index]
    return df


def acf(x, nlags=250):
    """Sample autocorrelation function.

    Parameters:
    x (array-like): Series.
    nlags (int): Largest lag. Default 250.

    Returns:
    np.ndarray: Autocorrelations at lags 0 to nlags (divisor n, as in the standard estimator).
    """
    x = np.asarray(x, float) - np.mean(x)
    n = len(x); v = np.dot(x, x)
    return np.array([np.dot(x[:n - k], x[k:]) / v for k in range(nlags + 1)])


def differential_diagnostics(L, models, nlags=250, n_pairs_cap=60, seed=0):
    """Autocorrelation and block length of the loss differentials of the best model against the others.

    The best model has the lowest mean loss. The block length is the Politis-White plug-in estimate (Politis & White, 2004;
    Patton et al., 2009) on each differential, and its median across pairs is the diagnostic that decides the MCS block
    length.

    Parameters:
    L (np.ndarray): (n, M) per-step loss.
    models (list): Model names.
    nlags (int): Largest lag of the ACF. Default 250.
    n_pairs_cap (int): At most this many comparison models are drawn at random. Default 60.
    seed (int): Seed of that draw. Default 0.

    Returns:
    dict: best (name), acf_mean, acf_all, block_length_median and block_lengths.
    """
    from data_split import block_length_joint
    best = int(np.argmin(L.mean(axis=0)))
    others = [j for j in range(L.shape[1]) if j != best]
    rng = np.random.default_rng(seed)
    if len(others) > n_pairs_cap:
        others = list(rng.choice(others, n_pairs_cap, replace=False))
    acfs, bls = [], []
    for j in others:
        d = L[:, j] - L[:, best]
        if np.std(d) < 1e-12:
            continue
        acfs.append(acf(d, nlags))
        try:
            bls.append(block_length_joint(d))
        except Exception:
            pass
    return dict(best=models[best], acf_mean=np.mean(acfs, axis=0) if acfs else None,
                acf_all=np.array(acfs), block_length_median=float(np.median(bls)) if bls else np.nan,
                block_lengths=bls)


def mcs_table(L, models, block_lens, alphas=(0.10, 0.05, 0.25), n_boot=5000, seed=0, method="R"):
    """MCS membership for every combination of block length and level.

    One bootstrap is run per block length, because the p-values do not depend on alpha (run_mcs reseeds Numba before
    compute); alpha only thresholds them. The Model Confidence Set is that of Hansen et al. (2011).

    Parameters:
    L (np.ndarray): (n, M) per-step loss.
    models (list): Model names.
    block_lens (sequence of int): Mean block lengths of the stationary bootstrap.
    alphas (sequence of float): MCS levels. Default (0.10, 0.05, 0.25).
    n_boot (int): Bootstrap replicates. Default 5000.
    seed (int): Seed. Default 0.
    method (str): Test statistic of run_mcs. Default 'R'.

    Returns:
    tuple: (summary DataFrame indexed by model with columns p@b<L> and in@b<L>@a<alpha>, dict of the full results by
        (block length, alpha)).

    Raises:
    ValueError: If the MCS returns invalid p-values.
    RuntimeError: If the package's inclusion status disagrees with the p-value threshold at the smallest level.
    """
    from model_evaluation import run_mcs
    out = pd.DataFrame(index=models)
    raw = {}
    for bl in block_lens:
        levels = sorted(set(alphas))
        mcs = run_mcs(L, models, block_len=bl, alpha=levels[0], n_boot=n_boot, seed=seed, method=method)
        res = mcs.results()
        pvalues = res["pvalues"].reindex(models)
        if pvalues.isna().any():
            raise ValueError(f"MCS did not return valid p-values for all models at block length {bl}")
        # The MCS p-values do not depend on alpha. The package's own inclusion status at the smallest level is compared
        # with the p-value threshold, to guard against changes of the package.
        ref_status = (res["status"].astype(str).str.strip().str.lower() == "included").reindex(models)
        if not np.array_equal(ref_status.to_numpy(), (pvalues >= levels[0]).to_numpy()):
            raise RuntimeError("MCS package status does not match p-value threshold; review package API")
        out[f"p@b{bl}"] = pvalues
        for a in levels:
            included = (pvalues >= a)
            out[f"in@b{bl}@a{a:g}"] = included
            result_at_a = res.copy()
            result_at_a["status"] = np.where(
                result_at_a["pvalues"] >= a, "Included", "Excluded")
            raw[(bl, a)] = result_at_a
    return out, raw


def mcs_seed_stability(L, models, block_len, alphas=(0.10, 0.05, 0.25), seeds=(0, 1, 2), n_boot=5000):
    """Survivor sets of the MCS under different bootstrap seeds.

    Parameters:
    L (np.ndarray): (n, M) per-step loss.
    models (list): Model names.
    block_len (int): Mean block length.
    alphas (sequence of float): MCS levels. Default (0.10, 0.05, 0.25).
    seeds (sequence of int): Bootstrap seeds. Default (0, 1, 2).
    n_boot (int): Bootstrap replicates. Default 5000.

    Returns:
    pd.DataFrame: One row per seed and level, with the sorted list of surviving models.
    """
    rows = []
    for s in seeds:
        tab, _ = mcs_table(L, models, (block_len,), alphas, n_boot=n_boot, seed=s)
        for a in alphas:
            rows.append(dict(seed=s, alpha=a,
                             survivors=sorted(tab.index[tab[f"in@b{block_len}@a{a:g}"].astype(bool)])))
    return pd.DataFrame(rows)


def joint_top_cvpe(store, arrays_by_fold, models=None, q=0.80):
    """CVPE on the joint upper region {u1 >= q-quantile, u2 >= q-quantile}.

    The region is defined per fold from that fold's test pseudo-observations (u differs across folds because the PIT is
    refit) and the fold values are combined with fold-share weights (fold test size over total test size). Descriptive,
    no confidence interval.

    Parameters:
    store (dict): Output of assemble_store.
    arrays_by_fold (dict): {fold: fold_arrays}.
    models (list): Models to include. Default all.
    q (float): Quantile defining the region. Default 0.80.

    Returns:
    pd.DataFrame: Per-fold CVPE in the region and the weighted mean, one row per model. The number of steps per fold
        is in df.attrs['n_per_fold'].
    """
    ms = models or store["models"]
    folds = store["folds"]
    rows, ns, w = {}, {}, {}
    tot = sum(int(arrays_by_fold[f]["test"].sum()) for f in folds)
    for k, f in enumerate(folds):
        A = arrays_by_fold[f]; te = A["test"]
        reg = te & (A["u1"] >= np.quantile(A["u1"][te], q)) & (A["u2"] >= np.quantile(A["u2"][te], q))
        ns[f] = int(reg.sum()); w[f] = int(te.sum()) / tot
        for m in ms:
            j = store["models"].index(m)
            rows.setdefault(m, {})[f] = float(np.nanmean(store["tensor"][reg, j, k])) if reg.any() else np.nan
    df = pd.DataFrame(rows).T
    wvec = np.array([w[f] for f in folds])
    df["weighted"] = (df[folds].values * wvec[None, :]).sum(axis=1) / wvec.sum()
    df.attrs["n_per_fold"] = ns
    return df


# ── seed-averaged joint-exceedance probabilities ───────────────────────────
def collect_exceed_probs(param, acar, neural, q, folds=FOLDS, seeds=SEEDS, nonparam=None):
    """Joint-exceedance probability of every model on each fold's test steps.

    Parametric and Acar values come straight from the jobs. Neural values are averaged over all available seeds, matching
    the loss average, and the KS diagnostics do not filter seeds.

    Parameters:
    param (dict): {fold: parametric job result}.
    acar (dict): {(fold, family): Acar job result}.
    neural (dict): {key: neural_job result}.
    q (float): Threshold, one of Q_GRID.
    folds (sequence of str): Folds. Default FOLDS.
    seeds (sequence of int): Seeds. Default SEEDS.
    nonparam (dict or None): {(fold, kind): nonparametric job result}. Default None.

    Returns:
    dict: {fold: {model: (n,) array of P(U1 > q, U2 > q | w_t)}}.
    """
    out = {f: {} for f in folds}
    for f in folds:
        for name, p in param[f]["p_step"][q].items():
            if name.startswith("Uncond:"):
                m = unconditional_name(name.split(':', 1)[1])
            else:
                lo, hi = name.split(":", 1)[1].split("|"); m = display_binary_name(lo, hi)
            out[f][m] = p
        for (ff, fam), r in acar.items():
            if ff == f:
                out[f][acar_name(fam)] = r["p_step"][q]
        for (ff, kind), r in (nonparam or {}).items():
            if ff == f:
                out[f][r["model"]] = r["p_step"][q]
        for m in [UNCOND_CODINE, SPLIT_CODINE] + NEURAL_MODELS:
            ps = []
            for s in seeds:
                if m == SPLIT_CODINE:
                    lo = neural.get(net_key(f, s, f"{SPLIT_CODINE}|low")); hi = neural.get(net_key(f, s, f"{SPLIT_CODINE}|high"))
                    if lo is None or hi is None:
                        continue
                    ps.append(np.where(np.isfinite(lo["p_step"][q]), lo["p_step"][q], hi["p_step"][q]))
                else:
                    r = neural.get(net_key(f, s, m))
                    if r is None:
                        continue
                    ps.append(r["p_step"][q])
            if ps:
                out[f][m] = np.nanmean(np.vstack(ps), axis=0)
    return out


# ── job builders for the parametric and Acar stages ────────────────────────
def param_key(j):
    """Resume key of a parametric job.

    Parameters:
    j (FoldJob or dict): Job with fold and pit.

    Returns:
    str: 'param__<fold>__<pit>'.
    """
    return f"param__{j['fold']}__{j['pit']}"


def acar_key(j):
    """Resume key of an Acar job.

    Parameters:
    j (FoldJob or dict): Job with fold and family.

    Returns:
    str: 'acar__<fold>__<family>'.
    """
    return f"acar__{j['fold']}__{j['family']}"


def nonparam_key(j):
    """Resume key of a nonparametric job.

    Parameters:
    j (FoldJob or dict): Job with fold and kind.

    Returns:
    str: 'nonparam__<fold>__<kind>'.
    """
    return f"nonparam__{j['fold']}__{j['kind']}"


def build_nonparametric_jobs(arrays_by_fold, folds=FOLDS, smoke=False):
    """Nonparametric jobs: the Gijbels et al. (2011) and the ECBC estimator on every fold.

    Parameters:
    arrays_by_fold (dict): {fold: fold_arrays}.
    folds (sequence of str): Folds. Default FOLDS.
    smoke (bool): Use K = 32 cells and 21 covariate grid points. Default False.

    Returns:
    list: FoldJob objects.
    """
    hyper = dict(K=32, G=21) if smoke else {}
    return [FoldJob(fold=f, arrays=arrays_by_fold[f], kind=kind, hyper=dict(hyper))
            for f in folds for kind in ("gijbels", "ecbc")]


def run_param_stage(arrays_by_fold, max_workers=None, csv_dir=None):
    """Parametric fits, marginal KS distances and per-step losses for every fold.

    One job per fold and PIT variant. Notebooks 01, 02 and 04 all call this with the same arguments, so whichever runs
    first computes the stage and the others resume from the same files.

    Parameters:
    arrays_by_fold (dict): {fold: fold_arrays}.
    max_workers (int): Number of workers. Default None.
    csv_dir (str or Path): Optional CSV directory, see run_batch.

    Returns:
    tuple: ({fold: result of the kernel-PIT job}, {fold: result of the unconditional-PIT ablation job}).
    """
    res = run_batch(build_param_jobs(arrays_by_fold), max_workers=max_workers,
                    batch_dir=BATCH_DIR / "param", fn=parametric_fold_job, key_fn=param_key,
                    desc="parametric folds", csv_dir=csv_dir)
    return ({f: res[f"param__{f}__kernel"] for f in FOLDS}, {f: res[f"param__{f}__uncond"] for f in FOLDS})


def build_param_jobs(arrays_by_fold, folds=FOLDS):
    """Parametric jobs: per fold, a kernel-PIT run (the model run) and an unconditional-PIT run (the ablation).

    The ablation compares losses and survivor sets with the model run.

    Parameters:
    arrays_by_fold (dict): {fold: fold_arrays}.
    folds (sequence of str): Folds. Default FOLDS.

    Returns:
    list: FoldJob objects.
    """
    jobs = []
    for f in folds:
        jobs.append(FoldJob(fold=f, arrays=arrays_by_fold[f], kind="param", pit="kernel"))
        jobs.append(FoldJob(fold=f, arrays=arrays_by_fold[f], kind="param", pit="uncond"))
    return jobs


def build_acar_jobs(arrays_by_fold, h_by_fold_family, folds=FOLDS, n_eval=GC.ACAR_N_EVAL, smoke=False):
    """Acar jobs, using only bandwidths selected on each fold's own training data.

    The nested structure is intentional: a flat family-to-h dictionary is rejected rather than silently reintroducing
    the tuning leakage of using the A1 bandwidth for all folds.

    Parameters:
    arrays_by_fold (dict): {fold: fold_arrays}.
    h_by_fold_family (dict): {fold: {family: bandwidth}}, with every family of ACAR_FAMILIES for every fold.
    folds (sequence of str): Folds. Default FOLDS.
    n_eval (int): Evaluation points of the calibration curve. Default gate_config.ACAR_N_EVAL.
    smoke (bool): Use 8 evaluation points. Default False.

    Returns:
    list: FoldJob objects.

    Raises:
    ValueError: If a fold or family is missing, or a bandwidth is not a positive finite number.
    """
    jobs = []
    for f in folds:
        if f not in h_by_fold_family or not isinstance(h_by_fold_family[f], dict):
            raise ValueError(f"Missing fold-specific Acar bandwidth dictionary for {f}")
        choices = h_by_fold_family[f]
        if set(choices) != set(ACAR_FAMILIES):
            raise ValueError(f"Incomplete Acar family bandwidths for {f}: "
                             f"missing {set(ACAR_FAMILIES) - set(choices)}")
        for fam in ACAR_FAMILIES:
            h = float(choices[fam])
            if not np.isfinite(h) or h <= 0:
                raise ValueError(f"Invalid Acar bandwidth for {f}/{fam}: {h}")
            jobs.append(FoldJob(fold=f, arrays=arrays_by_fold[f], kind="acar", family=fam, h=h,
                                n_eval=8 if smoke else n_eval))
    return jobs
