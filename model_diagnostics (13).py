"""model_diagnostics.py

Diagnostics of fitted models: finite-threshold tail dependence and the self-consistency check of the networks.

chi_table, chi_bands and chi_grid_with_bands give the empirical joint-exceedance measures chi(q) and chi-bar(q) with
stationary-bootstrap percentile bands. model_implied_chi and model_implied_grid give the same curve implied by a fitted
model, and asymptotic_lambda_table lists the asymptotic tail coefficients of the parametric families. run_self_consistency and
save_self_consistency check that every fitted network density integrates to one.

References:
Coles, S., Heffernan, J., & Tawn, J. (1999). Dependence measures for extreme value analyses. Extremes, 2(4), 339-365.
Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical Association,
    89(428), 1303-1313.
"""
import warnings
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import fold_pipeline as FP
import gate_config as GC
import reporting_tables
from codine import self_consistency_test as codine_self_consistency, self_consistency_test_conditional
from conditional_copula_flow import flow_self_consistency_test
from data_split import masked_stationary_indices, stationary_bootstrap_indices
from run_support import resolve_pool_workers


# ── finite-threshold tail dependence ───────────────────────────────────────
Q_THRESHOLDS = GC.Q_GRID
MIN_REPS_REPORTED = 200


def _cell(u1, u2, q, lower=False):
    """Counts and tail-dependence measures of one tail cell.

    For the upper tail the cell is {U1 > q, U2 > q}, for the lower tail {U1 < 1 - q, U2 < 1 - q}. With the empirical joint
    probability p, chi = P(U2 > q | U1 > q) is estimated by the number of joint events over the number of events in U1, and
    chi-bar = 2 log(1 - q) / log(p) - 1 (Coles et al., 1999). chi_fixed_denominator = p / (1 - q) divides by the nominal
    marginal probability instead of the observed count, which is the quantity a model implies (model_implied_chi).

    Parameters:
    u1 (array-like): First pseudo-observation.
    u2 (array-like): Second pseudo-observation.
    q (float): Threshold of the upper tail, in (0, 1).
    lower (bool): Use the lower tail instead of the upper tail. Default False.

    Returns:
    dict: n, n_joint, n_cond, expected_joint_indep (n (1 - q)^2), chi, chi_fixed_denominator and chibar. Undefined values
        are NaN.
    """
    u1 = np.asarray(u1); u2 = np.asarray(u2)
    n = len(u1)
    if lower:
        a, b = u1 < 1 - q, u2 < 1 - q
    else:
        a, b = u1 > q, u2 > q
    n_joint, n_cond = int((a & b).sum()), int(a.sum())
    chi = n_joint / n_cond if n_cond > 0 else np.nan
    p_joint = n_joint / n if n else np.nan
    chi_fixed_denominator = p_joint / (1 - q) if n else np.nan
    chibar = (2 * np.log(1 - q) / np.log(p_joint) - 1) if (n_joint > 0 and p_joint < 1) else np.nan
    return dict(n=n, n_joint=n_joint, n_cond=n_cond, expected_joint_indep=n * (1 - q) ** 2,
                chi=chi, chi_fixed_denominator=chi_fixed_denominator, chibar=chibar)


def chi_table(u1, u2, qs=Q_THRESHOLDS, label=None):
    """Point estimates and counts of the upper and lower tail on a grid of thresholds.

    Parameters:
    u1 (array-like): First pseudo-observation.
    u2 (array-like): Second pseudo-observation.
    qs (sequence of float): Thresholds. Default Q_THRESHOLDS.
    label (str): Value of an added column 'group'. Default None (no column).

    Returns:
    pd.DataFrame: One row per threshold and tail, with the output of _cell plus q, tail and independence (1 - q).
    """
    rows = []
    for q in qs:
        for tail, lower in (("upper", False), ("lower", True)):
            r = _cell(u1, u2, q, lower); r.update(q=q, tail=tail, independence=1 - q)
            if label is not None:
                r["group"] = label
            rows.append(r)
    return pd.DataFrame(rows)


def chi_bands(u1, u2, qs=Q_THRESHOLDS, b=None, n_boot=MIN_REPS_REPORTED, seed=0, level=0.95, sample_mask=None):
    """Stationary-bootstrap percentile bands for chi, chi-bar and chi with a fixed denominator.

    u1 and u2 must be in time order over the full series (or the full fold series). The resample is drawn over the full
    chronological index (Politis & Romano, 1994). With sample_mask only the resampled positions that belong to the subset
    are kept, so a block is a real contiguous run of time, and within-subset temporal clustering is respected. Replicates
    with an empty conditioning event give NaN and are ignored.

    Parameters:
    u1 (array-like): First pseudo-observation, full time axis.
    u2 (array-like): Second pseudo-observation, full time axis.
    qs (sequence of float): Thresholds. Default Q_THRESHOLDS.
    b (float): Mean block length of the stationary bootstrap.
    n_boot (int): Bootstrap replicates. Default MIN_REPS_REPORTED; a smaller value gives a warning.
    seed (int): Seed. Default 0.
    level (float): Coverage of the bands. Default 0.95.
    sample_mask (np.ndarray): Boolean mask (same length) of the subset to resample. Default None (all steps).

    Returns:
    pd.DataFrame: One row per threshold and tail with chi_lo, chi_hi, chibar_lo, chibar_hi, chi_fixed_lo, chi_fixed_hi and
        n_boot_used (number of replicates with a defined chi).

    Raises:
    ValueError: If sample_mask has the wrong length or selects no step.
    """
    if n_boot < MIN_REPS_REPORTED:
        warnings.warn(f"n_boot={n_boot} < {MIN_REPS_REPORTED}: too few replicates for reported "
                      f"percentile bands.")
    u1 = np.asarray(u1); u2 = np.asarray(u2)
    n = len(u1)
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool).copy()
        if len(mask) != n:
            raise ValueError("sample_mask and pseudo-observations must share the full time axis")
        positions = np.flatnonzero(mask)
        if not len(positions):
            raise ValueError("the selected subset is empty")
    else:
        positions = None
    rng = np.random.default_rng(seed)
    a = (1 - level) / 2
    store = {(q, t): ([], [], []) for q in qs for t in ("upper", "lower")}
    for _ in range(n_boot):
        if positions is not None:
            # Blocks are drawn on the original chronological axis and only subset positions are kept, so a bootstrap
            # block cannot cross the artificial join of the two B1 training segments.
            idx = positions[masked_stationary_indices(n, positions, b, rng)]
        else:
            idx = stationary_bootstrap_indices(n, b, rng)
        x1, x2 = u1[idx], u2[idx]
        for q in qs:
            for tail, lower in (("upper", False), ("lower", True)):
                c = _cell(x1, x2, q, lower)
                store[(q, tail)][0].append(c["chi"]); store[(q, tail)][1].append(c["chibar"]); store[(q, tail)][2].append(c["chi_fixed_denominator"])
    rows = []
    for (q, tail), (chis, cbs, fixed) in store.items():
        chis = np.array(chis, float); cbs = np.array(cbs, float); fixed = np.asarray(fixed, float)
        ok1, ok2 = np.isfinite(chis), np.isfinite(cbs)
        ok3 = np.isfinite(fixed)
        rows.append(dict(q=q, tail=tail,
                         chi_lo=np.quantile(chis[ok1], a) if ok1.any() else np.nan,
                         chi_hi=np.quantile(chis[ok1], 1 - a) if ok1.any() else np.nan,
                         chibar_lo=np.quantile(cbs[ok2], a) if ok2.any() else np.nan,
                         chibar_hi=np.quantile(cbs[ok2], 1 - a) if ok2.any() else np.nan,
                         chi_fixed_lo=np.quantile(fixed[ok3], a) if ok3.any() else np.nan,
                         chi_fixed_hi=np.quantile(fixed[ok3], 1 - a) if ok3.any() else np.nan,
                         n_boot_used=int(ok1.sum())))
    return pd.DataFrame(rows)


def chi_grid_with_bands(u1, u2, high_mask, qs=Q_THRESHOLDS, b=None, n_boot=MIN_REPS_REPORTED, seed=0,
                        sample_mask=None):
    """Estimates, counts and bands for all steps, the low-divergence regime and the high-divergence regime.

    The regime cells use the masked stationary bootstrap of chi_bands, and the counts refer to the regime's own steps.

    Parameters:
    u1 (array-like): First pseudo-observation, time ordered, full series.
    u2 (array-like): Second pseudo-observation, same length.
    high_mask (array-like): Boolean regime of each step (True = high divergence), same length.
    qs (sequence of float): Thresholds. Default Q_THRESHOLDS.
    b (float): Mean block length of the stationary bootstrap.
    n_boot (int): Bootstrap replicates. Default MIN_REPS_REPORTED.
    seed (int): Seed. Default 0.
    sample_mask (np.ndarray): Boolean mask of the steps to use. Default None (all steps).

    Returns:
    pd.DataFrame: chi_table output with the band columns of chi_bands; the column 'group' is 'all', 'low' or 'high'.

    Raises:
    ValueError: If the inputs do not share the same time axis.
    """
    n = len(u1)
    sample = np.ones(n, bool) if sample_mask is None else np.asarray(sample_mask, bool)
    if sample.shape != (n,) or len(u2) != n or len(high_mask) != n:
        raise ValueError("u1, u2, high_mask and sample_mask must share the original time axis")
    out = []
    for lab, m in (("all", np.ones(n, bool)), ("low", ~np.asarray(high_mask, bool)),
                   ("high", np.asarray(high_mask, bool))):
        selected = m & sample
        est = chi_table(np.asarray(u1)[selected], np.asarray(u2)[selected], qs, label=lab)
        bands = chi_bands(u1, u2, qs, b=b, n_boot=n_boot, seed=seed, sample_mask=selected)
        out.append(est.merge(bands, on=["q", "tail"], how="left"))
    return pd.concat(out, ignore_index=True)


def full_sample_regime_u(final_precip, final_wind, div_raw, h=None):
    """Conditional pseudo-observations computed with the whole sample as training sample.

    Descriptive only, never used for selection. The divergence is standardised. If h is given, the pseudo-observations are
    the kernel-weighted conditional ECDF with bandwidth h in standard-deviation units. If h is None, they are the ECDF
    within each of the two regimes, split at the median of the standardised divergence.

    Parameters:
    final_precip (array-like): Precipitation series.
    final_wind (array-like): Wind series.
    div_raw (array-like): Raw divergence series.
    h (float): Kernel bandwidth in SD units. Default None (binary-regime ECDF).

    Returns:
    tuple: (u1, u2, high_mask), with high_mask the median split of the standardised divergence, used for the regime cells.
    """
    from preprocessing import regime_train_only_pit, kernel_conditional_pit
    div = np.asarray(div_raw, float)
    x = (div - div.mean()) / div.std()
    high = x >= np.median(x)
    allm = np.ones(len(x), bool)
    if h is None:
        u1 = regime_train_only_pit(np.asarray(final_precip, float), high.astype(int), allm)
        u2 = regime_train_only_pit(np.asarray(final_wind, float), high.astype(int), allm)
    else:
        u1 = kernel_conditional_pit(np.asarray(final_precip, float), x, allm, h)
        u2 = kernel_conditional_pit(np.asarray(final_wind, float), x, allm, h)
    return u1, u2, high


# ── model-implied curves ───────────────────────────────────────────────────
def model_implied_chi(p_by_fold, arrays_by_fold, models, q):
    """Model-implied chi(q) = P_m(U1 > q, U2 > q) / (1 - q) by regime.

    The joint-exceedance probabilities are averaged over each fold's test steps and then over folds, weighted by the number
    of test steps.

    Parameters:
    p_by_fold (dict): {fold: {model: (n,) array of P(U1 > q, U2 > q | w_t)}}, from fold_pipeline.collect_exceed_probs.
    arrays_by_fold (dict): {fold: fold_arrays} with 'test' and 'high' masks.
    models (list): Model names.
    q (float): Threshold.

    Returns:
    pd.DataFrame: Columns model, regime ('all', 'low', 'high'), q and chi_model.
    """
    rows = []
    for m in models:
        for reg in ("all", "low", "high"):
            num = den = 0.0
            for f, P in p_by_fold.items():
                A = arrays_by_fold[f]
                te = np.asarray(A["test"], bool); hi = np.asarray(A["high"], bool)
                sel = te if reg == "all" else (te & hi if reg == "high" else te & ~hi)
                p = np.asarray(P[m], float)[sel]
                p = p[np.isfinite(p)]
                num += p.sum(); den += len(p)
            rows.append(dict(model=m, regime=reg, q=q, chi_model=(num / den) / (1 - q) if den else np.nan))
    return pd.DataFrame(rows)


def model_implied_grid(p_by_fold_by_q, arrays_by_fold, models):
    """model_implied_chi stacked over thresholds.

    Parameters:
    p_by_fold_by_q (dict): {q: {fold: {model: p}}}.
    arrays_by_fold (dict): {fold: fold_arrays}.
    models (list): Model names.

    Returns:
    pd.DataFrame: Columns model, regime, q and chi_model.
    """
    return pd.concat([model_implied_chi(P, arrays_by_fold, models, q) for q, P in p_by_fold_by_q.items()],
                     ignore_index=True)


def asymptotic_lambda_table(param, folds):
    """Asymptotic tail coefficients of each parametric family and fold, from the fitted parameters.

    These are the values the family assumes in the limit, listed next to the model-implied curve. Regime fits are listed with
    their regime. fold_pipeline.parametric_fold_job stores the coefficients.

    Parameters:
    param (dict): {fold: parametric job result}.
    folds (sequence of str): Folds.

    Returns:
    pd.DataFrame: Columns fold, scope ('uncond', 'low' or 'high'), family, lambda_U and lambda_L.
    """
    rows = []
    for f in folds:
        for r in param[f]["rows_uncond"]:
            rows.append(dict(fold=f, scope="uncond", family=r["name"], lambda_U=r["lambda_U"], lambda_L=r["lambda_L"]))
        for reg in ("low", "high"):
            for r in param[f]["rows_regime"][reg]:
                rows.append(dict(fold=f, scope=reg, family=r["name"], lambda_U=r["lambda_U"], lambda_L=r["lambda_L"]))
    return pd.DataFrame(rows)


# ── branch of the signed copula families ───────────────────────────────────


# ── self-consistency of the networks ───────────────────────────────────────
W_QUANTILES = GC.SELF_CONSISTENCY_PERCENTILES        # the w-quantiles of the KS gate, one rule for every network


def _one_self_consistency(args):
    """Process-safe self-consistency check of one saved network (or one provided in memory).

    Parameters:
    args (tuple): (fold, seed, model, w values, smoke flag, network or None); a network given as None is loaded from its state file.

    Returns:
    list: One row (dict) per check; a single row with status 'no net' if the network cannot be found.
    """
    f, s, m, w_chk, smoke, net = args
    base = dict(fold=f, seed=s, model=m)
    if net is None:
        try:
            import fold_models as MS
            net, _ = MS.load_fitted(FP.BATCH_DIR, FP.net_key(f, s, m))
        except Exception:
            net = None
    if net is None:
        return [dict(base, w=np.nan, w_quantile=np.nan, integral=np.nan, se=np.nan, deviation=np.nan,
                     passed=np.nan, status='no net')]
    nmc = 2_000 if smoke else GC.SELF_CONSISTENCY_N_MC
    nseeds = 3 if smoke else GC.SELF_CONSISTENCY_N_SEEDS
    if m == FP.UNCOND_CODINE:
        r = codine_self_consistency(net, n_mc=nmc, n_seeds=nseeds, seed=1, tol=GC.SELF_CONSISTENCY_TOL)
        return [dict(base, w=np.nan, w_quantile=np.nan, integral=r['integral'], se=r['se'],
                     deviation=r['deviation'], passed=r['passed'], status='ok')]
    if m in FP.CCODINE_VARIANTS:
        res = self_consistency_test_conditional(net, w_chk, n_mc=nmc, n_seeds=nseeds,
                                                seed=1, tol=GC.SELF_CONSISTENCY_TOL)
    else:
        res = flow_self_consistency_test(net, w_chk, n_mc=nmc, n_seeds=nseeds,
                                         seed=1, tol=GC.SELF_CONSISTENCY_TOL)
    return [dict(base, w=wv, w_quantile=q, integral=res[float(wv)]['integral'],
                 se=res[float(wv)]['se'], deviation=abs(res[float(wv)]['integral'] - 1.0),
                 passed=bool(res[float(wv)]['passed']), status='ok')
            for q, wv in zip(W_QUANTILES, w_chk)]


def run_self_consistency(NEURAL, ARR, folds, seeds, smoke=False):
    """Check that the density of every fitted network integrates to one.

    For the unconditional CODINE network the check is codine.self_consistency_test. For the conditional networks it is made
    at the seven w-quantiles (W_QUANTILES = gate_config.SELF_CONSISTENCY_PERCENTILES) of the fold training sample: C-CODINE by
    codine.self_consistency_test_conditional, C-CopFlow by conditional_copula_flow.flow_self_consistency_test. Every network is
    judged by the same rule: pass if |integral - 1| <= gate_config.SELF_CONSISTENCY_TOL, from gate_config.SELF_CONSISTENCY_N_SEEDS
    integrals of gate_config.SELF_CONSISTENCY_N_MC uniform points. A network
    that is not in NEURAL is loaded from its state file.

    Parameters:
    NEURAL (dict): {key: neural_job result}.
    ARR (dict): {fold: fold_arrays}.
    folds (sequence of str): Folds.
    seeds (sequence of int): Seeds.
    smoke (bool): Fewer Monte Carlo draws and seeds, for a quick test. Default False.

    Returns:
    tuple: (long_df, summary_df). long_df has one line per network (and w-quantile), with status 'no net' for a network that
        cannot be found. summary_df has one line per model and fold with n_nets, n_checks, n_failed, max_abs_dev and
        mean_integral.
    """
    jobs = []
    for f in folds:
        A = ARR[f]
        w_chk = [float(q) for q in np.quantile(np.asarray(A['w'])[np.asarray(A['train'], bool)], W_QUANTILES)]
        for s in seeds:
            for m in [FP.UNCOND_CODINE] + list(FP.NEURAL_MODELS):
                rec = NEURAL.get(FP.net_key(f, s, m))
                net = rec.get('net') if rec is not None else None
                jobs.append((f, s, m, w_chk, smoke, net))

    # Notebook 04 passes NEURAL={}, so every worker loads its own model from the
    # persisted state. If models were passed in memory, stay serial to avoid
    # pickling a live PyTorch module (and possibly duplicating large GPU state).
    workers = (1 if NEURAL else resolve_pool_workers(
        n_tasks=len(jobs), worker_gib=float(os.getenv('THESIS_SELFCHECK_WORKER_GIB', '2.0'))))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=FP._worker_init) as ex:
            chunks = list(ex.map(_one_self_consistency, jobs))
    else:
        chunks = [_one_self_consistency(job) for job in jobs]
    rows = [row for chunk in chunks for row in chunk]
    long_df = pd.DataFrame(rows)
    ok = long_df[long_df.status == 'ok'].copy()
    ok['passed'] = ok['passed'].astype(bool)
    summary = (ok.groupby(['model', 'fold'])
                 .agg(n_nets=('seed', 'nunique'), n_checks=('passed', 'size'), n_failed=('passed', lambda x: int((~x).sum())),
                      max_abs_dev=('deviation', 'max'), mean_integral=('integral', 'mean'))
                 .reset_index())
    return long_df, summary


def save_self_consistency(long_df, summary):
    """Write the self-consistency overview table of the report as LaTeX and CSV.

    Parameters:
    long_df (pd.DataFrame): Long table from run_self_consistency, one line per network (and w-quantile).
    summary (pd.DataFrame): Summary table from run_self_consistency.

    Returns:
    pd.DataFrame: The model-level overview (maximum absolute deviation per fold, number of checks and failed checks).
    """
    wide = summary.pivot(index='model', columns='fold', values='max_abs_dev')
    nfail = summary.groupby('model').n_failed.sum().rename('failed checks')
    nchk = summary.groupby('model').n_checks.sum().rename('checks')
    overview = wide.join(nchk).join(nfail)
    overview.index.name = 'Model'
    reporting_tables.df_to_latex(
        overview, 'self_consistency_overview.tex',
        caption=(r'Self-consistency check $|\overline{\hat c(\mathbf{u}\mid w)}-1|$: maximum absolute deviation per fold, '
                 r'over all seeds and the seven $w$-quantiles (5, 10, 30, 50, 70, 90, 95\%) of the fold training sample, the same as in the KS check. '
                 r'The last two columns count all checks and the failed ones (all networks alike: deviation $>0.01$).'),
        label='tab:self_consistency_overview', digits=4, col_format='l' + 'r' * (overview.shape[1]))
    return overview
