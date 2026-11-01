"""spatial_aggregation.py

Spatial-aggregation robustness of the wind-precipitation dependence, for the parametric models only.

The lake pixels are split into parts of equal pixel count (3 x 3: terciles of longitude, then of latitude within each; 6 x 6:
sextiles). For each part, series of precipitation, wind speed and divergence are built from the part's pixels only, with the
same definitions as the lake-mean series (the divergence is the pixel-level divergence from the full-grid gradients, averaged
over the part). A 2 x 2 design combines the scale of the responses (lake mean or part) with the scale of the divergence
covariate (lake mean or part). Each case goes through the same per-fold chain as the main pipeline (the pieces of
preprocessing.preprocess_fold and fold_models.fit_parametric_fold).

SpatialRunner runs the cases over the blocked folds, iter_unit_cases runs many parts in parallel, and summary_table,
reading_guide and the plot_summary_* functions report the results.

References:
Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Annals of Statistics, 17(3),
    1217-1241.
Liu, R. Y., & Singh, K. (1992). Moving blocks jackknife and bootstrap capture weak dependence. In R. LePage & L. Billard
    (Eds.), Exploring the limits of bootstrap (pp. 225-248). Wiley.
Patton, A., Politis, D. N., & White, H. (2009). Correction to "Automatic block-length selection for the dependent
    bootstrap" by D. Politis and H. White. Econometric Reviews, 28(4), 372-375.
Politis, D. N., & Romano, J. P. (1992). A circular block-resampling procedure for stationary data. In R. LePage & L.
    Billard (Eds.), Exploring the limits of bootstrap (pp. 263-270). Wiley.
Politis, D. N., & White, H. (2004). Automatic block-length selection for the dependent bootstrap. Econometric Reviews,
    23(1), 53-70.
"""
import contextlib
import io
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from data_split import moving_block_indices, pw_block_length_segments
from fold_models import fit_parametric_fold
from gate_config import Q_GRID
from preprocessing import ar_reach, deseasonalise_series, fold_regime_pit, fold_residuals_one, tune_kernel_bandwidth
from run_support import resolve_pool_workers
from figure_replay import recorded

def make_partitions(lat, lon, k):
    """Equal-pixel-count partitions of the lake pixels.

    k = m * m gives m equal-count slices of longitude and then m equal-count slices of latitude within each (k = 9: 3 x 3,
    k = 36: 6 x 6). Part sizes differ by at most one pixel, because the split is rank based, so ties in latitude or longitude
    cannot unbalance them.

    Parameters:
    lat (np.ndarray): Latitude of the pixels (lat_lake).
    lon (np.ndarray): Longitude of the pixels (lon_lake).
    k (int): A perfect square, for example 9 or 36.

    Returns:
    list: k boolean masks of length len(lat), disjoint and covering all pixels.

    Raises:
    ValueError: If k is not a perfect square.
    """
    m = int(round(np.sqrt(k)))
    if m * m != k or m < 1:
        raise ValueError("k must be a perfect square (9 for 3x3, 36 for 6x6)")
    lat = np.asarray(lat); lon = np.asarray(lon)
    N = len(lat)
    order_lon = np.argsort(lon, kind='stable')
    masks = []
    for chunk in np.array_split(order_lon, m):
        order_lat = chunk[np.argsort(lat[chunk], kind='stable')]
        for sub in np.array_split(order_lat, m):
            mk = np.zeros(N, dtype=bool); mk[sub] = True
            masks.append(mk)
    assert sum(mk.sum() for mk in masks) == N and not np.any(np.sum(masks, axis=0) > 1)
    return masks


def part_series(lakepts, mask):
    """Time series of the mean precipitation, wind and divergence over the pixels of one part.

    Parameters:
    lakepts (dict): 'precip', 'wind' and 'div' -> (n_t, n_pixels) arrays.
    mask (np.ndarray): Boolean mask of the pixels of the part.

    Returns:
    dict: 'precip', 'wind' and 'div' -> (n_t,) arrays (nanmean over the pixels).
    """
    return {v: np.nanmean(np.asarray(lakepts[v])[:, mask], axis=1)
            for v in ('precip', 'wind', 'div')}


def filter_series(precip, wind, div, times, jitter_seed=42):
    """Deseasonalise the series of one spatial unit as the lake-mean series are.

    Precipitation and wind get the cyclic hour-of-day-aware 90-day rolling deseasonalisation. The divergence gets a 90-day
    non-hourly moving average (preprocessing.deseasonalise_series with div_diurnal='keep'). Nothing is dropped.

    Parameters:
    precip (array-like): Precipitation series.
    wind (array-like): Wind series.
    div (array-like): Divergence series.
    times (array-like): Time axis.
    jitter_seed (int): Seed of the jitter of the deseasonalisation. Default 42.

    Returns:
    tuple: (precipitation, wind, divergence, hour of day), three pandas Series on the same DatetimeIndex and an hour-of-day
        Series.
    """
    p, w, d, hrs, _ = deseasonalise_series(precip, wind, div, times, jitter_seed=jitter_seed, div_diurnal="keep")
    return p, w, d, hrs


def kendall(a, b):
    """Kendall's tau of two arrays.

    Parameters:
    a (array-like): First variable.
    b (array-like): Second variable.

    Returns:
    float: Kendall's tau, or NaN if there are fewer than 3 points.
    """
    return float(kendalltau(a, b)[0]) if len(a) > 2 else np.nan


def chi_hat(u1, u2, q):
    """Finite-threshold upper-tail coefficient chi(q) = P(U1 > q, U2 > q) / (1 - q), with the joint-exceedance count.

    Parameters:
    u1 (np.ndarray): First pseudo-observation.
    u2 (np.ndarray): Second pseudo-observation.
    q (float): Threshold.

    Returns:
    tuple: (chi, number of joint exceedances). Under independence chi = 1 - q.
    """
    k = int(np.sum((u1 > q) & (u2 > q)))
    return k / (len(u1) * (1 - q)), k


def chi_hat_lower(u1, u2, q):
    """Finite-threshold lower-tail coefficient P(U1 < q, U2 < q) / q, with the joint count.

    Parameters:
    u1 (np.ndarray): First pseudo-observation.
    u2 (np.ndarray): Second pseudo-observation.
    q (float): Threshold (small values, for example 0.10).

    Returns:
    tuple: (coefficient, number of joint events). NaN if the series is empty. Under independence the coefficient is q.
    """
    k = int(np.sum((u1 < q) & (u2 < q)))
    return (k / (len(u1) * q) if len(u1) else np.nan), k


def tau_block_se(x, y, high, block, n_boot=100, seed=0):
    """Moving-block bootstrap standard errors of tau and of the regime contrast.

    The contrast is delta_tau = tau(high) - tau(low). Blocks are drawn over time positions with the circular block bootstrap of Politis and Romano (1992), a fixed block length with
    wrap-around (data_split.moving_block_indices), which is asymptotically equivalent to the moving-block bootstrap of Kunsch
    (1989) and Liu and Singh (1992). The block length is the data-driven Politis-White length for the circular bootstrap
    (Politis & White, 2004, with the correction of Patton et al., 2009; run_case passes pw_block_length_segments with
    variant='circular'). That length is derived for the variance of a sample mean and is used here as a practical choice for
    Kendall's tau. The regime labels travel with the resampled steps, so the regime sizes vary slightly
    across replicates.

    Parameters:
    x (array-like): First variable (precipitation residuals).
    y (array-like): Second variable (wind residuals).
    high (array-like): Boolean regime of each step (True = high divergence).
    block (int): Fixed block length of the circular bootstrap.
    n_boot (int): Bootstrap replicates. Default 100.
    seed (int): Seed. Default 0.

    Returns:
    tuple: (se_tau_all, se_delta_tau).
    """
    x = np.asarray(x); y = np.asarray(y); high = np.asarray(high, bool)
    rng = np.random.default_rng(seed)
    n = len(x)
    t_all, d = [], []
    for _ in range(n_boot):
        i = moving_block_indices(n, block, rng)
        xi, yi, hi = x[i], y[i], high[i]
        t_all.append(kendall(xi, yi))
        d.append(kendall(xi[hi], yi[hi]) - kendall(xi[~hi], yi[~hi]))
    return float(np.nanstd(t_all, ddof=1)), float(np.nanstd(d, ddof=1))


class SpatialRunner:
    """Run the 2 x 2 design on lake and part series over the blocked folds.

    Standardised residuals are cached per (series id, variable, fold), so a response series shared by several cases is
    estimated once.

    Parameters:
    series (dict): {series id: (precipitation, wind, divergence) filtered Series}; 'lake' is the lake mean of the main
        pipeline.
    hours (pd.Series): Hour of day on the full time index.
    common_idx (pd.DatetimeIndex): Global usable index (n positions).
    folds (list): Output of make_blocked_folds.
    n_starts (int): Number of GARCH multistarts. Default 20.
    verbose (bool): Print the preprocessing messages. Default False.
    ar_garch_sample (str): 'full' fits the AR and GARCH models once on the full usable sample and shares them across folds;
        'fold' refits them on each fold's training rows. Default 'full'.

    Raises:
    ValueError: If ar_garch_sample is not 'full' or 'fold'.
    """

    def __init__(self, series, hours, common_idx, folds, n_starts=20, verbose=False,
                 ar_garch_sample='full'):
        """Store the series, the time axes and the folds.

        Parameters:
        series (dict): {series id: (precipitation, wind, divergence) filtered Series}.
        hours (pd.Series): Hour of day on the full time index.
        common_idx (pd.DatetimeIndex): Global usable index.
        folds (list): Output of make_blocked_folds.
        n_starts (int): Number of GARCH multistarts. Default 20.
        verbose (bool): Print the preprocessing messages. Default False.
        ar_garch_sample (str): 'full' or 'fold'. Default 'full'.

        Raises:
        ValueError: If ar_garch_sample is not 'full' or 'fold'.
        """
        self.series, self.hours, self.common_idx = series, hours, common_idx
        self.folds, self.n_starts, self.verbose = folds, n_starts, verbose
        if ar_garch_sample not in ('full', 'fold'):
            raise ValueError("ar_garch_sample must be 'full' or 'fold'")
        self.ar_garch_sample = ar_garch_sample
        self._resid = {}
        self._lags = {}
        self.diag = {}

    def residuals(self, sid, var, fold):
        """Standardised residuals of one series and variable.

        Parameters:
        sid (str): Series id (a key of series).
        var (str): 'precip' or 'wind'.
        fold (dict): One element of folds, with 'name' and 'train'.

        Returns:
        tuple: (residuals over common_idx, label of the volatility model).
        """
        full = self.ar_garch_sample == 'full'
        tag = 'full' if full else fold['name']
        key = (sid, var, tag)
        if key not in self._resid:
            filt = {'precip': self.series[sid][0], 'wind': self.series[sid][1]}[var]
            mask = np.ones(len(self.common_idx), dtype=bool) if full else fold['train']
            # The AR fit drops its first max(lags) rows, and common_idx starts only `off` rows after the start of the
            # filtered series. Without a cap, a part whose PACF selects a lag above `off` would trim rows inside common_idx
            # (error "AR trim removed rows inside common_idx"). The candidate lags are capped at `off`, which keeps every
            # retained lag inside the available head-room. The lake-mean series is unaffected, because its longest retained
            # lag is at most `off`.
            off = int(filt.index.get_loc(self.common_idx[0]))
            r, lags, lab, diag = fold_residuals_one(filt, self.hours, self.common_idx, mask,
                                                    f"{var}[{sid}]", max_lags=min(64, off), n_starts=self.n_starts,
                                                    verbose=self.verbose)
            self._resid[key] = (r, lab)
            self._lags[key] = list(lags)
            self.diag[key] = dict(series=sid, var=var, fold=tag, lags=len(lags), **diag)
        return self._resid[key]

    def health_table(self):
        """Preprocessing diagnostics of every fit, one row per series, variable and fold.

        The columns give the volatility model, the GARCH persistence, the Ljung-Box p-values of the standardised training
        residuals and two flags (persistence >= 0.98; any Ljung-Box p-value < 0.01).

        Returns:
        pd.DataFrame: The table (empty if nothing has been fitted).
        """
        h = pd.DataFrame(list(self.diag.values()))
        if len(h):
            h['persistence_flag'] = h['persistence'] >= 0.98
            h['lb_flag'] = h[['lb_p8', 'lb_p16', 'lb_p24']].min(axis=1) < 0.01
        return h

    def run_case(self, resp_id, cov_id, meta=None):
        """One (responses, covariate) combination over all folds.

        The kernel bandwidth is tuned once per case on the case's own residuals and covariate, using training rows only.

        Parameters:
        resp_id (str): Series id of the responses.
        cov_id (str): Series id of the divergence covariate.
        meta (dict): Extra columns copied into every row. Default None.

        Returns:
        pd.DataFrame: One row per fold plus a 'pooled' row, the test-size-weighted mean of the fold values (NaN where a fold value
            is missing). The columns hold the block lengths b (stationary bootstrap) and b_circular (circular bootstrap, used for the
            standard errors), tau (all, low, high), delta_tau and its block-bootstrap SE, chi(q) and the counts, the
            lower-tail and regime-specific coefficients, the AIC-selected unconditional family with its CVPE, the AIC-selected
            regime pair with its CVPE, and the CVPE of each same-family pair.
        """
        meta = dict(meta or {}, resp=resp_id, cov=cov_id)
        rows = []
        div_filt = self.series[cov_id][2]
        finals = {f['name']: {v: self.residuals(resp_id, v, f)[0] for v in ('precip', 'wind')} for f in self.folds}
        def _reach(f):     # longest retained AR lag of the two response series in this fold (data driven)
            """Longest retained AR lag of the two response series in a fold (data driven).

            Parameters:
            f (dict): One element of folds, with 'name'.

            Returns:
            int: The longest retained lag.
            """
            tag = 'full' if self.ar_garch_sample == 'full' else f['name']
            return ar_reach({v: self._lags[(resp_id, v, tag)] for v in ('precip', 'wind')})
        pre = {f['name']: fold_regime_pit(finals[f['name']], div_filt, self.common_idx, f['train'], h=None, min_len=_reach(f))
               for f in self.folds}
        h_case, _ = tune_kernel_bandwidth(
            {n: {v: np.asarray(x) for v, x in fin.items()} for n, fin in finals.items()},
            {n: p['x_std'] for n, p in pre.items()}, {f['name']: f['train'] for f in self.folds},
            {n: p['b'] for n, p in pre.items()})
        for f in self.folds:
            final = finals[f['name']]
            r = fold_regime_pit(final, div_filt, self.common_idx, f['train'], h=h_case, min_len=_reach(f))
            hi, lo = r['high_mask'], r['low_mask']
            fp = fit_parametric_fold(r['u1'], r['u2'], f['train'], f['test'], hi)
            # Block length of the circular bootstrap behind the standard errors (the length r['b'] is that of the stationary bootstrap).
            b_circ = pw_block_length_segments([final['precip'], final['wind'], r['x_std']], f['train'], min_len=_reach(f),
                                              variant='circular')
            te = f['test']
            mloss = {k: float(np.nanmean(v[te])) for k, v in fp['loss'].items()}
            best_un = min(fp['aic_uncond'], key=fp['aic_uncond'].get)
            fl, fh = fp['selected_pair']
            row = dict(meta, fold=f['name'], n_test=int(te.sum()), b=r['b'], b_circular=b_circ, kernel_h=h_case,
                       tau_all=kendall(final['precip'], final['wind']),
                       tau_low=kendall(final['precip'][lo], final['wind'][lo]),
                       tau_high=kendall(final['precip'][hi], final['wind'][hi]),
                       high_share=float(hi.mean()),
                       uncond_family=best_un, cvpe_uncond=mloss[f"Uncond:{best_un}"],
                       pair_low=fl, pair_high=fh, cvpe_binary=mloss[f"Binary:{fl}|{fh}"])
            row['delta_tau'] = row['tau_high'] - row['tau_low']
            row['se_tau_all'], row['se_delta_tau'] = tau_block_se(final['precip'], final['wind'], hi, block=max(1, int(b_circ)))
            for q in Q_GRID:
                c, k = chi_hat(r['u1'], r['u2'], q)
                row[f'chi_{q:.2f}'], row[f'n_joint_{q:.2f}'] = c, k
            c, k = chi_hat_lower(r['u1'], r['u2'], 0.10)
            row['chiL_all_0.10'], row['nL_all_0.10'] = c, k
            # Regime-conditional tail co-exceedance over all steps of the regime (kernel-conditional PIT); 1 = independence.
            for nm, m in (('low', lo), ('high', hi)):
                c, k = chi_hat(r['u1'][m], r['u2'][m], 0.90)
                row[f'chiU_{nm}_0.90'], row[f'nU_{nm}_0.90'] = c, k
                c, k = chi_hat_lower(r['u1'][m], r['u2'][m], 0.10)
                row[f'chiL_{nm}_0.10'], row[f'nL_{nm}_0.10'] = c, k
            for fam in fp['aic_regime']['low']:
                if fam in fp['aic_regime']['high']:
                    row[f'cvpe_same_{fam}'] = mloss[f"Binary:{fam}|{fam}"]
            rows.append(row)
        df = pd.DataFrame(rows)
        num = df.select_dtypes('number').columns.drop(['n_test'])
        w = df['n_test'].values
        pooled = {c: float(np.average(df[c].values, weights=w)) if df[c].notna().all()
                  else np.nan for c in num}
        pooled.update(meta, fold='pooled', n_test=int(w.sum()),
                      uncond_family='/'.join(df.uncond_family),
                      pair_low='/'.join(df.pair_low), pair_high='/'.join(df.pair_high))
        return pd.concat([df, pd.DataFrame([pooled])], ignore_index=True)


# ── summaries and figure ───────────────────────────────────────────────────
CASE_ORDER = ["(lake, lake)", "(lake covariate, part responses)",
              "(part covariate, lake responses)", "(part, part)"]
CASE_SHORT = {"(lake, lake)": "baseline",
              "(lake covariate, part responses)": "partial responses",
              "(part covariate, lake responses)": "partial divergence",
              "(part, part)": "partial + partial"}
SCALE_ORDER = ["lake", "N/9", "N/36"]
SCALE_LABEL = {"lake": "Lake", "N/9": "Partial\n3$\\times$3", "N/36": "Partial\n6$\\times$6"}
SCALE_TEXT = {"lake": "Lake", "N/9": "Partial 3x3", "N/36": "Partial 6x6"}
# Okabe-Ito colours (colour-blind safe) plus a distinct marker per design
CASE_COLOR = {"(lake, lake)": "#000000", "(lake covariate, part responses)": "#0072B2",
              "(part covariate, lake responses)": "#E69F00", "(part, part)": "#CC79A7"}
CASE_MARKER = {"(lake, lake)": "D", "(lake covariate, part responses)": "o",
               "(part covariate, lake responses)": "s", "(part, part)": "^"}
CASE_LABEL = {"(lake, lake)": "Lake mean (baseline)",
              "(lake covariate, part responses)": "Partial responses",
              "(part covariate, lake responses)": "Partial divergence",
              "(part, part)": "Partial responses and divergence"}


def pooled_rows(results):
    """Pooled (fold-averaged) rows of the run_case results.

    Parameters:
    results (pd.DataFrame): Concatenated run_case results.

    Returns:
    pd.DataFrame: The rows with fold == 'pooled'.
    """
    return results[results.fold == 'pooled'].copy()


def _fmt(m, lo, hi, nd=3):
    """Format a mean with its range as 'm  [lo, hi]'.

    Parameters:
    m (float): Mean.
    lo (float): Lower end of the range.
    hi (float): Upper end of the range.
    nd (int): Decimals. Default 3.

    Returns:
    str: The text.
    """
    return f"{m:.{nd}f}  [{lo:.{nd}f}, {hi:.{nd}f}]"


def summary_table(results):
    """One row per design case and scale: the mean over parts with the min-max range of the parts in brackets.

    Covers tau, delta tau and the tail co-exceedance coefficients, and the mean standard error of delta tau.

    Parameters:
    results (pd.DataFrame): Concatenated run_case results with the columns 'case' and 'scale'.

    Returns:
    pd.DataFrame: The summary table.
    """
    pr = pooled_rows(results)
    cols = [('tau_all', 'tau (all)'), ('delta_tau', 'delta tau'),
            ('chi_0.90', 'chi(0.90)'),
            ('chiL_all_0.10', 'chiL(0.10)'),
            ('chiU_low_0.90', 'chiU low-div .90'), ('chiU_high_0.90', 'chiU high-div .90'),
            ('chiL_low_0.10', 'chiL low-div .10'), ('chiL_high_0.10', 'chiL high-div .10'),
            ]
    cols = [(c, l) for c, l in cols if c in pr.columns]
    rows = []
    for case in CASE_ORDER:
        for sc in SCALE_ORDER:
            sub = pr[(pr.case == case) & (pr.scale == sc)]
            if not len(sub):
                continue
            row = {'design': CASE_SHORT[case], 'scale': SCALE_TEXT[sc], 'units': len(sub)}
            for c, l in cols:
                row[l] = _fmt(sub[c].mean(), sub[c].min(), sub[c].max())
            if 'se_delta_tau' in sub:
                row['SE of delta tau'] = f"{sub['se_delta_tau'].mean():.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def reading_guide(results):
    """Plain-language reading of the run, computed from the numbers.

    Answers (i) whether tau changes with the averaging area, and (ii) and (iii) whether the regime contrast persists and whether
    part-level divergence sharpens it.

    Parameters:
    results (pd.DataFrame): Concatenated run_case results.

    Returns:
    list: Lines of text.
    """
    pr_all = pooled_rows(results)
    pr = pr_all[pr_all.scale.isin(['lake', 'N/9', 'N/36'])]
    base = pr[pr.case == "(lake, lake)"].iloc[0]
    out = []
    has_se = 'se_tau_all' in pr.columns
    resp = pr[pr.case == "(lake covariate, part responses)"]
    out.append("(i) Does tau rise with the averaging area (mechanical effect)?")
    out.append(f"    lake: {base.tau_all:.3f};  3x3 parts: {resp[resp.scale=='N/9'].tau_all.mean():.3f};  "
               f"6x6 parts: {resp[resp.scale=='N/36'].tau_all.mean():.3f}  (means over parts)")
    if has_se:
        k = int((np.abs(resp.tau_all - base.tau_all) <= 2 * resp.se_tau_all).sum())
        gap = resp.tau_all.mean() - base.tau_all
        out.append(f"    {k} of {len(resp)} parts lie within 2 standard errors of the lake value "
                   f"(typical SE of one tau: {resp.se_tau_all.mean():.3f}); mean part minus lake: {gap:+.3f}.")
        out.append("    -> " + ("part values are mostly within noise of the lake value: no scale effect "
                                "distinguishable from noise." if k >= 0.75 * len(resp) else
                                "part values differ from the lake value by more than noise."))
    pc = pr[pr.case == "(part covariate, lake responses)"]
    out.append("")
    out.append("(ii)/(iii) Does the regime contrast delta tau = tau(high) - tau(low) persist, "
               "and does part-level divergence sharpen it?")
    out.append(f"    lake baseline: {base.delta_tau:.3f}" + (f"  (SE {base.se_delta_tau:.3f})" if has_se else ""))
    for case in CASE_ORDER[1:]:
        sub = pr[pr.case == case]
        out.append(f"    {CASE_SHORT[case]:<16}: mean {sub.delta_tau.mean():+.3f}, "
                   f"{int((sub.delta_tau > 0).sum())} of {len(sub)} parts positive")
    diff = pc.delta_tau - base.delta_tau
    out.append(f"    part-level vs lake-mean divergence: {int((diff > 0).sum())} of {len(pc)} parts give a larger "
               f"delta tau; mean difference {diff.mean():+.3f}" +
               (f" (SE of one delta tau {base.se_delta_tau:.3f})." if has_se else "."))
    if has_se:
        out.append("    -> " + ("differences are within noise: no evidence that the lake mean dilutes the "
                                "regime signal." if abs(diff.mean()) < base.se_delta_tau else
                                "systematic difference larger than one SE."))
    return out


def _fold_stats(results, case, scale, col):
    """Point estimate and range over folds of one column for one case and scale.

    Parameters:
    results (pd.DataFrame): Concatenated run_case results.
    case (str): Design case.
    scale (str): 'lake', 'N/9' or 'N/36'.
    col (str): Column.

    Returns:
    tuple: (mean over parts of the fold-pooled value, smallest and largest fold value, each fold's value averaged over parts),
        or None if there are no rows.
    """
    sub = results[(results.case == case) & (results.scale == scale)]
    if not len(sub):
        return None
    point = sub[sub.fold == 'pooled'][col].mean()
    per_fold = sub[sub.fold != 'pooled'].groupby('fold')[col].mean()
    return point, per_fold.min(), per_fold.max()


# The panel title says which part of the sample the panel is about, the y label says which quantity is plotted.
SUMMARY_TOP = [('tau_all', "All steps"),
               ('delta_tau', "All steps"),
               ('chi_0.90', "Upper tail, all steps"),
               ('chiL_all_0.10', "Lower tail, all steps")]
SUMMARY_BOTTOM = [('chiU_low_0.90', "Upper tail, low divergence"),
                  ('chiU_high_0.90', "Upper tail, high divergence"),
                  ('chiL_low_0.10', "Lower tail, low divergence"),
                  ('chiL_high_0.10', "Lower tail, high divergence")]
SUMMARY_YLABEL = {'tau_all': "Kendall's $\\tau$",
                  'delta_tau': "Regime contrast\n$\\Delta\\tau=\\tau_{\\mathrm{high}}-\\tau_{\\mathrm{low}}$",
                  'chi_0.90': "Empirical co-exceedance\nprobability, $q=0.90$",
                  'chiL_all_0.10': "Empirical co-exceedance\nprobability, $q=0.10$",
                  'chiU_low_0.90': "Empirical co-exceedance\nprobability, $q=0.90$",
                  'chiU_high_0.90': "Empirical co-exceedance\nprobability, $q=0.90$",
                  'chiL_low_0.10': "Empirical co-exceedance\nprobability, $q=0.10$",
                  'chiL_high_0.10': "Empirical co-exceedance\nprobability, $q=0.10$"}
_NO_INDEP = ('tau_all', 'delta_tau')          # panels without the independence line
_OFFS = {"(lake, lake)": 0.0, "(lake covariate, part responses)": -0.24,
         "(part covariate, lake responses)": 0.0, "(part, part)": 0.24}
_XPOS = {'lake': 0, 'N/9': 1, 'N/36': 2}


def _tail_ylim(results):
    """Common y-range of all six tail panels (both rows), so that the two figures match.

    The range includes the independence level 0.10 and the fold ranges.

    Parameters:
    results (pd.DataFrame): Concatenated run_case results.

    Returns:
    tuple: (lower, upper) limit.
    """
    vals = [0.10]
    for col, _ in SUMMARY_TOP[2:] + SUMMARY_BOTTOM:
        for case in CASE_ORDER:
            for sc in SCALE_ORDER:
                st = _fold_stats(results, case, sc, col)
                if st is not None:
                    vals += [st[0], st[1], st[2]]
    lo, hi = min(vals), max(vals)
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _summary_row(results, panels, first_letter, path):
    """One row of four panels (estimate by spatial scale and design) with the shared legend underneath.

    Parameters:
    results (pd.DataFrame): Concatenated run_case results.
    panels (list): (column, panel title) pairs.
    first_letter (str): Letter of the first panel.
    path (str): Output file, or None to skip saving.

    Returns:
    matplotlib.figure.Figure: The figure.

    Raises:
    KeyError: If the results lack a plotted column.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    pr = pooled_rows(results)
    missing = [c for c, _ in panels if c not in pr.columns]
    if missing:
        raise KeyError(f"results lack the columns {missing}: they come from an older version of the cases, which "
                       "must be rerun")
    ylim = _tail_ylim(results)
    import plotting_core as _pc
    # 2 x 2 panels: four panels in one row of the text width leave no room for the titles and the y labels
    fig, axes = plt.subplots(2, 2, figsize=_pc.print_figsize(4.9), constrained_layout=True)
    for k, (ax, (col, lab)) in enumerate(zip(axes.ravel(), panels)):
        ax.set_axisbelow(True)
        ax.grid(True, color='#E3E3E3', lw=0.7)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if col not in _NO_INDEP:
            ax.axhline(0.10, color='#444444', lw=1.1, ls=':', zorder=2)   # independence
            ax.set_ylim(*ylim)
        if col == 'delta_tau':
            ax.axhline(0, color='#9A9A9A', lw=0.8, zorder=1)
        for case in CASE_ORDER:
            for sc in SCALE_ORDER:
                st = _fold_stats(results, case, sc, col)
                if st is None:
                    continue
                m, lo, hi = st
                ax.errorbar([_XPOS[sc] + _OFFS[case]], [m], yerr=[[max(m - lo, 0)], [max(hi - m, 0)]],
                            fmt=CASE_MARKER[case], color=CASE_COLOR[case], ecolor=CASE_COLOR[case],
                            elinewidth=1.2, capsize=2.5, markersize=5, markeredgecolor='white',
                            markeredgewidth=0.6, zorder=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Lake', '3$\\times$3', '6$\\times$6'], fontsize=7)
        ax.set_xlabel('spatial scale (parts)', fontsize=7)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel(SUMMARY_YLABEL.get(col, ''), fontsize=7)
        ax.set_title(f"({chr(ord(first_letter) + k)}) " + lab, fontsize=7.5)
        ax.tick_params(labelsize=7)
    handles = [Line2D([], [], marker=CASE_MARKER[c], color=CASE_COLOR[c], linestyle='none',
                      markersize=5, markeredgecolor='white', label=CASE_LABEL[c])
               for c in CASE_ORDER]
    handles.append(Line2D([], [], color='#444444', lw=1.1, ls=':', label='Independence (0.10)'))
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=7)
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    return fig


@recorded
def plot_summary_top(results, path=None):
    """Top row of the summary figure, panels (a) to (d).

    Shows Kendall's tau, the regime contrast, the upper tail (q = 0.90) and the lower tail (q = 0.10) over all steps, by spatial
    scale and design. Each marker is the fold-pooled estimate averaged over parts and the vertical bar is the range of that
    estimate across the four folds. The tail panels share their y-range with the bottom row (plot_summary_bottom).

    Parameters:
    results (pd.DataFrame): Concatenated run_case results.
    path (str): Output file. Default None.

    Returns:
    matplotlib.figure.Figure: The figure.
    """
    return _summary_row(results, SUMMARY_TOP, 'a', path)


@recorded
def plot_summary_bottom(results, path=None):
    """Bottom row of the summary figure, panels (e) to (h).

    Shows the upper (q = 0.90) and lower (q = 0.10) co-exceedance coefficients in the low- and high-divergence regime, with
    the same encoding and y-range as the tail panels of plot_summary_top. The dotted line is the independence level 0.10.

    Parameters:
    results (pd.DataFrame): Concatenated run_case results.
    path (str): Output file. Default None.

    Returns:
    matplotlib.figure.Figure: The figure.
    """
    return _summary_row(results, SUMMARY_BOTTOM, 'e', path)


# ── parallel runner: one task per spatial unit ─────────────────────────────
_WORKER = {}


def _pixel_init(shared, counter):
    """Worker initialiser.

    Pins the process to one core, so that resolve_max_workers inside the GARCH multistart returns 1 and no nested pool
    oversubscribes the node, and stores the shared read-only inputs.

    Parameters:
    shared (dict): Inputs shared by all tasks.
    counter (multiprocessing.Value): Shared counter used to assign cores.
    """
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = '1'
    try:
        cores = sorted(os.sched_getaffinity(0))
        with counter.get_lock():
            i = counter.value
            counter.value += 1
        os.sched_setaffinity(0, {cores[i % len(cores)]})
    except (AttributeError, OSError):
        pass
    _WORKER.update(shared)


def _unit_task(args):
    """Run the three designs for one spatial unit (a part or a pixel).

    The designs are (unit responses, lake divergence), (lake responses, unit divergence) and (unit, unit).

    Parameters:
    args (tuple): (unit id, scale code, number of pixels, part number, precipitation, wind, divergence series).

    Returns:
    tuple: (unit id, DataFrame or None, diagnostics dict, error text or None). A failing unit returns its error text instead
        of stopping the run.
    """
    uid, scale, npx, part, p, w, d = args
    S = _WORKER
    try:
        pp, ww, dd, _ = filter_series(p, w, d, S['time_valid'])
        runner = SpatialRunner({'lake': S['lake_series'], uid: (pp, ww, dd)}, S['hours'],
                               S['common_idx'], S['folds'], n_starts=S['n_starts'],
                               ar_garch_sample='full')
        runner._resid.update(S['lake_resid'])
        runner._lags = dict(S['lake_lags'])
        meta = dict(scale=scale, n_pixels=npx, part=part)
        specs = [(f"{uid}|lakecov", uid, 'lake', dict(meta, case="(lake covariate, part responses)")),
                 (f"{uid}|partcov", 'lake', uid, dict(meta, case="(part covariate, lake responses)")),
                 (f"{uid}|both", uid, uid, dict(meta, case="(part, part)"))]
        frames = []
        for key, resp, cov, m in specs:
            with contextlib.redirect_stdout(io.StringIO()):
                frames.append(runner.run_case(resp, cov, dict(m, case_key=key)))
        diag = {k: v for k, v in runner.diag.items() if k[0] == uid}
        return uid, pd.concat(frames, ignore_index=True), diag, None
    except Exception as e:                       # a failing unit must not stop the run
        return uid, None, {}, f"{type(e).__name__}: {e}"


def iter_unit_cases(lakepts, time_valid, runner, units, skip=(), max_workers=None):
    """Run the three designs for many spatial units in parallel (one core per unit) and yield each result when it is done.

    Parameters:
    lakepts (dict): 'precip', 'wind' and 'div' -> (n_t, n_pixels) arrays.
    time_valid (array-like): Time axis of lakepts.
    runner (SpatialRunner): Its lake residuals (fitted once on the full sample) are shared with the workers.
    units (list): (unit id, scale code, number of pixels, part number, mask) tuples, with mask a boolean array over the lake
        pixels or an integer pixel index.
    skip (iterable): Unit ids that are already done.
    max_workers (int): Upper limit of the number of worker processes. Default None, which uses run_support.resolve_pool_workers
        (CPU affinity minus two, capped by the number of units and by the free memory).

    Returns:
    generator: Yields (unit id, DataFrame or None, diagnostics dict, error text or None) for each unit as it finishes.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    skip = set(skip)
    units = [u for u in units if u[0] not in skip]
    if not units:
        return
    for var in ('precip', 'wind'):               # make sure the lake residuals exist
        runner.residuals('lake', var, runner.folds[0])
    shared = dict(time_valid=time_valid, lake_series=runner.series['lake'], hours=runner.hours,
                  common_idx=runner.common_idx, folds=runner.folds, n_starts=runner.n_starts,
                  lake_resid={k: v for k, v in runner._resid.items() if k[0] == 'lake'},
                  lake_lags={k: v for k, v in getattr(runner, '_lags', {}).items() if k[0] == 'lake'})
    P, W, D = (np.asarray(lakepts[v]) for v in ('precip', 'wind', 'div'))

    def raw(sel):
        """Precipitation, wind and divergence series of one unit.

        Parameters:
        sel (int or np.ndarray): Pixel index, or boolean pixel mask of a part.

        Returns:
        tuple: (precipitation, wind, divergence), each of length n_t.
        """
        if isinstance(sel, (int, np.integer)):
            return P[:, sel], W[:, sel], D[:, sel]
        ps = part_series(lakepts, sel)
        return ps['precip'], ps['wind'], ps['div']

    workers = resolve_pool_workers(n_tasks=len(units), max_workers=max_workers,
                                   worker_gib=float(os.getenv('THESIS_SPATIAL_WORKER_GIB', '4.0')))
    counter = mp.Value('i', 0)
    with ProcessPoolExecutor(max_workers=workers, initializer=_pixel_init,
                             initargs=(shared, counter)) as ex:
        futs = [ex.submit(_unit_task, (uid, sc, npx, part, *raw(sel))) for uid, sc, npx, part, sel in units]
        for f in as_completed(futs):
            yield f.result()
