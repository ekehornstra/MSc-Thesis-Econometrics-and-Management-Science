"""preprocessing.py

Raw-data loading and preprocessing utilities for the Lake Victoria wind-precipitation pipeline.

- load_forecast_data / open_nc / date_key / get_lake_mask /
  select_days_7_to_9_hourly / select_days_7_to_9_wind: load the raw NWP
  forecast NetCDF files, restrict to the lead-day-7-9 window, and build lake-wide mean and grid-point-sampled series.
- compute_spinup_thresholds: the spin-up cutoff analysis of storm object counts per forecast lead day
  under precipitation-only, wind-only, and compound-exceedance
  definitions. Called from 00_preprocessing.ipynb; its output
  (spinup_results, spinup_thresholds) is saved into the pipeline artifact
  and consumed by plot_spinup_threshold_sensitivity in
  06_final_figures.ipynb. Parallel across common_dates via
  ProcessPoolExecutor: pass 1 loads/interps each date once and caches
  compact arrays to scratch; pass 2 reads that cache (no repeated
  NetCDF/interp) to count storm objects against the thresholds pass 1
  computed.
- cyclic_rolling_stats / deseasonalise_series: removal of the seasonal cycle (and, for precipitation and wind,
  the diurnal cycle): a cyclic 90-day rolling mean and scale by hour of day for precipitation and wind, and a
  90-day non-hourly moving average for divergence (its diurnal cycle is kept).
- select_ar_lags / fit_ar_masked: the sparse-AR linear-autocorrelation
  removal step (AR filtering).
- The GARCH-family volatility model selection machinery (_hansen_skewt_pdf,
  _kappa_skewt, compute_persistence, _perturbed_garch_start, _condense_fail_reasons, _garch_one_start,
  fit_garch_multistart, select_best_volatility_model,
  day_night_diagnostic, select_volatility_for_variable, fit_garch_masked):
  volatility filtering. Fits the GARCH/GJR-GARCH/EGARCH candidate
  menu under Gaussian/Student-t/skewed-t innovations, selects by AIC among
  eligible (covariance-stationary, well-identified) candidates, and falls
  back to a day/night diagnostic when no candidate is eligible.
- train_only_pit / regime_train_only_pit / kernel_conditional_pit /
  tune_kernel_bandwidth / decile_uniformity / uniformity_row: train-only
  and kernel-conditional probability integral transforms, bandwidth
  selection and uniformity diagnostics.
- fold_residuals_one / fold_regime_pit / preprocess_fold / ar_reach: the
  per-fold preprocessing chain from deseasonalised series to
  pseudo-observations.
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import quad
from scipy.special import gammaln
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf
from tqdm.auto import tqdm

from gate_config import NIGHT_HOURS_UTC
from run_support import get_logger, resolve_max_workers

log = get_logger()

# Multistart fits whose log-likelihood is at least this far from the median across starts are
# treated as diverged optimiser runs and dropped before the stability metric is computed.
_LOGLIK_OUTLIER_TOL = 50

def load_forecast_data(common_dates, path, lsm_ds,
                       total_target_samples=500_000, smooth_size=1,
                       load_wst=False):
    """Load, align, and spatially reduce the raw NWP forecast NetCDFs for the lead-day-7-9 window.

    Per forecast run: opens the precip/wind (and optionally surface
    temperature) NetCDFs, restricts to the lead-day-7-9 window via select_days_7_to_9_hourly/select_days_7_to_9_wind,
    resamples precip to 3-hourly and aligns it to the wind timestamps,
    computes wind speed and horizontal divergence, then reduces the
    result three ways: lake-wide means, a random grid-point sample per
    date (for spatial-scale analysis), and a lake-subset spatial array
    (for spatial diagnostics/plots).

    Parameters:
    common_dates (list): Forecast initialisation date strings shared
        across all required variables.
    path (str): Directory containing the per-date NetCDF files, expected
        to be named like 'precip_sfc_{date}.nc', 'uwnd_10m_{date}.nc',
        'vwnd_10m_{date}.nc', and (if load_wst) 'st_sfc_{date}.nc'.
    lsm_ds (xarray.Dataset): Land-sea mask dataset, passed to get_lake_mask.
    total_target_samples (int): Total grid-point samples to draw across
        all dates combined; divided evenly per date. Default 500,000.
    smooth_size (int): Uniform filter size applied to the wind field
        before computing divergence (1 = no smoothing). Default 1.
    load_wst (bool): Also load and process surface temperature. Default False.

    Returns:
    tuple: (lake_means, samples, spatial).
        lake_means (dict): 'precip', 'wind', 'div', 'time' (and 'wst' if
            load_wst), each a concatenated 1-D array across all dates.
        samples (dict): 'precip', 'wind', 'div', 'month', 'hour' (and
            'wst' if load_wst), each a concatenated 1-D array of the
            random grid-point sample across all dates.
        spatial (dict): 'lat_lake', 'lon_lake', 'lat_sub', 'lon_sub',
            'lake_sub' (the lake bounding-box arrays and mask), and
            'lakepts' (dict of variable -> (n_t, n_lake_points) arrays
            concatenated across dates).
    """
    from scipy.ndimage import uniform_filter
    deg2m        = 111320.0
    pts_per_date = total_target_samples // len(common_dates)
    np.random.seed(42)

    # lake-wide mean accumulators
    precip_all_list, wind_all_list, div_all_list, time_all_list = [], [], [], []
    wst_all_list = [] if load_wst else None
    precip_samples, wind_samples, div_samples = [], [], []
    wst_samples = [] if load_wst else None
    month_samples, hour_samples = [], []

    # spatial accumulators
    r0s = r1s = c0s = c1s = None
    lakepts_lists = {"precip": [], "wind": [], "wind_u": [], "wind_v": [], "div": []}
    if load_wst:
        lakepts_lists["wst"] = []
    lat_lake = lon_lake = lat_sub = lon_sub = lake_sub = None

    log.info(f"Starting processing for {len(common_dates)} forecast runs...")
    # Loader audit: every run that is dropped, and every cell that is filled, is counted and returned in spatial['audit'].
    audit_skipped, audit_runs = [], []

    for date in tqdm(common_dates, desc="Loading forecasts"):
        ds_p = ds_uwnd = ds_vwnd = ds_wst = None
        try:
            ds_p    = open_nc(path + f'precip_sfc_{date}.nc')
            ds_uwnd = open_nc(path + f'uwnd_10m_{date}.nc')
            ds_vwnd = open_nc(path + f'vwnd_10m_{date}.nc')
            if load_wst:
                ds_wst = open_nc(path + f'st_sfc_{date}.nc')

            ds_p_sel    = select_days_7_to_9_hourly(ds_p)
            ds_uwnd_sel = select_days_7_to_9_wind(ds_uwnd)
            ds_vwnd_sel = select_days_7_to_9_wind(ds_vwnd)

            if any(len(ds.time) == 0 for ds in [ds_p_sel, ds_uwnd_sel, ds_vwnd_sel]):
                log.info(f"No day-7-9 timesteps found for {date}, skipping.")
                audit_skipped.append((str(date), "no day 7-9 time steps"))
                continue

            ds_p_3h    = ds_p_sel.resample(time='3h', closed='right', label='right').sum()
            wind_times = ds_uwnd_sel.time.values
            ds_p_3h    = ds_p_3h.reindex(time=wind_times, method='nearest',
                                          tolerance=np.timedelta64(5, 'm'))
            # whole time slices that found no precipitation stamp within 5 minutes: ffill would repeat the previous 3 h total
            n_ffilled_steps = int(ds_p_3h['precip'].isnull().all(dim=['latitude', 'longitude']).sum())
            ds_p_3h    = ds_p_3h.ffill('time')

            if load_wst:
                ds_wst_sel = select_days_7_to_9_hourly(ds_wst)
                ds_wst_sel = ds_wst_sel.resample(time='3h', closed='right', label='right').mean()
                ds_wst_sel = ds_wst_sel.reindex(time=wind_times, method='nearest',
                                                 tolerance=np.timedelta64(5, 'm')).ffill('time')

            valid_t     = ~np.isnat(ds_p_3h.time.values)
            ds_p_3h     = ds_p_3h.isel(time=valid_t)
            ds_uwnd_sel = ds_uwnd_sel.isel(time=valid_t)
            ds_vwnd_sel = ds_vwnd_sel.isel(time=valid_t)
            if load_wst:
                ds_wst_sel = ds_wst_sel.isel(time=valid_t)

            ds_u = ds_uwnd_sel.interp(latitude=ds_p_3h.latitude,
                                       longitude=ds_p_3h.longitude, method='linear')
            ds_v = ds_vwnd_sel.interp(latitude=ds_p_3h.latitude,
                                       longitude=ds_p_3h.longitude, method='linear')
            if load_wst:
                ds_wst_interp = ds_wst_sel.interp(latitude=ds_p_3h.latitude,
                                                    longitude=ds_p_3h.longitude, method='linear')

            lat_p = ds_p_3h.latitude.values
            lon_p = ds_p_3h.longitude.values
            lake_mask_p = get_lake_mask(ds_p_3h, lsm_ds).ravel()
            m2d_p       = lake_mask_p.reshape(len(ds_p_3h.latitude), len(ds_p_3h.longitude))
            lake_2d     = m2d_p

            if not lake_2d.any():
                log.info(f"No lake points found for {date}, skipping.")
                audit_skipped.append((str(date), "no lake points"))
                continue

            n_t      = len(ds_p_3h.time)
            u_arr    = ds_u['uwnd'].values
            v_arr    = ds_v['vwnd'].values
            wind_arr = np.sqrt(u_arr**2 + v_arr**2)

            if load_wst:
                wst_var = [v for v in ds_wst_interp.data_vars][0]
                wst_arr = ds_wst_interp[wst_var].values  # (n_t, lat, lon)
                wst_arr = wst_arr - 273.15

            # NaN wind cells that are set to 0 before differencing: in total, and on lake pixels
            nan_uv = np.isnan(u_arr) | np.isnan(v_arr)
            audit_runs.append(dict(date=str(date), n_steps=int(n_t), ffilled_precip_steps=n_ffilled_steps,
                                   nan_uv_cells=int(nan_uv.sum()),
                                   nan_uv_lake_cells=int(nan_uv.reshape(n_t, -1)[:, lake_mask_p].sum())))
            div_arr = np.zeros((n_t, len(lat_p), len(lon_p)))
            for t in range(n_t):
                u_filled = np.where(np.isnan(u_arr[t]), 0.0, u_arr[t])
                v_filled = np.where(np.isnan(v_arr[t]), 0.0, v_arr[t])
                u_s = uniform_filter(u_filled, size=smooth_size) if smooth_size > 1 else u_filled
                v_s = uniform_filter(v_filled, size=smooth_size) if smooth_size > 1 else v_filled
                _, du_dx = np.gradient(u_s, lat_p * deg2m, lon_p * deg2m)
                dv_dy, _ = np.gradient(v_s, lat_p * deg2m, lon_p * deg2m)
                div_arr[t] = du_dx + dv_dy

            # ── lake-wide means ───────────────────────────────────────────────
            p_mean    = ds_p_3h['precip'].where(m2d_p).mean(dim=['latitude', 'longitude'])
            wind_mean = wind_arr.reshape(n_t, -1)[:, lake_mask_p].mean(axis=1)
            div_mean  = div_arr.reshape(n_t, -1)[:, lake_mask_p].mean(axis=1)
            precip_all_list.append(p_mean.values)
            wind_all_list.append(wind_mean)
            div_all_list.append(div_mean)
            time_all_list.append(ds_p_3h.time.values)
            if load_wst:
                wst_mean = wst_arr.reshape(n_t, -1)[:, lake_mask_p].mean(axis=1)
                wst_all_list.append(wst_mean)

            # ── grid-point sampling ───────────────────────────────────────────
            p_pix    = ds_p_3h['precip'].values.reshape(n_t, -1)[:, lake_mask_p].ravel()
            wind_pix = wind_arr.reshape(n_t, -1)[:, lake_mask_p].ravel()
            div_pix  = div_arr.reshape(n_t, -1)[:, lake_mask_p].ravel()
            valid    = ~np.isnan(p_pix) & ~np.isnan(wind_pix) & ~np.isnan(div_pix)
            if load_wst:
                wst_pix = wst_arr.reshape(n_t, -1)[:, lake_mask_p].ravel()
                valid  &= ~np.isnan(wst_pix)
            if np.any(valid):
                p_pix, wind_pix, div_pix = p_pix[valid], wind_pix[valid], div_pix[valid]
                if load_wst:
                    wst_pix = wst_pix[valid]
                n_s = min(len(p_pix), pts_per_date)
                idx = np.random.choice(len(p_pix), size=n_s, replace=False)
                precip_samples.append(p_pix[idx])
                wind_samples.append(wind_pix[idx])
                div_samples.append(div_pix[idx])
                if load_wst:
                    wst_samples.append(wst_pix[idx])
                month_grid_full = np.repeat(ds_p_3h.time.dt.month.values, lake_mask_p.sum())
                month_samples.append(month_grid_full[valid][idx])
                hour_grid_full  = np.repeat(ds_p_3h.time.dt.hour.values, lake_mask_p.sum())
                hour_samples.append(hour_grid_full[valid][idx])

            # ── spatial / lakepts ─────────────────────────────────────────────
            lon_2d, lat_2d = np.meshgrid(lon_p, lat_p)
            if r0s is None:
                rows  = np.where(lake_2d.any(axis=1))[0]
                cols  = np.where(lake_2d.any(axis=0))[0]
                r0s, r1s = rows[0], rows[-1] + 1
                c0s, c1s = cols[0], cols[-1] + 1
                lat_sub  = lat_p[r0s:r1s]
                lon_sub  = lon_p[c0s:c1s]
                lake_sub = lake_2d[r0s:r1s, c0s:c1s]
                lat_lake = lat_2d[r0s:r1s, c0s:c1s][lake_sub]
                lon_lake = lon_2d[r0s:r1s, c0s:c1s][lake_sub]

            slc  = (slice(None), slice(r0s, r1s), slice(c0s, c1s))
            flat = lambda a: a.reshape(n_t, -1)[:, lake_sub.ravel()]
            lakepts_lists["precip"].append(flat(ds_p_3h['precip'].values[slc]))
            lakepts_lists["wind"].append(flat(wind_arr[slc]))
            lakepts_lists["div"].append(flat(div_arr[slc]))
            lakepts_lists["wind_u"].append(flat(u_arr[slc]))
            lakepts_lists["wind_v"].append(flat(v_arr[slc]))
            if load_wst:
                lakepts_lists["wst"].append(flat(wst_arr[slc]))

        except Exception as e:
            log.info(f"Error processing {date}: {e}")
            import traceback
            traceback.print_exc()
            audit_skipped.append((str(date), f"exception: {e!r}"))
            continue
        finally:
            for ds in [ds_p, ds_uwnd, ds_vwnd, ds_wst]:
                if ds is not None:
                    ds.close()

    # ── Concatenate ───────────────────────────────────────────────────────────
    lake_means = {
        'precip': np.concatenate(precip_all_list),
        'wind':   np.concatenate(wind_all_list),
        'div':    np.concatenate(div_all_list),
        'time':   np.concatenate(time_all_list),
    }
    if load_wst:
        lake_means['wst'] = np.concatenate(wst_all_list)

    samples = {
        'precip': np.concatenate(precip_samples),
        'wind':   np.concatenate(wind_samples),
        'div':    np.concatenate(div_samples),
        'month':  np.concatenate(month_samples),
        'hour':   np.concatenate(hour_samples),
    }
    if load_wst:
        samples['wst'] = np.concatenate(wst_samples)

    spatial = {
        'lat_lake': lat_lake, 'lon_lake': lon_lake,
        'lat_sub':  lat_sub,  'lon_sub':  lon_sub,
        'lake_sub': lake_sub,
        'lakepts':  {k: np.concatenate(v, axis=0) for k, v in lakepts_lists.items()},
    }

    runs_df = pd.DataFrame(audit_runs)
    spatial['audit'] = dict(
        n_requested=len(common_dates), n_loaded=len(audit_runs), skipped=audit_skipped, runs=runs_df,
        total_steps=int(runs_df['n_steps'].sum()) if len(runs_df) else 0,
        ffilled_precip_steps=int(runs_df['ffilled_precip_steps'].sum()) if len(runs_df) else 0,
        nan_uv_cells=int(runs_df['nan_uv_cells'].sum()) if len(runs_df) else 0,
        nan_uv_lake_cells=int(runs_df['nan_uv_lake_cells'].sum()) if len(runs_df) else 0)
    return lake_means, samples, spatial


# ──────────────────────────────────────────────────────────────────────────
# Seasonal and diurnal cycle removal
# ──────────────────────────────────────────────────────────────────────────
WINDOW_DAYS = 90
SCALE_FLOOR = 0.2          # sigma is never taken below this fraction of the hour-of-day standard deviation


def _day_position(times):
    """Position of each stamp on the seasonal circle (days) and the circle length (days).

    The circle length is the span of the series plus one median time step, so the series is treated as exactly
    one seasonal cycle on a regular grid.

    Parameters:
    times (array-like): Time stamps.

    Returns:
    tuple: (pos, period, idx) with pos the position of each stamp on the circle in days, period the circle length in days
        and idx the stamps as a pandas DatetimeIndex.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(times))
    t = np.asarray((idx - idx[0]) / pd.Timedelta(days=1), float)
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    period = float(t[-1] - t[0]) + step
    return np.mod(t, period), period, idx


def _window_matrix(pos, period, half):
    """0/1 matrix marking which samples lie within +- half days of each other on the circle.

    Parameters:
    pos (np.ndarray): Positions on the circle in days, from _day_position.
    period (float): Circle length in days.
    half (float): Half window width in days.

    Returns:
    np.ndarray: (len(pos), len(pos)) matrix, 1.0 where the circular distance is at most half.
    """
    d = np.abs(pos[:, None] - pos[None, :])
    d = np.minimum(d, period - d)
    return (d <= half).astype(float)


def cyclic_rolling_stats(x, times, window_days=WINDOW_DAYS, by_hour=True, scale_floor=SCALE_FLOOR):
    """Rolling mean and standard deviation on the seasonal circle.

    Parameters:
    x (array-like): Series, no NaN.
    times (array-like): Time stamps, same length.
    window_days (float): Full window width in days (centred, so +- window_days/2).
    by_hour (bool): True uses only samples of the same hour of day; False uses all hours.
    scale_floor (float): Lower bound of sigma as a fraction of the standard deviation of that hour group.

    Returns:
    tuple: (mean, sigma, floored_share) with mean and sigma as arrays aligned with x.
    """
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN/inf; the rolling statistics need a complete series")
    pos, period, idx = _day_position(times)
    groups = np.asarray(idx.hour) if by_hour else np.zeros(len(x), int)
    mu = np.empty(len(x)); sd = np.empty(len(x)); n_floor = 0
    for g in np.unique(groups):
        sel = np.flatnonzero(groups == g)
        W = _window_matrix(pos[sel], period, window_days / 2.0)
        c = W.sum(1)
        if c.min() < 10:
            raise ValueError(f"hour group {g}: a window holds only {int(c.min())} samples; the series is too sparse")
        xs = x[sel]; x0 = xs.mean()
        m = (W @ (xs - x0)) / c
        v = (W @ (xs - x0) ** 2) / c - m ** 2
        v = np.maximum(v, 0.0) * c / np.maximum(c - 1.0, 1.0)
        s = np.sqrt(v)
        floor = scale_floor * xs.std(ddof=1)
        n_floor += int(np.sum(s < floor))
        mu[sel] = m + x0; sd[sel] = np.maximum(s, floor)
    return mu, sd, n_floor / len(x)


def deseasonalise_series(precip, wind, div, times, window_days=WINDOW_DAYS, jitter_seed=42, div_diurnal="keep",
                         scale_floor=SCALE_FLOOR):
    """Cyclic rolling deseasonalisation of the three lake-mean series (used by notebook 00 and spatial_aggregation.filter_series).

    Parameters:
    precip, wind, div (array-like): Lake-mean series (precipitation may contain exact zeros: dry steps are jittered
        with uniform noise below half the smallest positive value).
    times (array-like): Time stamps.
    window_days (float): Rolling window width in days.
    jitter_seed (int): Seed of the dry-step jitter.
    div_diurnal (str): "keep" (default) subtracts a 90-day circular moving average over ALL steps (not hour-specific), so the
        diurnal cycle stays in the divergence; "pooled" full-sample hour-of-day mean and standard deviation; "rolling" the
        hour-specific 90-day mean and scale (as for precipitation and wind). Divergence is standardised later, on each
        fold's training steps.
    scale_floor (float): See cyclic_rolling_stats.

    Returns:
    tuple: (precip, wind, div) as pandas Series on one DatetimeIndex (no NaN, nothing dropped), the hour-of-day
        Series, and a dict of diagnostics (share of floored scales per variable).
    """
    precip = np.asarray(precip, float).copy()
    dry = precip == 0
    if dry.any() and (~dry).any():
        rng = np.random.default_rng(jitter_seed)
        precip[dry] = rng.uniform(0, precip[~dry].min() / 2, size=dry.sum())
    idx = pd.DatetimeIndex(pd.to_datetime(times))
    diag, out = {}, {}
    for name, arr in (("precip", precip), ("wind", np.asarray(wind, float))):
        mu, sd, fl = cyclic_rolling_stats(arr, idx, window_days, True, scale_floor)
        out[name] = pd.Series((arr - mu) / sd, index=idx, name=name); diag[name + "_floored_share"] = fl
    d = np.asarray(div, float)
    if div_diurnal == "pooled":
        hrs = np.asarray(idx.hour); z = np.empty_like(d)
        for h in np.unique(hrs):
            m = hrs == h; sdv = d[m].std(ddof=1)
            z[m] = (d[m] - d[m].mean()) / (sdv if sdv > 0 else 1.0)
        out["div"] = pd.Series(z, index=idx, name="div")
    elif div_diurnal == "keep":
        mu, _, _ = cyclic_rolling_stats(d, idx, window_days, False, scale_floor)
        out["div"] = pd.Series(d - mu, index=idx, name="div")
    elif div_diurnal == "rolling":
        mu, sd, fl = cyclic_rolling_stats(d, idx, window_days, True, scale_floor)
        out["div"] = pd.Series((d - mu) / sd, index=idx, name="div"); diag["div_floored_share"] = fl
    else:
        raise ValueError("div_diurnal must be 'pooled', 'rolling' or 'keep'")
    return out["precip"], out["wind"], out["div"], pd.Series(idx.hour, index=idx), diag


def select_ar_lags(series, max_lags=64, alpha=0.05):
    """Select statistically significant AR lags via the PACF, for the AR filtering step.

    Retains only lags whose PACF value exceeds the significance
    threshold, rather than a fixed-order AR(p), to keep the parameter
    count minimal.

    Parameters:
    series (pandas.Series or np.ndarray): The deseasonalised series.
    max_lags (int): Maximum lag to test. Default 64 (8 days at
        3-hourly resolution: 64 * 3h = 192h = 8 days), as in
        00_preprocessing.ipynb.
    alpha (float): Significance level for the PACF confidence interval.
        Default 0.05.

    Returns:
    list: Significant lag numbers.
    """
    pacf_vals, confint = pacf(series, nlags=max_lags, alpha=alpha)
    lags = []
    for lag in range(1, max_lags + 1):
        ci_half = (confint[lag, 1] - confint[lag, 0]) / 2
        if abs(pacf_vals[lag]) > ci_half:
            lags.append(lag)
    return lags


def fit_ar_masked(filt_full, train_mask, max_lags=64, alpha=0.05):
    """Fit a sparse AR model on the rows selected by train_mask and apply it to the full series.

    The coefficients are estimated only on the rows of train_mask and then
    applied, never re-estimated, to the full series, giving residuals for both
    training and test rows. With a fold's training mask the residuals therefore
    carry no test information. preprocess_fold passes the full-sample mask by
    default (ar_garch_sample='full'), in which case the fit is in-sample; this
    is a known limitation of that design.

    Lags are selected with select_ar_lags and the model is fitted with
    AutoReg on filt_full[train_mask]. AutoRegResults.apply(new_y, refit=False)
    keeps the parameters unchanged and evaluates the residuals at every point
    of new_y using that point's own actual lagged values, which matches a
    manual AR(p) computation exactly, including for sparse non-contiguous lag
    lists.

    Parameters:
    filt_full (pandas.Series): Deseasonalised (filtered) series, full length,
        train and test both present, chronologically ordered. The
        deseasonalisation is estimated on the full series (one seasonal
        cycle), like the AR and GARCH fits.
    train_mask (np.ndarray): Boolean mask into filt_full, True for training
        observations. May be any boolean mask (prefix, suffix or several
        segments, as in the four blocked folds). Lags and coefficients are
        estimated on the concatenated training values (pieces joined by
        data_split.train_estimation_order: end piece before start piece when the
        pieces wrap around the one-year record). Up to max(lags) rows at the
        junction form artificial lag windows (documented in the thesis). The
        residuals on the full series use the fixed coefficients with real
        lagged values, so no junction enters the residuals.
    max_lags (int): Passed to select_ar_lags. Default 64.
    alpha (float): Passed to select_ar_lags. Default 0.05.

    Returns:
    tuple: (resid_full, lags, fit_train, trim).
        resid_full (pandas.Series): Residuals aligned to filt_full's own index,
            shorter by max(lags) at the start (statsmodels' convention: no
            residual exists before enough lagged history is available).
        lags (list): Selected lag list. If empty, resid_full is the
            train-demeaned series and fit_train is None.
        fit_train (statsmodels AutoRegResultsWrapper or None): The AR fit on the
            training rows.
        trim (int): Number of leading observations dropped (0 if lags is
            empty), so callers can realign other full-length arrays (e.g.
            train_mask or an hour-of-day array) to resid_full via array[trim:].
    """
    from data_split import train_estimation_order
    est_order = train_estimation_order(train_mask)
    filt_train = filt_full.iloc[est_order]
    if not np.array_equal(est_order, np.flatnonzero(np.asarray(train_mask))):
        filt_train = filt_train.reset_index(drop=True)   # wrapped (end piece first) join
    lags = select_ar_lags(filt_train, max_lags=max_lags, alpha=alpha)
    if len(lags) == 0:
        resid_full = filt_full - filt_train.mean()
        return resid_full, lags, None, 0

    # Non-contiguous train masks (folds B1, and suffix folds) give a gapped
    # date index; AutoReg only needs the values (lag windows are positional,
    # junction rows are documented in the thesis), so drop the index to avoid
    # statsmodels' frequency warning. Results are identical for contiguous masks.
    fit_train = AutoReg(np.asarray(filt_train), lags=lags, trend='c').fit()
    applied = fit_train.apply(np.asarray(filt_full), refit=False)
    trim = max(lags)
    resid_full = pd.Series(np.asarray(applied.resid), index=filt_full.index[trim:], name=filt_full.name)
    return resid_full, lags, fit_train, trim


def fit_garch_masked(ar_resid_full, train_mask, var_name, hours_full,
                             n_starts=20, seed=1, lb_lags=(8, 16, 24)):
    """Select a volatility model on the rows of train_mask and apply its fixed parameters to the full series.

    Reuses select_volatility_for_variable (menu, AIC, gamma gate) on the
    estimation rows, then applies the chosen model's fitted parameters, never
    re-estimated, to the full series to get standardised residuals for both
    training and test rows. As for the AR step, the estimation rows are those
    of train_mask: training rows only for a fold mask, the full sample when
    preprocess_fold is run with ar_garch_sample='full' (the default).

    The model specification (vol type, p/o/q, distribution) is read back off
    the fitted arch result (fit.model.volatility / fit.model.distribution), so
    the same code path serves whichever candidate in the menu AIC picked.

    Rescaling: arch's rescale=True (used throughout fit_garch_multistart) is a
    fit-time convenience and is not preserved by .fix() on a freshly
    constructed model: a new full-series model built with rescale=True
    computes its own scale factor from the full series instead of reusing the
    training fit's. The approach used here (multiply the full series by the
    training fit's own .scale, construct with rescale=False, then .fix())
    matches a manual GARCH variance recursion to floating-point precision in
    both the training and the out-of-sample region, for GARCH, GJR-GARCH and
    EGARCH specifications.

    Parameters:
    ar_resid_full (pandas.Series): AR-whitened residuals, full length, train
        and test both present.
    train_mask (np.ndarray): Boolean mask into ar_resid_full, True for
        estimation observations. Must align index-for-index with ar_resid_full
        (realign to fit_ar_masked's trim first if ar_resid_full came from that
        function).
    var_name (str): Passed through to select_volatility_for_variable.
    hours_full (np.ndarray or pandas.Series): Hour-of-day, aligned with
        ar_resid_full; sliced to the estimation rows internally for the
        day/night fallback diagnostic.
    n_starts (int): Passed to select_volatility_for_variable. Default 20.
    seed (int): Passed to select_volatility_for_variable. Default 1.
    lb_lags (tuple): Passed to select_volatility_for_variable. Default (8, 16, 24).

    Returns:
    dict: The keys of select_volatility_for_variable's return dict (var_name,
        chosen_label, fit, candidates, verdict, final_residuals,
        fallback_triggered, diagnostic, skip_reasons), where 'fit' and
        'final_residuals' come from the estimation rows only, plus
        'final_residuals_full': the full-length (train+test) standardised
        residuals. If no GARCH-family model is eligible
        (fallback_triggered=True), 'final_residuals_full' is the full-length
        AR residuals unchanged.
    """
    from arch import arch_model

    from data_split import train_estimation_order
    est_order = train_estimation_order(train_mask)
    ar_resid_train = ar_resid_full.iloc[est_order] if hasattr(ar_resid_full, 'iloc') \
        else np.asarray(ar_resid_full)[est_order]
    if hasattr(ar_resid_train, 'reset_index') and \
            not np.array_equal(est_order, np.flatnonzero(np.asarray(train_mask))):
        ar_resid_train = ar_resid_train.reset_index(drop=True)   # wrapped join
    hours_arr = np.asarray(hours_full)
    hours_train = hours_arr[est_order]

    result = select_volatility_for_variable(
        ar_resid_train, var_name, hours_train, n_starts=n_starts, seed=seed, lb_lags=lb_lags)

    if result['fallback_triggered'] or result['fit'] is None:
        result['final_residuals_full'] = ar_resid_full
        return result

    fit_train = result['fit']
    vol = fit_train.model.volatility
    dist_obj = fit_train.model.distribution
    dist_name_map = {'Normal': 'normal', 'StudentsT': 't', 'SkewStudent': 'skewt'}
    dist_name = dist_name_map.get(type(dist_obj).__name__, 'normal')
    vol_type = type(vol).__name__  # 'GARCH' or 'EGARCH' -- matches arch_model's vol= kwarg directly

    s = fit_train.scale
    ar_resid_full_arr = np.asarray(ar_resid_full)
    am_full = arch_model(ar_resid_full_arr * s, mean='Zero',
                          vol=vol_type, p=vol.p, o=vol.o, q=vol.q,
                          dist=dist_name, rescale=False)
    res_fixed = am_full.fix(fit_train.params)
    cv_full = np.asarray(res_fixed.conditional_volatility)

    final_residuals_full = pd.Series(
        (ar_resid_full_arr * s) / cv_full,
        index=(ar_resid_full.index if hasattr(ar_resid_full, 'index') else None),
        name=getattr(ar_resid_full, 'name', var_name),
    )
    result['final_residuals_full'] = final_residuals_full
    return result


def train_only_pit(train_vals, query_vals, eps=None):
    """Empirical rank transform (PIT) fitted on the training sample only and applied to arbitrary query values.

    Replaces rank(x)/(n+1) over the full (train+test) sample with a transform
    that uses no test information. On the training sample itself it reduces
    exactly to scipy.stats.rankdata(train_vals) / (n_train + 1). A query value
    outside the training range never returns exactly 0.0 or 1.0 (which would
    break any downstream log-density evaluation); it is clipped instead, in
    the same spirit as the zero-precipitation jittering used elsewhere in the
    pipeline (the preprocessing chapter).

    Parameters:
    train_vals (np.ndarray): Training-portion values the empirical CDF is
        built from.
    query_vals (np.ndarray): Values to transform. Pass the full (train+test)
        series to get pseudo-observations for both, using only the training
        distribution.
    eps (float): Clip boundary. Defaults to 1 / (2 * (n_train + 1)), half the
        smallest step the training-only rank transform itself can produce, so
        the clip never binds for an in-range training value and only affects a
        query value that falls outside the training range. Optional.

    Returns:
    np.ndarray: Pseudo-observations in [eps, 1-eps], same length as query_vals.
    """
    train_vals = np.asarray(train_vals)
    query_vals = np.asarray(query_vals)
    n_train = len(train_vals)
    if eps is None:
        eps = 1.0 / (2 * (n_train + 1))
    sorted_train = np.sort(train_vals)
    ranks = np.searchsorted(sorted_train, query_vals, side='right')
    u = ranks / (n_train + 1)
    return np.clip(u, eps, 1 - eps)


def regime_train_only_pit(x, regime, train_mask):
    """Regime-conditional, train-only empirical PIT.

    u_t = F_hat_{r(t)}(x_t), where F_hat_r is the empirical CDF of the
    TRAINING observations that lie in regime r(t); the regime of step t is
    taken from the observed covariate at t (w is a covariate, not a
    target, so using the test-step regime is legitimate). Each regime's
    ECDF goes through train_only_pit, so the clip eps depends on that
    regime's training count.

    Parameters:
    x (np.ndarray): Series to transform, full length (train + test).
    regime (np.ndarray): 0/1 (or bool) regime label per step, full length.
    train_mask (np.ndarray): Boolean, True for training steps.

    Returns:
    np.ndarray: Pseudo-observations, same length as x.
    """
    x = np.asarray(x, dtype=float)
    regime = np.asarray(regime).astype(int)
    train_mask = np.asarray(train_mask, dtype=bool)
    u = np.full(len(x), np.nan)
    for r in (0, 1):
        in_r = regime == r
        tr = train_mask & in_r
        if tr.sum() < 2:
            raise ValueError(f"regime {r}: fewer than 2 training observations")
        u[in_r] = train_only_pit(x[tr], x[in_r])
    return u


def kernel_conditional_pit(x, w, train_mask, h, exclude_block=0, eps=None, leave_self_out=False):
    """Kernel-weighted conditional PIT (full conditional marginals) given a covariate w.

        u_t = sum_s K((w_s - w_t)/h) 1{x_s <= x_t} / (sum_s K((w_s - w_t)/h) + 1),

    with s running over the training steps and a Gaussian kernel
    K(z) = exp(-z^2/2). The covariate w is standardised with the fold-training
    mean and SD, so h is in SD units. The "+1" is K(0), the analogue of the
    n+1 in the rank transform: with a huge h this reduces to the unconditional
    train-only PIT. Because w is a covariate, using the observed w_t of a test
    step is legitimate; x of a test step never enters the weights. This is the
    indicator (non-smoothed in x) version of the kernel conditional CDF
    estimator of Li and Racine (2008).

    Li, Q., & Racine, J. S. (2008). Nonparametric estimation of conditional
    CDF and quantile functions with mixed categorical and continuous data.
    Journal of Business & Economic Statistics, 26(4), 423-434.

    Parameters:
    x (np.ndarray): Series to transform, full length (train + test).
    w (np.ndarray): Standardised covariate, full length.
    train_mask (np.ndarray): Boolean, True for training steps.
    h (float): Kernel bandwidth in SD units of w.
    exclude_block (int): Leave-block-out for training steps only: training
        steps s with |s - t| <= exclude_block get zero weight when scoring
        step t (used by the bandwidth tuner so a step does not inform its own
        transform). Pass 0 for the final pseudo-observations. Default 0.
    eps (float): Clip boundary. Default None: 1 / (2 * (sum K + 1)) per step.
    leave_self_out (bool): If True, a training step does not count itself (its
        K(0) term and its own indicator) when its own PIT is computed, so
        training and validation pseudo-observations are out-of-sample in the
        same way as test steps. Used for the final pseudo-observations; the +1
        in the denominator is kept. Default False.

    Returns:
    np.ndarray: Pseudo-observations of length len(x), clipped to [eps, 1-eps].
    """
    x = np.asarray(x, float); w = np.asarray(w, float)
    tr = np.flatnonzero(np.asarray(train_mask, bool))
    n = len(x)
    xt, wt = x[tr], w[tr]
    u = np.empty(n)
    chunk = 512
    for a in range(0, n, chunk):
        idx = np.arange(a, min(a + chunk, n))
        K = np.exp(-0.5 * ((wt[None, :] - w[idx, None]) / h) ** 2)
        if exclude_block > 0:
            K[np.abs(tr[None, :] - idx[:, None]) <= exclude_block] = 0.0
        if leave_self_out:
            K[tr[None, :] == idx[:, None]] = 0.0
        ind = (xt[None, :] <= x[idx, None])
        den = K.sum(axis=1) + 1.0
        ui = (K * ind).sum(axis=1) / den
        e = 1.0 / (2.0 * den) if eps is None else eps
        u[idx] = np.clip(ui, e, 1 - e)
    return u


def tune_kernel_bandwidth(series, w_by_fold, train_by_fold, purge_by_fold,
                          grid=None, h0_grid=None, step=3):
    """Choose one bandwidth h (SD units of w) for the supplied training folds by least-squares cross-validation.

    Notebook 00 calls this separately per outer fold with a one-key
    dictionary, so a fold's bandwidth depends only on its own training set.

    Method of Li and Racine (2008), Section 2.3: the bandwidths of the
    conditional CDF estimator are chosen by least-squares cross-validation of
    the conditional PDF (criterion of Hall, Racine and Li, 2004, Section 2, as
    restated in eq. 13 of Li and Racine, 2008, with weight s = 1),

        CV(h0, h) = 1/m sum_i G_{-i}(w_i) / mu_{-i}(w_i)^2 - 2/m sum_i g_{-i}(x_i, w_i) / mu_{-i}(w_i),
        mu_{-i}(w)   = sum_j K_h(w_j - w),
        g_{-i}(x, w) = sum_j K_h(w_j - w) k_{h0}(x - x_j),
        G_{-i}(w)    = sum_j sum_l K_h(w_j - w) K_h(w_l - w) (k_{h0} * k_{h0})(x_j - x_l),

    with Gaussian kernels, so the integral in G is the N(0, 2 h0^2) density at
    x_j - x_l (no numerical integration). h0 smooths the dependent variable
    and is only a nuisance parameter of the criterion. Because the CDF
    estimator used here does not smooth x (the indicator version, eq. 4 of Li
    and Racine, 2008), only the covariate bandwidth is kept, and it is rescaled
    to the rate of the CDF estimator (Li and Racine, 2008, text before eq. 14):

        h = h_cv * n^(1/(5+q) - 1/(4+q)),  q = 1 continuous covariate.

    Observations are serially dependent, so "leave one out" becomes "leave
    block out": training steps within the fold's purge distance of step i get
    zero weight (same purge as the folds). i runs over every `step`-th
    training step. The criterion is summed over both margins and over the
    supplied training sets. In notebook 00 exactly one outer fold is supplied
    per call, so h is shared between the two margins of that fold, not among
    all four outer folds.

    Hall, P., Racine, J., & Li, Q. (2004). Cross-validation and the estimation
    of conditional probability densities. Journal of the American Statistical
    Association, 99(468), 1015-1026.

    Li, Q., & Racine, J. S. (2008). Nonparametric estimation of conditional
    CDF and quantile functions with mixed categorical and continuous data.
    Journal of Business & Economic Statistics, 26(4), 423-434.

    Parameters:
    series (dict): {'precip': x_full, 'wind': x_full}, or
        {fold: {'precip': ..., 'wind': ...}} when the residuals differ by fold.
        x is standardised inside by its training mean/SD, so h0 is in SD units.
    w_by_fold (dict): fold -> covariate array, already standardised with that
        fold's training mean/SD.
    train_by_fold (dict): fold -> boolean training mask.
    purge_by_fold (dict): fold -> purge distance (positions) for leave-block-out.
    grid (array-like): Candidate h_cv values. Default np.geomspace(0.1, 5.0, 12).
    h0_grid (array-like): Candidate h0 values. Default np.geomspace(0.05, 0.8, 5).
    step (int): Evaluate the criterion at every step-th training position.
        Default 3.

    Returns:
    tuple: (h, df). h (float) is the rescaled bandwidth; df (pandas.DataFrame)
        has columns h0, h_cv, cv, and h_cv, h0, n_bar, rescale and
        at_grid_edge of the optimum are in df.attrs.
    """
    grid = np.geomspace(0.1, 5.0, 12) if grid is None else np.asarray(grid, float)
    h0_grid = np.geomspace(0.05, 0.8, 5) if h0_grid is None else np.asarray(h0_grid, float)
    cv = np.zeros((len(h0_grid), len(grid))); cnt = 0
    n_sizes = []
    for f in train_by_fold:
        tr = np.asarray(train_by_fold[f], bool); w = np.asarray(w_by_fold[f], float)
        trpos = np.flatnonzero(tr); pos = trpos[::step]; n_sizes.append(len(trpos))
        far = np.abs(trpos[None, :] - pos[:, None]) > purge_by_fold[f]          # (m, n): usable j for step i
        Ks = []
        for h in grid:
            K = np.exp(-0.5 * ((w[trpos][None, :] - w[pos][:, None]) / h) ** 2) * far
            Ks.append((K, K.sum(axis=1)))
        for x in (series[f] if f in series else series).values():
            x = np.asarray(x, float)
            mu, sd = x[trpos].mean(), x[trpos].std()
            xt = (x[trpos] - mu) / sd; xi = (x[pos] - mu) / sd
            for a, h0 in enumerate(h0_grid):
                d = xt[:, None] - xt[None, :]
                Psi = np.exp(-0.25 * (d / h0) ** 2) / (2.0 * h0 * np.sqrt(np.pi))     # N(0, 2 h0^2) density
                Om = np.exp(-0.5 * ((xi[:, None] - xt[None, :]) / h0) ** 2) / (h0 * np.sqrt(2 * np.pi))
                for b, (K, den) in enumerate(Ks):
                    ok = den > 1e-12
                    quad = np.einsum("ij,ij->i", K @ Psi, K)
                    g = np.einsum("ij,ij->i", K, Om)
                    cv[a, b] += float((quad[ok] / den[ok] ** 2 - 2.0 * g[ok] / den[ok]).sum())
            cnt += len(pos)
    cv /= cnt
    n_bar = float(np.mean(n_sizes)); q = 1
    rescale = n_bar ** (1.0 / (5 + q) - 1.0 / (4 + q))
    a, b = np.unravel_index(np.argmin(cv), cv.shape)
    h_cv = float(grid[b]); h = h_cv * rescale
    df = pd.DataFrame([dict(h0=float(h0_grid[i]), h_cv=float(grid[j]), cv=float(cv[i, j]))
                       for i in range(len(h0_grid)) for j in range(len(grid))])
    df.attrs.update(h_cv=h_cv, h0=float(h0_grid[a]), n_bar=n_bar, rescale=float(rescale),
                    at_grid_edge=bool(b in (0, len(grid) - 1)))
    return h, df


def decile_uniformity(u, w_std, mask, label, n_bins=10):
    """Uniformity of pseudo-observations within deciles of w.

    Per decile the KS distance to U(0,1) and the tail shares P(u>0.95) and
    P(u<0.05) are computed. Deciles are cut at the quantiles of w over `mask`;
    deciles with fewer than 30 steps are skipped.

    Parameters:
    u (np.ndarray): Pseudo-observations.
    w_std (np.ndarray): Standardised covariate, same length as u.
    mask (np.ndarray): Boolean mask of the steps used to cut and fill the deciles.
    label (str): Prefix for the group names.
    n_bins (int): Number of bins. Default 10.

    Returns:
    list: One dict per retained decile with keys group, n, mean, P_gt95,
        P_lt05 and KS_D.
    """
    from scipy.stats import kstest
    u = np.asarray(u); w = np.asarray(w_std); mask = np.asarray(mask, bool)
    edges = np.quantile(w[mask], np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-9
    rows = []
    for k in range(n_bins):
        sel = mask & (w >= edges[k]) & (w < edges[k + 1])
        if sel.sum() >= 30:
            rows.append(dict(group=f"{label} decile {k + 1}", n=int(sel.sum()), mean=u[sel].mean(),
                             P_gt95=(u[sel] > 0.95).mean(), P_lt05=(u[sel] < 0.05).mean(),
                             KS_D=kstest(u[sel], 'uniform').statistic))
    return rows


def uniformity_row(u, label):
    """Summarise a pseudo-observation sample by mean, SD, tail shares and KS distance to U(0,1).

    Parameters:
    u (np.ndarray): Pseudo-observations.
    label (str): Group name stored in the result.

    Returns:
    dict: Keys group, n, mean, sd, P_gt95, P_lt05 and KS_D.
    """
    from scipy.stats import kstest
    u = np.asarray(u)
    return dict(group=label, n=len(u), mean=u.mean(), sd=u.std(),
                P_gt95=(u > 0.95).mean(), P_lt05=(u < 0.05).mean(),
                KS_D=kstest(u, 'uniform').statistic)


def fold_residuals_one(filt, hours_full, common_idx, train_pos, var_name,
                       max_lags=64, n_starts=20, seed=1, verbose=True):
    """Fit AR and a GARCH-family volatility model on one filtered series and return standardised residuals over common_idx.

    The models are fitted on the rows of one estimation mask (the fold's
    training rows, or all rows for ar_garch_sample='full') and the
    standardised residuals are returned over common_idx, the shared time axis
    of all folds. See preprocess_fold for the estimation-mask convention (rows
    before common_idx[0] take the label of position 0) and the junction caveat
    for non-contiguous training masks.

    Parameters:
    filt (pandas.Series): Deseasonalised series on the full index (before the
        AR trim).
    hours_full (pandas.Series): Hour of day, .loc-addressable by the filtered index.
    common_idx (pandas.DatetimeIndex): Shared usable index; must be a
        contiguous sub-range of filt.index.
    train_pos (np.ndarray): Boolean over common_idx positions marking the
        estimation rows.
    var_name (str): Variable name used in log messages.
    max_lags (int): Passed to fit_ar_masked. Default 64.
    n_starts (int): Passed to fit_garch_masked. Default 20.
    seed (int): Passed to fit_garch_masked. Default 1.
    verbose (bool): Log the selected AR lags and volatility model. Default True.

    Returns:
    tuple: (resid, lags, label, diag).
        resid (np.ndarray): Standardised residuals over common_idx.
        lags (list): Selected AR lags.
        label (str): Chosen volatility model label (None if the fallback ran).
        diag (dict): model, fallback, persistence, Ljung-Box p-values at lags
            8/16/24 of the estimation-row standardised residuals, and the
            number of estimation rows (n_est).
    """
    train_pos = np.asarray(train_pos, dtype=bool)
    if len(train_pos) != len(common_idx):
        raise ValueError("train_pos must have one entry per element of common_idx")
    filt_idx = filt.index
    off = filt_idx.get_loc(common_idx[0])
    if not (filt_idx[off:off + len(common_idx)] == common_idx).all():
        raise ValueError("common_idx is not a contiguous sub-range of the filtered index")
    est_mask = np.empty(len(filt_idx), dtype=bool)
    est_mask[off:off + len(common_idx)] = train_pos
    est_mask[:off] = train_pos[0]
    est_mask[off + len(common_idx):] = train_pos[-1]

    resid_full, lags, _, trim = fit_ar_masked(filt, est_mask, max_lags=max_lags)
    if verbose:
        log.info(f"  [{var_name}] AR lags={lags}")
    idx_f = resid_full.index
    if not common_idx.isin(idx_f).all():
        raise ValueError("AR trim removed rows inside common_idx; common_idx must start "
                         f">= max_lags={max_lags} rows after the filtered series start")
    est_f = pd.Series(est_mask, index=filt_idx).loc[idx_f].values
    hours_f = np.asarray(hours_full.loc[idx_f])
    gres = fit_garch_masked(resid_full, est_f, var_name, hours_f,
                                   n_starts=n_starts, seed=seed)
    fr = gres['final_residuals_full']
    out = np.asarray(fr.loc[common_idx] if hasattr(fr, 'loc') else fr)
    if verbose:
        log.info(f"  [{var_name}] volatility model: {gres['chosen_label']}")
    diag = dict(model=gres['chosen_label'], fallback=bool(gres['fallback_triggered']),
                persistence=np.nan, lb_p8=np.nan, lb_p16=np.nan, lb_p24=np.nan,
                n_est=int(est_f.sum()))
    try:
        cand = gres.get('candidates') or {}
        if gres['chosen_label'] in cand:
            diag['persistence'] = float(cand[gres['chosen_label']]['persistence'])
        z = np.asarray(gres['final_residuals'], dtype=float)
        lb = acorr_ljungbox(z, lags=[8, 16, 24], return_df=True)['lb_pvalue'].values
        diag.update(lb_p8=float(lb[0]), lb_p16=float(lb[1]), lb_p24=float(lb[2]))
    except Exception:
        pass   # diagnostics are informational; never break the fit
    if diag['fallback']:
        log.warning(f"[{var_name}] GARCH FALLBACK: no eligible volatility model, "
                    f"returning unstandardised AR residuals (n_est={diag['n_est']}).")
    return out, lags, gres['chosen_label'], diag


def ar_reach(lags):
    """Longest retained AR lag over the variables of `lags`.

    Parameters:
    lags (dict): {variable: list of AR lags}.

    Returns:
    int: The largest retained lag, or 1 if none was retained.
    """
    return max([max(v) for v in lags.values() if len(v)] + [1])


def fold_regime_pit(final, div_raw, common_idx, train_pos, h=None, *, min_len):
    """Divergence standardisation, regime split and conditional PIT for one fold, given standardised residuals over common_idx.

    With h given, u1/u2 are the kernel-weighted conditional pseudo-observations
    (main analysis, kernel_conditional_pit) and the binary-regime PIT is kept
    as u1_regime/u2_regime (ablation). With h=None, u1/u2 are the binary-regime
    PIT. The regime split of w (median of the training steps) is returned in
    both cases, because the binary-split copula models need it.
    u1_uncond/u2_uncond is the unconditional train-only PIT (ablation).

    Parameters:
    final (dict): {'precip': array, 'wind': array} standardised residuals over
        common_idx.
    div_raw (pandas.Series): Deseasonalised divergence, indexed so that
        div_raw.loc[common_idx] works.
    common_idx (pandas.DatetimeIndex): The shared usable index.
    train_pos (np.ndarray): Boolean over common_idx positions, the fold's
        training mask.
    h (float): Kernel bandwidth in SD units of w. Default None: binary-regime PIT.
    min_len (int): Minimum length of a contiguous training run for the
        Politis-White diagnostic b: the longest retained AR lag of
        precipitation and wind (see ar_reach). Keyword-only, no default.

    Returns:
    dict: The keys of preprocess_fold's return dict minus the AR/GARCH
        metadata (ar_lags, vol_model).
    """
    train_pos = np.asarray(train_pos, dtype=bool)
    div = np.asarray(div_raw.loc[common_idx], dtype=float)
    x_mean, x_sd = div[train_pos].mean(), div[train_pos].std()
    x_std = (div - x_mean) / x_sd
    med = float(np.median(x_std[train_pos]))
    high = x_std >= med
    regime = high.astype(int)
    u1_r = regime_train_only_pit(final['precip'], regime, train_pos)
    u2_r = regime_train_only_pit(final['wind'], regime, train_pos)
    if h is None:
        u1, u2 = u1_r, u2_r
    else:
        u1 = kernel_conditional_pit(final['precip'], x_std, train_pos, h, leave_self_out=True)
        u2 = kernel_conditional_pit(final['wind'], x_std, train_pos, h, leave_self_out=True)
    u1_un = train_only_pit(final['precip'][train_pos], final['precip'])
    u2_un = train_only_pit(final['wind'][train_pos], final['wind'])
    # b: Politis-White DIAGNOSTIC on the rank-transformed (pseudo-observation scale) series, computed inside
    # each contiguous run of the training rows (no artificial join in fold B1). It is reported next to the
    # leakage length L (purge, LSCV leave-block, Acar CV) and only drives descriptive bootstrap intervals.
    from data_split import pw_block_length_segments
    b = pw_block_length_segments([final['precip'], final['wind'], x_std], train_pos, min_len=min_len)
    return dict(train=train_pos, final_resid=final, div=div, x_std=x_std, x_mean=x_mean,
                x_sd=x_sd, div_median=med, high_mask=high, low_mask=~high,
                u1=u1, u2=u2, u1_uncond=u1_un, u2_uncond=u2_un, u1_regime=u1_r, u2_regime=u2_r,
                kernel_h=h, b=b)


def preprocess_fold(precip_filt, wind_filt, div_raw, hours_full, common_idx, train_pos,
                    max_lags=64, n_starts=20, seed=1, verbose=True,
                    ar_garch_sample='full', resid_cache=None, h=None):
    """Run the preprocessing chain for one fold and return kernel-conditional pseudo-observations for the full usable series.

    Chain: sparse AR (coefficients applied with fixed values to the full
    series), then GARCH-family volatility selection (fixed parameters applied
    to the full series; the hour of day enters only the day/night diagnostic
    used when no candidate is eligible, not the variance equation). With the
    default ar_garch_sample='full' both are estimated once on the full usable
    sample and shared by all folds, so the standardised residuals are not
    fold-wise (a known limitation of this design); with 'fold' they use the
    fold's training rows only. The remaining steps are always fold-wise and
    train-only: divergence standardised with the fold-training mean/SD, regime
    split at the fold-training median (high = at or above), kernel-weighted
    conditional train-only PIT when h is given (else the binary-regime PIT),
    plus the unconditional train-only PIT kept for the ablation without
    conditioning.

    The filtered series are longer than common_idx (the first max_lags rows
    are lost to the AR lags). Rows before common_idx[0] are labelled like
    position 0 for the estimation mask; all outputs are restricted to
    common_idx so every fold lives on the same time axis.

    Non-contiguous training masks (B1; suffix folds A2/B2) concatenate
    training pieces for lag and volatility selection; up to max_lags rows at a
    junction form artificial lag windows (documented in the thesis). The
    residuals themselves use fixed parameters on the real full series.

    Parameters:
    precip_filt (pandas.Series): Deseasonalised precipitation on the full
        index (before the AR trim).
    wind_filt (pandas.Series): Deseasonalised wind on the full index (before
        the AR trim).
    div_raw (pandas.Series): Deseasonalised divergence (90-day mean removed),
        indexed so that div_raw.loc[common_idx] works.
    hours_full (pandas.Series): Hour of day, .loc-addressable by the filtered index.
    common_idx (pandas.DatetimeIndex): The global usable index (positions 0..n-1
        used by data_split.make_blocked_folds).
    train_pos (np.ndarray): Boolean over common_idx positions, the fold's
        training mask.
    max_lags (int): Passed to the AR selection. Default 64.
    n_starts (int): Passed to the GARCH selection. Default 20.
    seed (int): Passed to the GARCH selection. Default 1.
    verbose (bool): Log the selected AR lags and volatility models. Default True.
    ar_garch_sample (str): 'full' (default) fits AR and GARCH once on the full
        usable sample, so every fold shares the same standardised residuals;
        only the divergence standardisation, the regime split and the PIT (all
        train-only) are fold-specific. 'fold' refits AR and GARCH on the
        fold's training rows only (no test information in the residuals, but
        unstable on the half-year training sets; see the health table of
        notebook 07).
    resid_cache (dict): Optional dict reused across folds so the full-sample
        fits are done once (only used with ar_garch_sample='full').
    h (float): Kernel bandwidth in SD units of the divergence covariate.
        Default None: binary-regime PIT.

    Returns:
    dict: train (position mask), final_resid ({'precip','wind'} arrays over
        common_idx), div (raw), x_std (standardised), x_mean, x_sd, div_median,
        high_mask, low_mask, u1, u2, u1_uncond, u2_uncond (unconditional
        train-only PIT), u1_regime, u2_regime, kernel_h, b, ar_lags
        ({var: list}) and vol_model ({var: label}). u1/u2 are the
        kernel-conditional PIT when h is given, otherwise the binary-regime PIT.
    """
    if ar_garch_sample not in ('full', 'fold'):
        raise ValueError("ar_garch_sample must be 'full' or 'fold'")
    train_pos = np.asarray(train_pos, dtype=bool)
    full = ar_garch_sample == 'full'
    fit_mask = np.ones(len(common_idx), dtype=bool) if full else train_pos
    final, lags, lab = {}, {}, {}
    for var, filt in (('precip', precip_filt), ('wind', wind_filt)):
        ck = (var, 'full')
        if full and resid_cache is not None and ck in resid_cache:
            final[var], lags[var], lab[var] = resid_cache[ck]
            continue
        final[var], lags[var], lab[var], _ = fold_residuals_one(
            filt, hours_full, common_idx, fit_mask, var,
            max_lags=max_lags, n_starts=n_starts, seed=seed, verbose=verbose)
        if full and resid_cache is not None:
            resid_cache[ck] = (final[var], lags[var], lab[var])
    out = fold_regime_pit(final, div_raw, common_idx, train_pos, h=h, min_len=ar_reach(lags))
    out['ar_lags'] = lags
    out['vol_model'] = lab
    return out


def open_nc(filepath):
    """Open a forecast NetCDF file, normalising its time dimension name.

    Parameters:
    filepath (str): Path to the .nc file.

    Returns:
    xarray.Dataset: The opened dataset, with 'dim0' swapped for 'time' if
        the file uses that convention instead of a native time dimension.
    """
    ds = xr.open_dataset(filepath)
    if 'dim0' in ds.dims and 'time' not in ds.dims and 'time' in ds.coords:
        ds = ds.swap_dims({'dim0': 'time'})
    return ds


def date_key(f, sep):
    """Extract the forecast initialisation date string from a filename.

    Parameters:
    f (str): Filename.
    sep (str): Separator splitting the filename into parts; the date is
        taken from the second part (index 1).

    Returns:
    str: The date string, with a trailing '.nc' stripped if present.
    """
    return f.split(sep)[1].replace('.nc', '')

def get_lake_mask(ds_p, lsm_ds):
    """Boolean mask of grid points inside the Lake Victoria bounding box and over water.

    Parameters:
    ds_p (xarray.Dataset): Dataset providing the target latitude/longitude
        grid to interpolate the land-sea mask onto.
    lsm_ds (xarray.Dataset): Land-sea mask dataset, with a
        'land_binary_mask' variable.

    Returns:
    np.ndarray: 2-D boolean array, True where the point is within the
        lake's bounding box (-3.0 to 0.5 N, 31.6 to 34.9 E) and the
        interpolated land fraction is below 0.5.
    """
    lon_2d, lat_2d = np.meshgrid(ds_p.longitude.values, ds_p.latitude.values)
    lsm_p = lsm_ds['land_binary_mask'].interp(
        latitude=ds_p.latitude, longitude=ds_p.longitude,
        method='linear').values
    return ((lat_2d >= -3.0) & (lat_2d <= 0.5) &
            (lon_2d >= 31.6) & (lon_2d <= 34.9) &
            (lsm_p < 0.5))


def _spinup_worker_init(path, lsm_ds):
    """Initialiser for the spin-up worker pool: stash path and lsm_ds once per worker process.

    Avoids pickling them into every per-date task.

    Parameters:
    path (str): Directory containing the per-date NetCDF files.
    lsm_ds (xarray.Dataset): Land-sea mask dataset.
    """
    global _SP_PATH, _SP_LSM_DS
    _SP_PATH = path
    _SP_LSM_DS = lsm_ds


def _spinup_load_and_cache_date(args):
    """Load and interp one date's files ONCE, and cache what pass 2 needs.

    Top-level (picklable) function for ProcessPoolExecutor -- must not be
    nested or a lambda. Runs in a worker process; see _spinup_worker_init
    for path/lsm_ds setup.

    Does the expensive part (NetCDF open + xarray interp of wind onto the
    precip grid) exactly once per date, then writes the compact arrays
    pass 2 needs (p_arr, w_arr, lake_mask, lead_days -- not the full
    xarray Datasets) to a scratch .npz. Pass 2 reads that file instead of
    re-opening/re-interpolating the NetCDFs, cutting this function's I/O
    and interp cost roughly in half relative to two independent passes
    that each redo the load.

    Parameters:
    args (tuple): (date, cache_dir).

    Returns:
    tuple: (date, (wet_precip_lake, wind_lake)_or_None, error_str_or_None).
    """
    import os
    date, cache_dir = args
    try:
        ds_p    = open_nc(_SP_PATH + f'precip_sfc_{date}.nc')
        ds_uwnd = open_nc(_SP_PATH + f'uwnd_10m_{date}.nc')
        ds_vwnd = open_nc(_SP_PATH + f'vwnd_10m_{date}.nc')
        lake_mask = get_lake_mask(ds_p, _SP_LSM_DS)
        ds_u = ds_uwnd.interp(latitude=ds_p.latitude, longitude=ds_p.longitude,
                               time=ds_p.time, method='linear')
        ds_v = ds_vwnd.interp(latitude=ds_p.latitude, longitude=ds_p.longitude,
                               time=ds_p.time, method='linear')
        p_arr = ds_p['precip'].values
        w_arr = np.sqrt(ds_u['uwnd'].values**2 + ds_v['vwnd'].values**2)
        times = ds_p.time.values

        p_lake = p_arr.reshape(len(ds_p.time), -1)[:, lake_mask.ravel()].ravel()
        w_lake = w_arr.reshape(len(ds_p.time), -1)[:, lake_mask.ravel()].ravel()

        init = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}")
        lead_days = np.array([
            int((pd.Timestamp(t) - init).total_seconds() / 86400) + 1 for t in times
        ])

        np.savez_compressed(
            os.path.join(cache_dir, f'{date}.npz'),
            p_arr=p_arr, w_arr=w_arr, lake_mask=lake_mask, lead_days=lead_days,
        )
        out = (p_lake[p_lake > 0], w_lake)
        for ds in [ds_p, ds_uwnd, ds_vwnd]: ds.close()
        return date, out, None
    except Exception as e:
        return date, None, str(e)


def _spinup_count_from_cache(args):
    """Apply thresholds to one date's cached arrays and count storm objects.

    Top-level (picklable) function for ProcessPoolExecutor -- must not be
    nested or a lambda. Pure array compute -- reads the .npz written by
    _spinup_load_and_cache_date, no NetCDF I/O or interpolation here, so
    that cost from pass 1 is not repeated.

    Parameters:
    args (tuple): (date, cache_dir, thresholds), where thresholds is
        pass 1's pct -> (precip_threshold, wind_threshold) dict.

    Returns:
    tuple: (date, local_results_or_None, error_str_or_None). local_results
        is pct -> {lead_day: [n_precip, n_wind, n_compound]} for this
        date only; the caller sums these across dates.
    """
    import os
    from scipy.ndimage import label
    date, cache_dir, thresholds = args
    conn8 = np.ones((3, 3), dtype=int)
    try:
        cached = np.load(os.path.join(cache_dir, f'{date}.npz'))
        p_arr, w_arr, lake_mask, lead_days = (
            cached['p_arr'], cached['w_arr'], cached['lake_mask'], cached['lead_days'])
        local_results = {pct: {} for pct in thresholds}
        for i in range(len(lead_days)):
            lead_day = int(lead_days[i])
            for pct, (thresh_p, thresh_w) in thresholds.items():
                exc_p = (p_arr[i] > thresh_p) & lake_mask
                exc_w = (w_arr[i] > thresh_w) & lake_mask
                _, n_p = label(exc_p,         structure=conn8)
                _, n_w = label(exc_w,         structure=conn8)
                _, n_c = label(exc_p & exc_w, structure=conn8)
                if lead_day not in local_results[pct]:
                    local_results[pct][lead_day] = [0, 0, 0]
                local_results[pct][lead_day][0] += n_p
                local_results[pct][lead_day][1] += n_w
                local_results[pct][lead_day][2] += n_c
        return date, local_results, None
    except Exception as e:
        return date, None, str(e)


def compute_spinup_thresholds(common_dates, path, lsm_ds, threshold_levels=None,
                               n_workers=None, cache_dir=None):
    """Count extreme-exceedance storm objects per forecast lead day, for the spin-up cutoff analysis.

    Generates the data behind the spin-up cutoff figure: for each
    requested percentile threshold, computes lake-wide precipitation and
    wind exceedance thresholds from a first pass over all dates, then in
    a second pass counts connected exceedance regions ("storm objects",
    8-connected) per forecast lead day under three definitions
    (precipitation only, wind only, and both simultaneously).

    Each date's NetCDF files are opened and interpolated exactly ONCE
    (pass 1), not twice. Pass 1 also writes each date's compact arrays
    (p_arr, w_arr, lake_mask, lead_days -- not the full Datasets) to a
    scratch .npz in cache_dir; pass 2 reads that cache and does pure
    array math (threshold + scipy.ndimage.label), with no further NetCDF
    I/O or xarray interp. This avoids the 2x-reads tradeoff of running
    two fully independent passes over the raw files, at the cost of one
    local disk write/read per date -- much cheaper than re-doing the
    interp, but real: cache_dir needs write access and enough scratch
    space for len(common_dates) compact arrays (a few MB each, typically
    -- check your JASMIN quota if common_dates is very large). The cache
    is written to a fresh tempfile.mkdtemp() and removed at the end
    unless you pass your own cache_dir, in which case cleanup is your
    responsibility.

    Both passes still parallelize across common_dates via a
    ProcessPoolExecutor (one date per task, independent across dates,
    single flat pool per pass -- no per-item nesting, so n_workers can be
    set directly to whatever's free without an oversubscription risk from
    this function's own structure). It can still oversubscribe if called
    back-to-back with another parallel step that hasn't released its
    workers yet -- confirm nothing else is holding a pool open before
    raising n_workers.

    Parameters:
    common_dates (list): Forecast initialisation date strings.
    path (str): Directory containing the per-date NetCDF files (same
        naming convention as load_forecast_data).
    lsm_ds (xarray.Dataset): Land-sea mask dataset, passed to get_lake_mask.
    threshold_levels (list): Upper-tail percentages to threshold at (e.g.
        5 for the top 5%). Defaults to [5] if None. Optional.
    n_workers (int): Worker processes per pass's date pool. If None
        (default), resolved via resolve_max_workers(reserve=2) at call
        time -- available cores minus 2, re-checked on every call.
    cache_dir (str): Where to write pass 1's per-date .npz cache. Defaults
        to a fresh temp directory, removed automatically at the end.
        Pass your own path to inspect/reuse the cache; then you own
        cleanup. Optional.

    Returns:
    tuple: (results, thresholds).
        results (dict): threshold_level -> {lead_day: [n_precip_objects,
            n_wind_objects, n_compound_objects]}.
        thresholds (dict): threshold_level -> (precip_threshold,
            wind_threshold), the values exceedance was measured against.
    """
    import shutil
    import tempfile
    from concurrent.futures import ProcessPoolExecutor

    if threshold_levels is None:
        threshold_levels = [5]

    own_cache_dir = cache_dir is None
    if own_cache_dir:
        cache_dir = tempfile.mkdtemp(prefix='spinup_cache_')

    try:
        wet_precip_sample, wind_sample, ok_dates = [], [], []
        tasks1 = [(date, cache_dir) for date in common_dates]
        _n_workers_auto = n_workers is None
        if _n_workers_auto:
            n_workers = resolve_max_workers(n_tasks=len(common_dates))
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_spinup_worker_init,
                                  initargs=(path, lsm_ds)) as ex:
            for date, out, err in tqdm(ex.map(_spinup_load_and_cache_date, tasks1),
                                        total=len(tasks1), desc='Pass 1: loading + caching'):
                if err is not None:
                    tqdm.write(f'Pass 1 skipping {date}: {err}')
                    continue
                p_lake, w_lake = out
                wet_precip_sample.append(p_lake)
                wind_sample.append(w_lake)
                ok_dates.append(date)

        all_precip_wet = np.concatenate(wet_precip_sample)
        all_wind_flat  = np.concatenate(wind_sample)
        thresholds = {}
        for pct in threshold_levels:
            thresholds[pct] = (
                np.nanpercentile(all_precip_wet, 100 - pct),
                np.nanpercentile(all_wind_flat,  100 - pct),
            )
            log.info(f"Top {pct:4.1f}%  →  precip > {thresholds[pct][0]:.3f},  wind > {thresholds[pct][1]:.3f}")

        results = {pct: {} for pct in threshold_levels}
        tasks2 = [(date, cache_dir, thresholds) for date in ok_dates]
        if _n_workers_auto:
            n_workers = resolve_max_workers(n_tasks=len(ok_dates))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for date, local_results, err in tqdm(ex.map(_spinup_count_from_cache, tasks2),
                                                  total=len(tasks2),
                                                  desc='Pass 2: counting storm objects (from cache)'):
                if err is not None:
                    tqdm.write(f'Pass 2 skipping {date}: {err}')
                    continue
                for pct, leadmap in local_results.items():
                    for lead_day, counts in leadmap.items():
                        if lead_day not in results[pct]:
                            results[pct][lead_day] = [0, 0, 0]
                        for k in range(3):
                            results[pct][lead_day][k] += counts[k]

        return results, thresholds
    finally:
        if own_cache_dir:
            shutil.rmtree(cache_dir, ignore_errors=True)


def select_days_7_to_9_hourly(ds):
    """Restrict an hourly (precipitation) dataset to forecast lead days 7-9.

    Uses elapsed_days in [5, 9) measured from this dataset's own first
    timestamp, intentionally one day earlier than select_days_7_to_9_wind's
    [6, 9), since the precipitation and wind NetCDFs have different native
    first-available timestamps within the same forecast run (hourly vs.
    3-hourly source resolution). load_forecast_data immediately reindexes the
    resulting precip series onto the wind timestamps
    (`ds_p_3h.reindex(time=wind_times, ...)`), so only the days-7-9 precip
    values are used regardless of this wider initial selection; the extra day
    only guarantees the reindex has data to pull from at wind's earliest
    days-7-9 timestamp.

    Parameters:
    ds (xarray.Dataset): Dataset with a 'time' coordinate.

    Returns:
    xarray.Dataset: Subset to the selected time window.
    """
    t0 = ds.time.values[0]
    elapsed_days = (ds.time.values - t0) / np.timedelta64(1, 'D')
    return ds.isel(time=(elapsed_days >= 5) & (elapsed_days < 9))

def select_days_7_to_9_wind(ds):
    """Restrict a wind (3-hourly) dataset to forecast lead days 7-9.

    Uses elapsed_days in [6, 9) measured from this dataset's own first
    timestamp; see select_days_7_to_9_hourly for why this differs by one day
    from that function's [5, 9) window.

    Parameters:
    ds (xarray.Dataset): Dataset with a 'time' coordinate.

    Returns:
    xarray.Dataset: Subset to the selected time window.
    """
    t0 = ds.time.values[0]
    elapsed_days = (ds.time.values - t0) / np.timedelta64(1, 'D')
    return ds.isel(time=(elapsed_days >= 6) & (elapsed_days < 9))

def _hansen_skewt_pdf(z, eta, lam):
    """Hansen (1994) skewed-t density, standardised to mean 0 and variance 1.

    Same parametrisation arch's dist='skewt' fits (eta = shape/df,
    lam = skewness in (-1, 1)).

    Hansen, B. E. (1994). Autoregressive conditional density estimation.
    International Economic Review, 35(3), 705-730.

    Parameters:
    z (float): Standardised innovation value.
    eta (float): Shape (degrees-of-freedom) parameter.
    lam (float): Skewness parameter in (-1, 1).

    Returns:
    float: Density value at z.
    """
    c = np.exp(gammaln((eta + 1) / 2) - gammaln(eta / 2)) / np.sqrt(np.pi * (eta - 2))
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(max(1 + 3 * lam**2 - a**2, 1e-12))
    thresh = -a / b
    arg = (b * z + a) / (1 - lam) if z < thresh else (b * z + a) / (1 + lam)
    return b * c * (1 + arg**2 / (eta - 2)) ** (-(eta + 1) / 2)


_KAPPA_SKEWT_CACHE = {}

def _kappa_skewt(eta, lam):
    """GJR persistence multiplier on the gamma term under Hansen's skew-t innovations.

    kappa(eta, lam) = E[z^2 * I(z<0)]. The skew-t is asymmetric by
    construction, so the symmetric-case value of 0.5 (normal and Student-t) does not apply.
    The value is computed by numerical integration of the density in
    _hansen_skewt_pdf (Hansen, 1994), exact to quadrature tolerance, and
    cached by rounded (eta, lam) since the same fitted values recur across
    multistart perturbations.

    Hansen, B. E. (1994). Autoregressive conditional density estimation.
    International Economic Review, 35(3), 705-730.

    Parameters:
    eta (float): Shape (degrees-of-freedom) parameter.
    lam (float): Skewness parameter in (-1, 1).

    Returns:
    float: E[z^2 * I(z<0)] under the skew-t density, or np.nan if either
        input is non-finite.
    """
    if not (np.isfinite(eta) and np.isfinite(lam)):
        return np.nan
    key = (round(float(eta), 4), round(float(lam), 4))
    if key in _KAPPA_SKEWT_CACHE:
        return _KAPPA_SKEWT_CACHE[key]
    val, _ = quad(lambda z: z**2 * _hansen_skewt_pdf(z, eta, lam), -np.inf, 0.0, limit=200)
    _KAPPA_SKEWT_CACHE[key] = val
    return val


def compute_persistence(params, vol_type, dist):
    """Volatility persistence for a fitted GARCH-family model.

        GARCH symmetric:  alpha[1] + beta[1]
        GJR, normal:      alpha[1] + beta[1] + 0.5*gamma[1]
        GJR, t:           alpha[1] + beta[1] + 0.5*gamma[1]                  (exact, by symmetry)
        GJR, skewt:       alpha[1] + beta[1] + kappa_skewt(eta,lam)*gamma[1] (numerically integrated)
        EGARCH(1,1,1):    abs(beta[1])

    For GJR the second-moment (covariance-stationarity) condition is
    alpha + beta + gamma * kappa < 1 with kappa = E[z^2 * I(z<0)] for
    unit-variance innovations z (Ling & McAleer, 2002, Example 2.1, p. 112,
    condition 2.1, who note kappa = 1/2 for symmetric innovations). That value is exact for the normal
    and Student-t cases (z^2 f(z) is even), and is obtained by numerical
    integration for the skew-t (see _kappa_skewt). For EGARCH, ln(sigma^2)
    follows an AR(1)-type recursion with coefficient beta[1]; it is strictly
    stationary and ergodic iff abs(beta[1]) < 1 (Nelson, 1991, Theorem 2.1
    and the ARMA discussion on p. 352), and Nelson measures the persistence
    of shocks by that AR root (p. 360). The persistence is therefore
    abs(beta[1]). This concerns ln(sigma^2); for fat-tailed innovations the
    unconditional moments of sigma^2 itself may not exist (Nelson, 1991,
    Appendix I). For higher-order
    specifications all alpha[.] and beta[.] terms are summed.

    Ling, S., & McAleer, M. (2002). Stationarity and the existence of moments
    of a family of GARCH processes. Journal of Econometrics, 106(1), 109-117.

    Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A
    new approach. Econometrica, 59(2), 347-370.

    Parameters:
    params (pandas.Series): Fitted model parameters, indexed by name (e.g.
        'alpha[1]', 'beta[1]', 'gamma[1]', 'nu', 'eta', 'lambda').
    vol_type (str): 'EGARCH' or a symmetric/GJR GARCH specification.
    dist (str): 'normal', 't', or 'skewt'.

    Returns:
    float or None: The persistence value, or None (not NaN) if a required
        parameter is missing or dist is unrecognised. This lets the caller
        treat "could not compute" and "computed, non-finite" as the same
        reject condition.
    """
    try:
        if vol_type == 'EGARCH':
            return abs(params['beta[1]'])
        # Sum ALL alpha/beta terms, not just [1] -- the menu includes
        # GARCH(1,2) and GARCH(2,1), which have beta[2] or alpha[2] too.
        # A formula using only alpha[1]+beta[1] would silently undercount
        # persistence for those two candidates and let a non-stationary
        # fit slip through.
        alpha = sum(v for k, v in params.items() if k.startswith('alpha['))
        beta = sum(v for k, v in params.items() if k.startswith('beta['))
        if 'gamma[1]' not in params.index:
            return alpha + beta
        gamma = params['gamma[1]']
        if dist == 'normal':
            kappa = 0.5
        elif dist == 't':
            # Student-t is symmetric about zero, so E[z^2 * I(z<0)] = 0.5 * E[z^2] exactly
            # (z^2 f(z) is even), independent of the degrees of freedom.
            kappa = 0.5
        elif dist == 'skewt':
            kappa = _kappa_skewt(params.get('eta', np.nan), params.get('lambda', np.nan))
        else:
            return None
        return alpha + beta + kappa * gamma
    except (KeyError, TypeError):
        return None


def _perturbed_garch_start(model, rng_local, scale=0.3):
    """Perturb a baseline GARCH fit's parameters for multi-start stability checking.

    Variance-equation parameters (omega, alpha, beta, gamma) get a simple
    multiplicative jitter. Distributional parameters near a hard boundary
    -- nu/eta (Student-t / skew-t tail shape, must stay >2) and lambda
    (skew-t skew, must stay in (-1,1)) -- get boundary-aware handling,
    since naive multiplicative jitter on a low nu/eta (expected for
    heavy-tailed series) easily pushes it past the constraint on most
    draws.

    Parameters:
    model (arch.univariate.base.ARCHModel): The model to fit a baseline
        for, then perturb.
    rng_local (np.random.Generator): RNG for the jitter draws.
    scale (float): Multiplicative jitter scale for variance-equation
        parameters and nu/eta. Default 0.3.

    Returns:
    np.ndarray: Perturbed starting values, same order as the model's
        baseline fitted parameters.
    """
    base = model.fit(disp='off', show_warning=False)
    names = base.params.index
    sv = base.params.values.copy()
    for i, name in enumerate(names):
        if name in ('nu', 'eta'):
            jittered = sv[i] * (1 + rng_local.uniform(-scale, scale))
            sv[i] = max(jittered, 2.5)
        elif name == 'lambda':
            jittered = sv[i] + rng_local.uniform(-0.15, 0.15)
            sv[i] = float(np.clip(jittered, -0.95, 0.95))
        else:
            sv[i] = sv[i] * (1 + rng_local.uniform(-scale, scale))
    return sv

def _condense_fail_reasons(fail_reasons):
    """Group per-start failure reasons into categories by stripping out numeric values.

    Numeric parameter values in a failure message vary start-to-start
    even when the underlying failure mode is identical (e.g. 20 starts
    all rejected as 'boundary-adjacent' but each with slightly different
    alpha/gamma).

    Parameters:
    fail_reasons (list): Raw per-start failure message strings.

    Returns:
    dict: category -> count, in first-seen order.
    """
    import re
    categories = {}
    for r in fail_reasons:
        body = r.split(': ', 1)[1] if ': ' in r else r
        category = re.sub(r'[-+]?\d*\.?\d+', '#', body)
        categories[category] = categories.get(category, 0) + 1
    return categories

def _garch_one_start(i, resid, p, o, q, vol, dist, seed_i):
    """Fit one perturbed-start GARCH candidate and apply the per-start sanity checks.

    Top-level (picklable) function for ProcessPoolExecutor, so it must not be
    nested or a lambda. Each start is an independent fit. Start i=0 uses the
    default initialisation of arch (no perturbation, no RNG dependence); a
    start i>0 perturbs around a freshly fitted baseline using its own seed_i
    (_perturbed_garch_start does its own internal fit, so keeping a shared RNG
    stream across starts would force that step to stay serial).

    Parameters:
    i (int): Start index. i=0 uses starting_values=None; i>0 uses perturbed
        starting values.
    resid (pandas.Series or np.ndarray): The residual series to fit.
    p (int): GARCH order (ARCH terms).
    o (int): Asymmetry order (0 symmetric, 1 GJR).
    q (int): GARCH order (variance terms).
    vol (str): Volatility specification, e.g. 'GARCH' or 'EGARCH'.
    dist (str): Innovation distribution: 'normal', 't', or 'skewt'.
    seed_i (int): This start's own RNG seed for the jitter draws (unused if i == 0).

    Returns:
    tuple: (i, res_or_None, fail_reason_or_None). res is an arch fit result with
        ._persistence and ._near_boundary attached if it passed every sanity
        check, otherwise None with a fail_reason string.
    """
    import warnings
    from arch import arch_model

    rng_local = np.random.default_rng(seed_i)
    try:
        g = arch_model(resid, mean='Zero', vol=vol, p=p, o=o, q=q,
                        dist=dist, rescale=True)
        # arch re-registers its own filter for ConvergenceWarning at
        # warn-time (to defeat exactly the kind of blanket
        # warnings.filterwarnings('ignore') a notebook sets up front),
        # so it has to be suppressed here, scoped to this call, not at
        # the notebook level. Safe to drop entirely: convergence_flag
        # is already captured below and reported in aggregate.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            res = g.fit(disp='off', update_freq=0,
                         options={'maxiter': 1000},
                         starting_values=None if i == 0 else _perturbed_garch_start(g, rng_local))
        params = res.params
        if vol == 'EGARCH':
            sane = np.isfinite(res.loglikelihood) and np.isfinite(params.get('alpha[1]', np.nan))
            fail_why = 'EGARCH loglik/alpha not finite'
        else:
            # Check EVERY alpha[i]/beta[i] term, not just index 1 --
            # GARCH(1,2) has beta[1] AND beta[2]; GARCH(2,1) has
            # alpha[1] AND alpha[2]. A check that only inspected index
            # 1 could pass a fit with a negative or non-finite
            # second-order coefficient as "sane" even though
            # compute_persistence's aggregate sum (which does account
            # for every order) would be built from an invalid
            # individual term -- variance-equation coefficients must
            # each be non-negative for the conditional variance to be
            # guaranteed positive pointwise, not just in aggregate.
            alpha_terms = {k: v for k, v in params.items() if k.startswith('alpha[')}
            beta_terms  = {k: v for k, v in params.items() if k.startswith('beta[')}
            sane = (
                np.isfinite(res.loglikelihood)
                and params.get('omega', np.nan) > 0
                and len(alpha_terms) > 0
                and all(np.isfinite(v) and v >= 0 for v in alpha_terms.values())
                and len(beta_terms) > 0
                and all(np.isfinite(v) and v >= 0 for v in beta_terms.values())
            )
            fail_why = (f"omega={params.get('omega', 'NA')}, "
                        f"alpha_terms={dict(alpha_terms)}, beta_terms={dict(beta_terms)}")
        if dist == 't' and sane:
            nu = params.get('nu', np.nan)
            if not (np.isfinite(nu) and nu > 2.01):
                sane = False
                fail_why = f"nu={nu} (must be >2.01)"
        elif dist == 'skewt' and sane:
            eta = params.get('eta', np.nan)
            lam = params.get('lambda', np.nan)
            if not (np.isfinite(eta) and eta > 2.01):
                sane = False
                fail_why = f"eta={eta} (must be >2.01)"
            elif not (np.isfinite(lam) and -0.999 < lam < 0.999):
                sane = False
                fail_why = f"lambda={lam} (must be in (-1,1))"
        # Boundary check for GJR-GARCH's alpha and asymmetry term: arch box-constrains
        # alpha to [0,1] and gamma similarly. Rather than only flagging
        # a fit that lands within 0.001 of the wall, use the parameter's
        # own standard error: if the 95% CI around alpha contains 1.0
        # (or around gamma contains -1.0), the estimate is statistically
        # indistinguishable from a boundary solution even if the point
        # estimate itself isn't exactly at the wall -- a "significant"
        # t-stat in this situation doesn't mean a genuine interior
        # optimum, since standard MLE asymptotics assume one.
        # EGARCH is excluded: its alpha and gamma act on the log variance, so
        # alpha >= 1 or gamma <= -1 is not a constraint boundary there.
        if o > 0 and vol != 'EGARCH' and sane:
            alpha_v = params.get('alpha[1]', np.nan)
            gamma_v = params.get('gamma[1]', np.nan)
            alpha_tv = res.tvalues.get('alpha[1]', np.nan)
            gamma_tv = res.tvalues.get('gamma[1]', np.nan)
            alpha_se = abs(alpha_v / alpha_tv) if (np.isfinite(alpha_tv) and alpha_tv != 0) else np.inf
            gamma_se = abs(gamma_v / gamma_tv) if (np.isfinite(gamma_tv) and gamma_tv != 0) else np.inf
            alpha_ci_hi = alpha_v + 1.96 * alpha_se
            gamma_ci_lo = gamma_v - 1.96 * gamma_se
            if (np.isfinite(alpha_v) and alpha_ci_hi >= 1.0) or \
               (np.isfinite(gamma_v) and gamma_ci_lo <= -1.0):
                sane = False
                fail_why = (f"boundary-adjacent: alpha={alpha_v:.4f} (95% CI upper "
                            f"≈{alpha_ci_hi:.3f}), gamma={gamma_v:.4f} (95% CI lower "
                            f"≈{gamma_ci_lo:.3f}) — confidence interval contains the "
                            f"constraint boundary, not a genuine interior optimum")
        # Degenerate variance-path check: parameter-level sanity above
        # doesn't catch a conditional_volatility series that collapses
        # to near-zero at isolated timesteps even when every individual
        # parameter is in-bounds. Dividing residuals by a near-zero
        # volatility produces extreme standardised residuals that can
        # dominate downstream Ljung-Box tests and PIT ranks without
        # tripping any check above.
        if sane:
            cv = res.conditional_volatility
            cv_min, cv_median = np.min(cv), np.median(cv)
            if cv_min < 0.05 * cv_median:
                sane = False
                fail_why = (f"conditional_volatility collapses "
                            f"(min={cv_min:.4g}, median={cv_median:.4g}, "
                            f"ratio={cv_min / cv_median:.4f} < 0.05)")
        # Stationarity check. A
        # persistence >= 1.0 is rejected at the same tier as omega<=0
        # etc -- it isn't a genuine interior optimum for a covariance-
        # stationary process. 0.98 is an arbitrary flag threshold (no
        # literature standard) kept SEPARATE from the reject decision:
        # it marks the fit as near_boundary without discarding it.
        persistence, near_boundary = None, False
        if sane:
            persistence = compute_persistence(params, vol_type=vol, dist=dist)
            if persistence is None or not np.isfinite(persistence):
                sane = False
                fail_why = f"persistence could not be computed from params={dict(params)}"
            elif persistence >= 1.0:
                sane = False
                fail_why = f"persistence={persistence:.4f} >= 1.0 (non-stationary)"
            elif persistence >= 0.98:
                near_boundary = True
        if sane:
            res._persistence = persistence
            res._near_boundary = near_boundary
            return i, res, None
        else:
            return i, None, f"start {i}: rejected by sanity check ({fail_why})"
    except Exception as e:
        return i, None, f"start {i}: exception during fit — {type(e).__name__}: {e}"


def fit_garch_multistart(resid, p=1, o=0, q=1, vol='GARCH', dist='normal',
                          n_starts=8, seed=0, var_name=None, max_workers=None):
    """Refit a GARCH-family spec from several starting points and select the most stable fit.

    The n_starts restarts run in parallel via ProcessPoolExecutor (each start
    is a fully independent fit; see _garch_one_start for the per-start logic).

    Discards fits that did not converge or produced parameter-insane output
    (omega<=0, alpha/beta outside [0,1], nu/eta<=2, |lambda|>=1), degenerate
    conditional variance paths (collapsing near zero), and drops any remaining
    fit whose log-likelihood is implausibly far from the median (catches single
    diverged optimizer runs before they poison the stability metric). Also
    rejects persistence>=1.0 (the covariance-stationarity requirement) and
    flags 0.98<=persistence<1.0 as near_boundary without rejecting; 0.98 is a
    chosen threshold, not a literature standard.

    Parameters:
    resid (pandas.Series or np.ndarray): The residual series to fit.
    p (int): GARCH order (ARCH terms). Default 1.
    o (int): Asymmetry order (0 for symmetric GARCH, 1 for GJR-GARCH). Default 0.
    q (int): GARCH order (variance terms). Default 1.
    vol (str): Volatility specification, e.g. 'GARCH' or 'EGARCH'. Default 'GARCH'.
    dist (str): Innovation distribution: 'normal', 't', or 'skewt'. Default 'normal'.
    n_starts (int): Number of restarts. Default 8.
    seed (int): Base RNG seed; start i>0 uses its own seed derived from this
        (seed * 10_000 + i). Default 0.
    var_name (str): Used only in the error message. Optional.
    max_workers (int): Concurrent worker processes. Default None: resolved by
        resolve_max_workers(n_tasks=n_starts).

    Returns:
    tuple: (best_fit, plausible_loglikelihoods, spread, n_dropped, n_total,
        persistence, near_boundary), where the last two describe best_fit.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    fits, fail_reasons = [], []
    if max_workers is None:
        max_workers = resolve_max_workers(n_tasks=n_starts)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_garch_one_start, i, resid, p, o, q, vol, dist, seed * 10_000 + i)
            for i in range(n_starts)
        ]
        for future in as_completed(futures):
            i, res, fail_reason = future.result()
            if res is not None:
                fits.append(res)
            else:
                fail_reasons.append(fail_reason)

    if not fits:
        categories = _condense_fail_reasons(fail_reasons)
        tag = f"{var_name}, " if var_name else ""
        err = RuntimeError(
            f"[{tag}{p},{o},{q},vol={vol},dist={dist}] all {n_starts} starts failed: "
            + "; ".join(f"{count}/{n_starts} {cat}" for cat, count in categories.items())
        )
        err.categories = categories
        raise err

    lls = [f.loglikelihood for f in fits]
    med = np.median(lls)
    plausible = [f for f in fits if abs(f.loglikelihood - med) < _LOGLIK_OUTLIER_TOL]
    n_dropped = len(fits) - len(plausible)
    if not plausible:
        plausible = fits
    best = max(plausible, key=lambda f: f.loglikelihood)
    plausible_lls = [f.loglikelihood for f in plausible]
    spread = (max(plausible_lls) - min(plausible_lls)) if len(plausible_lls) > 1 else 0.0
    return best, plausible_lls, spread, n_dropped, len(fits), best._persistence, best._near_boundary


def select_best_volatility_model(resid, var_name, n_starts=20, seed=1, lb_lags=(8, 16, 24)):
    """Compare the standard GARCH-family menu with multi-start stability checking and select by AIC.

    Menu: GARCH(1,1), GARCH(1,2), GARCH(2,1) (Bollerslev, 1986), GJR(1,1,1)
    (Glosten, Jagannathan, & Runkle, 1993), EGARCH(1,1,1) (Nelson, 1991),
    plus Student-t and skewed-t variants of the symmetric and GJR specs.

    Selection rule (AIC-driven, not a Ljung-Box pass/fail):

    1. LB(z)/LB(z^2) at lags 8/16/24 are computed and reported for every
       candidate but play no role in selection.
    2. Eligibility pool = sane candidates (fit_garch_multistart's per-start
       checks, which already reject persistence>=1.0) and, for asymmetric
       models only, gamma individually significant (|t|>1.96). The gamma gate
       is an added restriction, not implied by AIC theory: it stops an
       unidentified asymmetry term from winning on AIC noise.
    3. If the gamma gate eliminates every asymmetric candidate but at least
       one symmetric candidate is still eligible, fall back to the
       symmetric-only pool and say so in the verdict.
    4. Within the eligibility pool, lowest AIC wins.
    5. If the eligibility pool ends up empty, returns chosen=None. The
       day/night diagnostic and the AR-only fallback are run by
       select_volatility_for_variable, not here.

    Bollerslev, T. (1986). Generalized autoregressive conditional
    heteroskedasticity. Journal of Econometrics, 31(3), 307-327.

    Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation
    between the expected value and the volatility of the nominal excess return
    on stocks. The Journal of Finance, 48(5), 1779-1801.

    Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A
    new approach. Econometrica, 59(2), 347-370.

    Parameters:
    resid (pandas.Series or np.ndarray): The residual series to fit.
    var_name (str): Used in log messages.
    n_starts (int): Multi-start restarts per candidate, passed to
        fit_garch_multistart. Default 20.
    seed (int): RNG seed. Default 1.
    lb_lags (tuple): Ljung-Box lags to report. Default (8, 16, 24).

    Returns:
    tuple: (chosen_label_or_None, fit_or_None, candidates_dict, verdict_str,
        skip_reasons).
    """
    model_menu = [
        ('GARCH(1,1)',          dict(p=1, o=0, q=1), False, 'normal'),
        ('GARCH(1,2)',          dict(p=1, o=0, q=2), False, 'normal'),
        ('GARCH(2,1)',          dict(p=2, o=0, q=1), False, 'normal'),
        ('GJR(1,1,1)',          dict(p=1, o=1, q=1), True,  'normal'),
        ('EGARCH(1,1,1)',       dict(p=1, o=1, q=1, vol='EGARCH'), True, 'normal'),
        ('GARCH(1,1)-t',        dict(p=1, o=0, q=1), False, 't'),
        ('GARCH(1,2)-t',        dict(p=1, o=0, q=2), False, 't'),
        ('GARCH(2,1)-t',        dict(p=2, o=0, q=1), False, 't'),
        ('GJR(1,1,1)-t',        dict(p=1, o=1, q=1), True,  't'),
        ('GJR(1,1,1)-skewt',    dict(p=1, o=1, q=1), True,  'skewt'),
        ('EGARCH(1,1,1)-t',     dict(p=1, o=1, q=1, vol='EGARCH'), True, 't'),
        ('EGARCH(1,1,1)-skewt', dict(p=1, o=1, q=1, vol='EGARCH'), True, 'skewt'),
    ]
    candidates = {}
    skipped_models = []
    skip_reasons = {}
    for label, kwargs, is_asym, dist in model_menu:
        kwargs = dict(kwargs)
        vol_type = kwargs.pop('vol', 'GARCH')
        try:
            best, lls, spread, n_dropped, n_total, persistence, near_boundary = fit_garch_multistart(
                resid, n_starts=n_starts, seed=seed, vol=vol_type, dist=dist, var_name=var_name, **kwargs)
        except RuntimeError as e:
            skipped_models.append(label)
            skip_reasons[label] = str(e)
            continue
        if n_dropped > 0:
            log.info(f"  [{var_name}] {label}: dropped {n_dropped}/{n_total} starts "
                  f"as degenerate before computing stability.")
        z = resid / best.conditional_volatility
        lb_z  = acorr_ljungbox(z,    lags=list(lb_lags), return_df=True)['lb_pvalue']
        lb_z2 = acorr_ljungbox(z**2, lags=list(lb_lags), return_df=True)['lb_pvalue']
        gamma_t = best.tvalues['gamma[1]'] if 'gamma[1]' in best.params.index else None
        gamma_significant = (abs(gamma_t) > 1.96) if gamma_t is not None else None
        alpha_gamma_mirror = None
        if gamma_t is not None and vol_type == 'GARCH':
            alpha = best.params['alpha[1]']
            gamma = best.params['gamma[1]']
            alpha_gamma_mirror = abs(alpha + gamma)
        nu = best.params.get('nu', None) if dist == 't' else (best.params.get('eta', None) if dist == 'skewt' else None)
        nu_t = best.tvalues.get('nu', None) if dist == 't' else (best.tvalues.get('eta', None) if dist == 'skewt' else None)
        lam = best.params.get('lambda', None)
        lam_t = best.tvalues.get('lambda', None)
        candidates[label] = dict(
            fit=best, AIC=best.aic, loglik_spread_across_starts=spread,
            n_dropped=n_dropped, n_total=n_total, is_asymmetric=is_asym, dist=dist,
            LB_z_p8=float(lb_z.iloc[0]), LB_z_p16=float(lb_z.iloc[1]), LB_z_p24=float(lb_z.iloc[2]),
            LB_z2_p8=float(lb_z2.iloc[0]), LB_z2_p16=float(lb_z2.iloc[1]), LB_z2_p24=float(lb_z2.iloc[2]),
            gamma_t=gamma_t, gamma_significant=gamma_significant,
            alpha_plus_gamma=alpha_gamma_mirror,
            nu=nu, nu_t=nu_t, lam=lam, lam_t=lam_t,
            persistence=persistence, near_boundary=near_boundary,
        )

    if not candidates:
        raise RuntimeError(
            f"[{var_name}] Every model in the menu failed to estimate "
            f"(skipped: {skipped_models}). This is a genuine result worth "
            f"reporting, not a bug — but it needs a manual decision (e.g. "
            f"widen the menu, or report that no GARCH-family model fits "
            f"this series for this variable).")

    # LB(z)/LB(z²) are already stored per candidate for reporting; they play
    # no role in the pool below (AIC selection, LB reported but not used as a
    # gate: post-GARCH Ljung-Box is a difficult test to use as a hard gate).

    # Eligibility: sane candidates already have persistence<1.0 by
    # construction (fit_garch_multistart rejects >=1.0 at the per-start
    # sanity stage) -- re-checking it here is redundant given the current
    # implementation, but kept explicit in case that upstream guarantee
    # ever changes.
    eligible = [k for k, c in candidates.items()
                if c['persistence'] is not None and c['persistence'] < 1.0
                and (not c['is_asymmetric'] or c['gamma_significant'])]

    sig_note = ""
    if not eligible:
        symmetric_only = [k for k, c in candidates.items()
                           if c['persistence'] is not None and c['persistence'] < 1.0
                           and not c['is_asymmetric']]
        if symmetric_only:
            eligible = symmetric_only
            sig_note = (" (NOTE: gamma gate eliminated every asymmetric candidate; "
                         "proceeding symmetric-only.)")

    if not eligible:
        verdict = (f"[{var_name}] Eligibility pool empty -- no candidate is both "
                   f"stationary (persistence<1.0) and, if asymmetric, gamma-significant. "
                   f"Caller must run the day/night diagnostic + AR-only fallback "
                   f"(see select_volatility_for_variable).")
        log.info(verdict)
        return None, None, candidates, verdict, skip_reasons

    chosen = min(eligible, key=lambda k: candidates[k]['AIC'])
    c = candidates[chosen]
    nb_str = " [near_boundary]" if c['near_boundary'] else ""
    sig_str = f", gamma t={c['gamma_t']:.2f} (significant)" if c['is_asymmetric'] else ""
    lb_str = (f"LB(z) p@8/16/24={c['LB_z_p8']:.3f}/{c['LB_z_p16']:.3f}/{c['LB_z_p24']:.3f}, "
              f"LB(z²) p@8/16/24={c['LB_z2_p8']:.3f}/{c['LB_z2_p16']:.3f}/{c['LB_z2_p24']:.3f} "
              f"(informational only)")
    verdict = (f"[{var_name}] {chosen} selected: AIC={c['AIC']:.2f}, "
               f"persistence={c['persistence']:.4f}{nb_str}{sig_str}. {lb_str}.{sig_note}")
    log.info(verdict)
    return chosen, c['fit'], candidates, verdict, skip_reasons


def day_night_diagnostic(ar_resid, hours, var_name, n_starts=20, seed=1,
                          lb_lags=(8, 16, 24), night_hours=NIGHT_HOURS_UTC, day_hours=None,
                          spread_flag_threshold=50.0):
    """Refit the GARCH menu independently on day and night hour-of-day subsets as a robustness check.

    Reuses the pooled AR residuals (AR is not refitted per regime; that fit
    happened once upstream in the sparse-AR step) and refits the full GARCH
    menu independently on the day and night subsets via
    select_best_volatility_model.

    hours must be hour-of-day values aligned index-for-index with ar_resid.
    The default night_hours is gate_config.NIGHT_HOURS_UTC, the night mask of
    notebook 00; day_hours defaults to its complement (gate_config.DAY_HOURS_UTC
    for the 3-hourly stamps).

    The night subset is smaller than the full sample. If its eligibility pool
    is empty, or if a majority of its fitted candidates show high
    loglik_spread_across_starts, the subset's result is marked inconclusive
    rather than treated as evidence.

    Interpretive limit: because this reuses pooled-AR residuals rather than
    refitting AR per regime, a persistence drop under the split shows that
    GARCH-stage refitting on regime-subsetted residuals resolves the
    stationarity problem. It does not by itself establish regime-dependent
    volatility, since some of the effect could originate from mean-equation
    structure the pooled AR model left in the residuals. Lamoureux and
    Lastrapes (1990) motivate the diagnostic by analogy: they show that
    ignoring deterministic shifts in the unconditional variance (the GARCH
    constant) biases estimated GARCH persistence toward unity, in daily stock
    returns and in a Monte Carlo experiment. Their shifts are regimes in time
    rather than a day/night split, so this is motivation for running the
    check, not proof of the specific causal claim above.

    Lamoureux, C. G., & Lastrapes, W. D. (1990). Persistence in variance,
    structural change, and the GARCH model. Journal of Business & Economic
    Statistics, 8(2), 225-234.

    Parameters:
    ar_resid (pandas.Series or np.ndarray): Pooled AR residuals.
    hours (np.ndarray): Hour-of-day values, aligned index-for-index with ar_resid.
    var_name (str): Used in log messages.
    n_starts (int): Passed to select_best_volatility_model. Default 20.
    seed (int): RNG seed. Default 1.
    lb_lags (tuple): Ljung-Box lags to report. Default (8, 16, 24).
    night_hours (tuple): Hours defining the night subset. Default gate_config.NIGHT_HOURS_UTC.
    day_hours (tuple): Hours defining the day subset. Defaults to the
        complement of night_hours if None. Optional.
    spread_flag_threshold (float): loglik_spread_across_starts threshold above
        which a candidate counts toward "high spread" when deciding
        inconclusiveness. Default 50.0.

    Returns:
    dict: Per-subset persistence, chosen model, inconclusive flag and reason,
        plus 'either_subset_below_0.98'.
    """
    hours_arr = np.asarray(hours)
    resid_arr = np.asarray(ar_resid)
    if len(hours_arr) != len(resid_arr):
        raise ValueError(f"[{var_name}] hours length {len(hours_arr)} != "
                          f"ar_resid length {len(resid_arr)}")
    night_mask = np.isin(hours_arr, night_hours)
    day_mask = ~night_mask if day_hours is None else np.isin(hours_arr, day_hours)

    report = {'var_name': var_name, 'night_n': int(night_mask.sum()), 'day_n': int(day_mask.sum())}

    for subset_name, mask in (('night', night_mask), ('day', day_mask)):
        resid_sub = resid_arr[mask]
        try:
            chosen, fit, candidates, verdict, skip_reasons = select_best_volatility_model(
                resid_sub, f"{var_name}-{subset_name}", n_starts=n_starts,
                seed=seed, lb_lags=lb_lags)
        except RuntimeError as e:
            report[f'{subset_name}_persistence'] = None
            report[f'{subset_name}_chosen'] = None
            report[f'{subset_name}_inconclusive'] = True
            report[f'{subset_name}_reason'] = f"every model failed to estimate: {e}"
            continue

        n_fitted = len(candidates)
        n_high_spread = sum(1 for c in candidates.values()
                             if c['loglik_spread_across_starts'] > spread_flag_threshold)
        inconclusive = (chosen is None) or (n_fitted > 0 and n_high_spread / n_fitted > 0.5)

        report[f'{subset_name}_persistence'] = candidates[chosen]['persistence'] if chosen else None
        report[f'{subset_name}_chosen'] = chosen
        report[f'{subset_name}_inconclusive'] = bool(inconclusive)
        if inconclusive and subset_name == 'night':
            report[f'{subset_name}_reason'] = (
                f"night subset n={report['night_n']}: eligibility pool empty or "
                f"majority of fitted candidates unstable -- treat as inconclusive, "
                f"not as evidence either way.")

    below_98 = any(
        report.get(f'{s}_persistence') is not None
        and report[f'{s}_persistence'] < 0.98
        and not report.get(f'{s}_inconclusive', False)
        for s in ('day', 'night'))
    report['either_subset_below_0.98'] = below_98

    log.info(f"[{var_name}] day/night diagnostic -- "
          f"day: persistence={report.get('day_persistence')}, n={report['day_n']}, "
          f"inconclusive={report.get('day_inconclusive')} | "
          f"night: persistence={report.get('night_persistence')}, n={report['night_n']}, "
          f"inconclusive={report.get('night_inconclusive')} | "
          f"either subset < 0.98: {below_98}. "
          f"Does NOT by itself establish regime-dependent volatility -- see "
          f"interpretive limit in this function's docstring / thesis text.")
    return report


def select_volatility_for_variable(ar_resid, var_name, hours, n_starts=20, seed=1,
                                    lb_lags=(8, 16, 24)):
    """Select a volatility model for one variable, falling back to a day/night diagnostic if needed.

    Runs the pooled GARCH-menu selection; if the eligibility pool comes back
    empty, runs the day/night diagnostic (report only) and falls back to the
    raw AR residuals as that variable's pseudo-observations.

    The fallback also triggers if every candidate fails to estimate at all
    (select_best_volatility_model raising RuntimeError), not only in the
    narrower case where candidates fit but none is eligible. Crashing the
    whole per-variable loop on a total fit failure would be worse than falling
    back in the same way for both cases.

    Parameters:
    ar_resid (pandas.Series or np.ndarray): Pooled AR residuals.
    var_name (str): Used in log messages.
    hours (np.ndarray): Hour-of-day values, aligned index-for-index with
        ar_resid; passed to day_night_diagnostic if needed.
    n_starts (int): Passed to select_best_volatility_model. Default 20.
    seed (int): RNG seed. Default 1.
    lb_lags (tuple): Ljung-Box lags to report. Default (8, 16, 24).

    Returns:
    dict: Keys var_name, chosen_label, fit, candidates, verdict,
        final_residuals, fallback_triggered, diagnostic, skip_reasons.
    """
    try:
        chosen, fit, candidates, verdict, skip_reasons = select_best_volatility_model(
            ar_resid, var_name, n_starts=n_starts, seed=seed, lb_lags=lb_lags)
    except RuntimeError as e:
        chosen, fit, candidates, skip_reasons = None, None, {}, {}
        verdict = f"[{var_name}] Every candidate failed to estimate: {e}"
        log.info(verdict)

    if chosen is not None:
        # Keep whatever container type ar_resid came in as (typically a
        # pandas Series with a DatetimeIndex) -- downstream code aligns
        # variables on that index (common_idx etc). Forcing to a bare
        # ndarray here would silently break that alignment.
        final_residuals = (ar_resid * fit.scale) / fit.conditional_volatility
        return dict(var_name=var_name, chosen_label=chosen, fit=fit, candidates=candidates,
            verdict=verdict, final_residuals=final_residuals,
            fallback_triggered=False, diagnostic=None, skip_reasons=skip_reasons)

    diagnostic = day_night_diagnostic(ar_resid, hours, var_name, n_starts=n_starts,
                                       seed=seed, lb_lags=lb_lags)
    log.info(f"[{var_name}] No eligible models. Fallback: AR-only residuals.")
    return dict(var_name=var_name, chosen_label=None, fit=None, candidates=candidates,
            verdict=verdict, final_residuals=ar_resid,
            fallback_triggered=True, diagnostic=diagnostic, skip_reasons=skip_reasons)
