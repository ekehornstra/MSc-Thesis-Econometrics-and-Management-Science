"""plotting_preprocessing.py

Figures of the data description and preprocessing chapters: marginal distributions, the diurnal cycle by season, the spin-up
check, and the AR/GARCH residual diagnostics (ACF/PACF, volatility before and after GARCH, deseasonalisation stages).

Figures are saved as PNG under plotting_core.IMAGES_DIR, which plotting_core.configure sets per notebook.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import plotting_core as _core
from plotting_core import C_PRECIP, C_WIND
from figure_replay import recorded


def _periodic_diurnal_curve(hours, hourly_mean, n_fine=200):
    """Cubic spline through 3-hourly means, periodic over the 24-hour day.

    Parameters:
    hours (np.ndarray): The sampled hours of day, equally spaced from 0 (for example 0, 3, ..., 21).
    hourly_mean (np.ndarray): Mean value at each hour in hours.
    n_fine (int): Number of points of the evaluation grid on [0, 24]. Default 200.

    Returns:
    tuple: (hours_fine, smooth), both of length n_fine; smooth(24) equals smooth(0).
    """
    period = 24.0
    x = np.append(hours, period)
    y = np.append(hourly_mean, hourly_mean[0])
    hours_fine = np.linspace(0, period, n_fine)
    return hours_fine, make_interp_spline(x, y, k=3, bc_type='periodic')(hours_fine)


def _small_markers(ax, size=1.6):
    """Draw the dots of an ACF or PACF stem plot at a fixed small size (in points on paper).

    The lines are flagged so that plotting_core.fit_to_print does not rescale or raise their marker size when the figure is fitted
    to the print width.

    Parameters:
    ax (matplotlib.axes.Axes): Axes of a plot_acf or plot_pacf call, changed in place.
    size (float): Marker size in points. Default 1.6.
    """
    for line in ax.lines:
        if line.get_marker() not in (None, 'None', '', ' '):
            line.set_markersize(size)
            line.set_gid('fixed_marker')


@recorded
def plot_marginals(lake_mean, time_pd, name, color, unit='', xmin=None, xmax=None, label=''):
    """Plot the marginal distribution, seasonal cycle and diurnal cycle of one lake-wide mean series.

    The series is the 3-hourly lake-wide mean used in the copula analysis, not raw grid-point samples. The hour of day is taken
    from time_pd.hour. The layout depends on name:
      'Precipitation': 2 x 2 panels, histogram, log-scale histogram, time series, diurnal cycle;
      'Divergence': 1 x 2 panels, time series and diurnal cycle, each with a zero reference line (no histogram);
      anything else (for example 'Wind Speed'): 1 x 3 panels, histogram, time series, diurnal cycle.
The diurnal curve is a cubic spline through the eight 3-hourly means, periodic over the day; between the markers it is
interpolation, not data.

    Parameters:
    lake_mean (array-like): Lake-wide mean series.
    time_pd (pd.DatetimeIndex): Time stamps of the series.
    name (str): Variable name, used in the titles and the file name.
    color (str): Plot colour.
    unit (str): Axis label of the values. Default ''.
    xmin (float or None): Lower end of the histogram range. Default None (the minimum).
    xmax (float or None): Upper end of the histogram range. Default None (the maximum).
    label (str): Suffix of the file name 'marginal_<name>[_<label>].png' under plotting_core.IMAGES_DIR. Default ''.
    """
    vals = np.asarray(lake_mean)
    is_divergence = (name == 'Divergence')
    is_precip     = (name == 'Precipitation')

    if is_divergence:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_ts, ax_dc = axes
        ax_hist = ax_hist_log = None
    elif is_precip:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_hist, ax_hist_log = axes[0]
        ax_ts, ax_dc = axes[1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_hist, ax_ts, ax_dc = axes
        ax_hist_log = None

    if ax_hist is not None:
        lo = xmin if xmin is not None else np.nanmin(vals)
        hi = xmax if xmax is not None else np.nanmax(vals)
        ax_hist.hist(vals, bins=100, color=color, range=(lo, hi), alpha=0.85)
        ax_hist.set_title(f'{name} — distribution (lake-wide mean)', fontsize=16)
        ax_hist.set_xlabel(unit, fontsize=16)
        ax_hist.set_ylabel('Count', fontsize=16)
        ax_hist.set_xlim(lo, hi)
        ax_hist.tick_params(labelsize=16, rotation=30)
        ax_hist.spines[['top', 'right']].set_visible(False)

    if ax_hist_log is not None:
        vals_pos = vals[vals > 0]
        ax_hist_log.hist(vals_pos, bins=np.logspace(np.log10(vals_pos.min()),
                                                     np.log10(vals_pos.max()), 100),
                         color=color, alpha=0.85)
        ax_hist_log.set_xscale('log')
        ax_hist_log.set_title(f'{name} — distribution, log scale (lake-wide mean)', fontsize=16)
        ax_hist_log.set_xlabel(unit, fontsize=16)
        ax_hist_log.set_ylabel('Count', fontsize=16)
        ax_hist_log.tick_params(labelsize=16, rotation=30)
        ax_hist_log.spines[['top', 'right']].set_visible(False)

    df = pd.Series(vals, index=time_pd).sort_index()
    df_daily = df.resample('D').mean()
    df_14d   = df_daily.rolling(14, center=True, min_periods=7).mean()
    ax_ts.plot(df_daily.index, df_daily.values, color=color, lw=0.6, alpha=0.4)
    ax_ts.plot(df_14d.index,   df_14d.values,   color=color, lw=2.0, label='14-day rolling mean')
    ax_ts.set_title(f'{name} — seasonal cycle (lake-wide mean)', fontsize=16)
    ax_ts.set_xlabel('Date', fontsize=16)
    ax_ts.set_ylabel(unit, fontsize=16)
    ax_ts.tick_params(labelsize=16, rotation=30)
    ax_ts.legend(fontsize=16)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.spines[['top', 'right']].set_visible(False)
    if is_divergence:
        ax_ts.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)

    hours_3h  = np.arange(0, 24, 3)
    hour_vals = time_pd.hour
    hourly_mean = np.array([np.nanmean(vals[hour_vals == h]) for h in hours_3h])
    hours_fine, smooth = _periodic_diurnal_curve(hours_3h, hourly_mean)
    ax_dc.plot(hours_3h, hourly_mean, color=color, lw=0, linestyle='none', marker='o', ms=6, alpha=0.7, zorder=5)
    ax_dc.plot(hours_fine, smooth, color=color, lw=2.5)
    ax_dc.set_title(f'{name} — diurnal cycle (annual mean)', fontsize=16)
    ax_dc.set_xlabel('Hour (UTC)', fontsize=16)
    ax_dc.set_ylabel(unit, fontsize=16)
    ax_dc.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax_dc.set_xticklabels(['00','03','06','09','12','15','18','21','24'], rotation=30)
    ax_dc.tick_params(labelsize=16)
    ax_dc.set_xlim(-0.5, 24.5)
    ax_dc.grid(True, alpha=0.3)
    ax_dc.spines[['top', 'right']].set_visible(False)
    if is_divergence:
        ax_dc.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(_core.IMAGES_DIR,
        f'marginal_{name.lower().replace(" ", "_")}{"_" + label if label else ""}.png'),
        dpi=150, bbox_inches='tight')
    plt.show()

@recorded
def plot_seasonal_diurnal(precip_flat, wind_flat, div_flat,
                          month_flat, hour_flat, label=''):
    """Diurnal cycle by meteorological season (DJF/MAM/JJA/SON), one panel per variable.

    The curves are cubic splines through the eight 3-hourly means, periodic over the day (24:00 equals 00:00); between the
    markers they are interpolation, not data. Precipitation and wind are clipped at zero.

    Parameters:
    precip_flat (np.ndarray): Grid-point-sampled precipitation values.
    wind_flat (np.ndarray): Grid-point-sampled wind speed values.
    div_flat (np.ndarray): Grid-point-sampled divergence values.
    month_flat (np.ndarray): Calendar month per sample, aligned index-for-index.
    hour_flat (np.ndarray): Hour-of-day per sample, aligned index-for-index.
    label (str): Suffix of the file name 'meandiurnalcycle[_<label>].png' under plotting_core.IMAGES_DIR. Default ''.
    """
    fig, axes   = plt.subplots(1, 3, figsize=(18, 5))
    hours       = np.arange(0, 24, 3)
    tick_hours  = [0, 6, 12, 18, 24]
    tick_labels = ['00:00', '06:00', '12:00', '18:00', '24:00']

    panels = [
        (axes[0], precip_flat, 'Precip (mm/3hr)',  True),
        (axes[1], wind_flat,   'Wind speed (m/s)', True),
        (axes[2], div_flat,    'Divergence (s\u207b\u00b9)', False),
    ]

    for ax, data, ylabel, clip_zero in panels:
        for season, months, color, ls, season_label in [
            ('DJF', [12, 1, 2],  '#0072B2', '-',  'Dec-Jan-Feb'),
            ('MAM', [3,  4, 5],  '#D55E00', '--', 'Mar-Apr-May'),
            ('JJA', [6,  7, 8],  '#009E73', '-.', 'Jun-Jul-Aug'),
            ('SON', [9, 10, 11], '#E69F00', ':',  'Sep-Oct-Nov'),
        ]:
            mask        = np.isin(month_flat, months)
            hourly_mean = np.array([np.nanmean(data[mask & (hour_flat == h)]) for h in hours])
            hours_fine, smooth = _periodic_diurnal_curve(hours, hourly_mean)
            if clip_zero:
                smooth = np.clip(smooth, 0, None)
            ax.plot(hours,      hourly_mean, color=color, lw=0, linestyle='none', marker='o', ms=4, alpha=0.6)
            ax.plot(hours_fine, smooth,      color=color, lw=2, linestyle=ls, label=season_label)

        ax.set_xticks(tick_hours)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlim(0, 24)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel('Hour (UTC)', fontsize=16)
        ax.set_axisbelow(True)                                     # grid lines under the curves
        ax.set_xticks(np.arange(0, 25, 3), minor=True)             # one fine line per 3-hourly time step
        ax.grid(True, which='major', lw=0.4, alpha=0.35)
        ax.grid(True, which='minor', axis='x', lw=0.25, alpha=0.25)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.spines[['top', 'right']].set_visible(False)
        if not clip_zero:
            ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=16, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.savefig(os.path.join(_core.IMAGES_DIR,
        f'meandiurnalcycle{"_" + label if label else ""}.png'),
        dpi=150, bbox_inches='tight')
    plt.show()


@recorded
def plot_acf_pacf(var_specs_lw, residuals, filename='acf_pacf_lakewide.png'):
    """Plot the ACF and PACF of the raw series and of the AR residuals, one row per variable.

    The number of rows and the row labels follow var_specs_lw, so a precipitation and wind comparison (2 rows) and the full
    precipitation, wind and divergence diagnostic (3 rows) use the same function. Lags 8 and 16 (one and two days) are marked.

    Parameters:
    var_specs_lw (list): (var_name, series, color) tuples, one per variable, in the order drawn from top to bottom.
    residuals (dict): var_name -> AR-residual series, aligned with the series of var_specs_lw.
    filename (str): File name under plotting_core.IMAGES_DIR. Pass a different name for a variable subset, for example
        'acf_pacf_lakewide_precipwind.png', so that it does not overwrite the full figure. Default 'acf_pacf_lakewide.png'.
    """
    n_vars = len(var_specs_lw)
    fig, axes = plt.subplots(n_vars, 4, figsize=(22, 4 * n_vars))
    axes = np.atleast_2d(axes)    # plt.subplots returns a 1-D array for a single variable

    col_titles = ['ACF — Raw data', 'PACF — Raw data',
                  'ACF — Residuals', 'PACF — Residuals']
    row_labels = [var_name.capitalize() for var_name, series, color in var_specs_lw]

    for row, (var_name, series, color) in enumerate(var_specs_lw):
        for col, (data, func, kwargs) in enumerate([
            (series,              plot_acf,  {}),
            (series,              plot_pacf, {'method': 'ywm'}),
            (residuals[var_name], plot_acf,  {}),
            (residuals[var_name], plot_pacf, {'method': 'ywm'}),
        ]):
            ax = axes[row, col]
            func(data[~np.isnan(data)], lags=24, ax=ax, color=color, title='', markersize=3.5, **kwargs)
            _small_markers(ax)
            ax.axvline(8,  color='gray', ls='--', alpha=0.5)
            ax.axvline(16, color='gray', ls=':',  alpha=0.5)
            ax.set_xlabel('Lag (3-hourly steps)' if row == n_vars - 1 else '', fontsize=16)
            ax.set_title('')

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=16, fontweight='bold', pad=8)

    # the variable name is the y label of the first column
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=16, fontweight='bold')

    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=16)
        ax.set_xlabel(ax.get_xlabel(), fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(_core.IMAGES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.show()


@recorded
def plot_spinup_threshold_sensitivity(results, thresholds, threshold_levels, filename='spinup_threshold_sensitivity.png'):
    """Plot the number of extreme-exceedance storm objects per forecast lead day as bar charts.

    The input is the (results, thresholds) output of preprocessing.compute_spinup_thresholds. There is one row per threshold
    level and three columns (precipitation only, wind only, compound exceedance), with the selected lead-day 7 to 9 window
    shaded.

    Parameters:
    results (dict): threshold_level -> {lead_day: [n_precip, n_wind, n_compound]}.
    thresholds (dict): threshold_level -> (precip_threshold, wind_threshold).
    threshold_levels (list): Threshold levels (keys of results and thresholds) to plot.
    filename (str): File name under plotting_core.IMAGES_DIR. Default 'spinup_threshold_sensitivity.png'.
    """
    lead_days = sorted(results[threshold_levels[0]].keys())
    n_levels  = len(threshold_levels)
    fig, axes = plt.subplots(n_levels, 3, figsize=(18, 3.5 * n_levels),
                              sharex=True, sharey=False)
    if n_levels == 1:
        axes = axes[np.newaxis, :]
    for row, pct in enumerate(threshold_levels):
        thresh_p, thresh_w = thresholds[pct]
        counts = results[pct]
        for col, (idx, title, color) in enumerate([
            (0, "Extreme Precipitation",     C_PRECIP),
            (1, "Extreme Wind",       C_WIND),
            (2, "Compound Extremes", 'grey'),
        ]):
            ax = axes[row, col]
            ax.bar(lead_days, [counts[ld][idx] for ld in lead_days],
                   color=color, edgecolor='white')
            ax.axvspan(6.5, 9.5, color='orange', alpha=0.2, label='days 7–9')
            ax.set_xticks(lead_days)
            ax.grid(axis='y', alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=16, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"Top {pct}%\n(p>{thresh_p:.2f}\nw>{thresh_w:.2f})", fontsize=16)
            if row == n_levels - 1:
                ax.set_xlabel('Forecast lead day', fontsize=16)
            if row == 0 and col == 0:
                ax.legend(fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(_core.IMAGES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.show()


@recorded
def plot_pre_post_garch_volatility(var_name, color, resid_pre, resid_post):
    """Plot the AR-only (pre-GARCH) and the final standardised (post-GARCH) residuals of one variable in a 2 x 2 figure.

    Top left: the pre-GARCH residuals over time. Top right: the ACF of their squares. Bottom left: the post-GARCH residuals over
    time. Bottom right: the ACF of their squares. Call it once per variable. The figure is saved as
    'cyclic_pre_post_garch_volatility_<var_name>.png' under plotting_core.IMAGES_DIR (the file names of the thesis figures).

    Parameters:
    var_name (str): Variable name, used in the file name.
    color (str): Plot colour.
    resid_pre (np.ndarray): Residuals of the AR step.
    resid_post (np.ndarray): Standardised residuals after the GARCH step.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row, (resid, stage) in enumerate([(resid_pre, 'Pre-GARCH (AR-only)'),
                                           (resid_post, 'Post-GARCH (standardised)')]):
        resid_sq = resid ** 2

        ax = axes[row, 0]
        ax.plot(resid, color=color, lw=0.5, alpha=0.6)
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Residual')
        ax.set_title(f'{stage} resid.', fontsize=13, fontweight='bold', pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=45)

        ax = axes[row, 1]
        plot_acf(resid_sq[~np.isnan(resid_sq)], lags=24, ax=ax, color=color, title='', markersize=3.5)
        _small_markers(ax)
        ax.axvline(8,  color='gray', ls='--', alpha=0.5)
        ax.axvline(16, color='gray', ls=':',  alpha=0.5)
        ax.set_xlabel('Lag (3-hourly steps)')
        ax.set_title(f'{stage} — ACF of squared resid.', fontsize=13, fontweight='bold', pad=10)

    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f'cyclic_pre_post_garch_volatility_{var_name}.png'
    plt.savefig(os.path.join(_core.IMAGES_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()

@recorded
def plot_deseasonalisation_ar_pipeline(var_name, color, raw, deseasonalised, ar_resid):
    """Plot a series at three stages in a 1 x 3 figure: raw, deseasonalised, and deseasonalised plus AR residuals.

    The panels autoscale independently on purpose: deseasonalisation can remove so much variance before the AR step that a
    shared y axis would flatten the later panels into lines. The figure is saved as
    'cyclic_deseasonalisation_ar_pipeline_<var_name>.png' under plotting_core.IMAGES_DIR (the file names of the thesis figures).

    Parameters:
    var_name (str): Variable name, used in the axis label and the file name.
    color (str): Plot colour.
    raw (np.ndarray): Raw series.
    deseasonalised (np.ndarray): Deseasonalised series.
    ar_resid (np.ndarray): Residuals of the AR step.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    stages = [('Raw', raw), ('Deseasonalised', deseasonalised), ('Deseasonalised + AR', ar_resid)]
    for ax, (title, series) in zip(axes, stages):
        ax.plot(series, color=color, lw=0.5, alpha=0.6)
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.4)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel(var_name.capitalize())
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f'cyclic_deseasonalisation_ar_pipeline_{var_name}.png'
    plt.savefig(os.path.join(_core.IMAGES_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()


