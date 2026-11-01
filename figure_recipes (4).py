"""Drawing functions for the figures of the report that are drawn from finished numbers.

Every function takes the finished numbers (arrays, data frames, dicts) as arguments and only draws. The calls are recorded by
figure_replay, so `python replot.py <name>` redraws a figure from the stored arguments in seconds. To change the look of a figure,
edit the function here and run replot.py; no notebook has to be rerun.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotting_core
from figure_replay import recorded


def _show():
    """plt.show() in an interactive session, nothing under a non-interactive backend (replot.py)."""
    if 'agg' not in mpl.get_backend().lower():
        plt.show()


def _save(fig, filename):
    """Save under plotting_core.IMAGES_DIR (PNG at print resolution plus the PDF, see plotting_core).

    Parameters:
    fig (matplotlib.figure.Figure): The figure.
    filename (str): File name under plotting_core.IMAGES_DIR.
    """
    fig.savefig(os.path.join(plotting_core.IMAGES_DIR, filename))


# ── notebook 03 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _calibration_figure(G, best, grid, chosen, group_col, best_col, name, filename, n_cells):
    """Two panels of a calibration: mean log excess-CVPE ratio relative to the value 0, and the best value per cell.

    Parameters:
    G (pd.DataFrame): Index (group, value) with columns 'mean' and 'se' of the log ratio of the excess CVPE to value 0.
    best (pd.DataFrame): One row per cell with columns group_col and best_col.
    grid (list): Values on the x axis.
    chosen (dict): group -> chosen value.
    group_col (str): Column of best that holds the group.
    best_col (str): Column of best that holds the best value of the cell.
    name (str): Name of the tuned quantity ('penalty weight' or 'weight decay').
    filename (str): Output file name.
    n_cells (int): Number of cells per group (for the y label).
    """
    groups = list(chosen)
    col = dict(zip(groups, ["#0072B2", "#D55E00", "#009E73"]))
    mk = dict(zip(groups, ["o", "s", "^"]))
    ls = dict(zip(groups, ["-", "--", ":"]))
    fig, axes = plt.subplots(1, 2, figsize=plotting_core.print_figsize(2.9))
    xs = np.arange(len(grid))
    ax = axes[0]
    for g in groups:
        d = G.loc[g].reindex(grid)
        ax.errorbar(xs, d["mean"], yerr=d["se"], color=col[g], marker=mk[g], ls=ls[g], capsize=2, label=g)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([f"{w:g}" for w in grid])
    ax.set_xlabel(name); ax.set_ylabel("log excess-CVPE ratio")
    ax.set_title("(a) Mean over cells, +- 1 SE"); ax.legend(loc='best', fontsize=7)
    ax = axes[1]
    width = 0.8 / len(groups)
    for i, g in enumerate(groups):
        c = pd.Series(best[best[group_col] == g][best_col]).value_counts().reindex(grid, fill_value=0)
        off = (i - (len(groups) - 1) / 2) * width
        ax.bar(xs + off, c.values, width=width, color=col[g], hatch=["", "//", ".."][i % 3], edgecolor="k", lw=0.5, label=g)
        ax.plot([list(grid).index(chosen[g]) + off], [1.0], transform=ax.get_xaxis_transform(), color=col[g], ls="", marker="v", ms=7,
                clip_on=False)
    ax.set_xticks(xs); ax.set_xticklabels([f"{w:g}" for w in grid])
    ax.set_xlabel(name); ax.set_ylabel(f"cells (of {n_cells})")
    ax.set_title("(b) Best value per cell"); ax.set_ylim(0, ax.get_ylim()[1] * 1.5); ax.legend(loc='upper left', fontsize=7)
    fig.tight_layout(w_pad=2.0)
    plotting_core.fit_to_print(fig, 1.0)
    _save(fig, filename); _show()


@recorded
def penalty_calibration(G, best, LAMBDAS, CHOSEN, filename='penalty_calibration.png'):
    """Calibration of the marginal-uniformity penalty weight: mean excess CVPE per weight and the best weight per cell.

    Parameters:
    G (pd.DataFrame): Index (family, weight) with columns 'mean' and 'se' of the log excess-CVPE ratio to weight 0.
    best (pd.DataFrame): One row per cell with columns 'family' and 'best_lam'.
    LAMBDAS (list): Penalty weights on the x axis.
    CHOSEN (dict): group -> chosen weight.
    filename (str): Output file name. Default 'penalty_calibration.png'.
    """
    _calibration_figure(G, best, list(LAMBDAS), CHOSEN, 'family', 'best_lam', 'penalty weight', filename,
                        int(best.groupby('family').size().max()))


@recorded
def weight_decay_calibration(G, best, WDS, CHOSEN, filename='weight_decay_calibration.png'):
    """Calibration of the weight decay: mean excess CVPE per weight decay and the best weight decay per cell.

    Parameters:
    G (pd.DataFrame): Index (group, weight decay) with columns 'mean' and 'se' of the log ratio to weight decay 0.
    best (pd.DataFrame): One row per cell with columns 'group' and 'best_wd'.
    WDS (list): Weight decays on the x axis.
    CHOSEN (dict): group -> chosen weight decay.
    filename (str): Output file name. Default 'weight_decay_calibration.png'.
    """
    _calibration_figure(G, best, list(WDS), CHOSEN, 'group', 'best_wd', 'weight decay', filename,
                        int(best.groupby('group').size().max()))


# ── notebook 05 ───────────────────────────────────────────────────────────────────────────────────────────────────────────


@recorded
def chi_grid_empirical(GRID_FULL, filename='chi_grid_empirical.png'):
    """Empirical co-exceedance chi(q) with a block-bootstrap band, upper and lower tail, for all steps and both regimes.

    Same layout, colours, panel titles and axis labels as chi_grid_model_implied. Each point carries the number of joint
    exceedances observed / expected under independence. The label sits on the side of the point that points away from the
    independence line, so that it does not cross that line.

    Parameters:
    GRID_FULL (pd.DataFrame): Columns group, tail, q, chi, chi_lo, chi_hi, independence, n_joint, expected_joint_indep.
    filename (str): Output file name. Default 'chi_grid_empirical.png'.
    """
    REGIMES = (('all', 'all time steps'), ('low', 'low divergence'), ('high', 'high divergence'))
    fig, axes = plt.subplots(2, 3, figsize=plotting_core.print_figsize(5.2))
    for ri, tail in enumerate(('upper', 'lower')):
        for ci, (reg, reg_label) in enumerate(REGIMES):
            ax = axes[ri, ci]
            g = GRID_FULL[(GRID_FULL.group == reg) & (GRID_FULL['tail'] == tail)].sort_values('q')
            ax.fill_between(g.q, g.chi_lo, g.chi_hi, color='#D55E00', alpha=0.18, lw=0, label='Empirical 95% block-bootstrap band')
            ax.plot(g.q, g.chi_lo, color='#D55E00', lw=0.7, ls='-', zorder=4)
            ax.plot(g.q, g.chi_hi, color='#D55E00', lw=0.7, ls='-', zorder=4)
            ax.plot(g.q, g.independence, 'k:', lw=1.6, label='Independence (1 - q)', zorder=5)
            ax.plot(g.q, g.chi, '-o', color='#D55E00', lw=1.8, ms=4, label='Empirical', zorder=6)
            lo_y = min(g.chi_lo.min(), g.independence.min()); hi_y = max(g.chi_hi.max(), g.independence.max())
            pad = 0.22 * (hi_y - lo_y)                           # room for the labels above and below the band
            for _, row in g.iterrows():
                # Both the independence line and the empirical curve fall from left to right. A label up and to the right of a point
                # on or above the independence line, or down and to the left of a point below it, therefore stays clear of that line.
                above = row.chi >= row.independence
                ax.annotate(f"{int(row.n_joint)}/{row.expected_joint_indep:.0f}", (row.q, row.chi), textcoords='offset points',
                            xytext=(3, 4) if above else (-3, -4), ha='left' if above else 'right',
                            va='bottom' if above else 'top', fontsize=7, color='black', zorder=7)
            ax.set_xticks(g.q.values); ax.set_xticklabels([f"{v:.2f}" for v in g.q.values])
            ax.set_xlim(g.q.min() - 0.065, g.q.max() + 0.065)    # room for the labels beside the first and last point
            ax.set_ylim(lo_y - pad, hi_y + pad)
            ax.set_title(f"{tail.capitalize()} tail, {reg_label}", fontweight='bold')
            ax.grid(True, alpha=0.25); ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlabel('$q$')
            if ci == 0:
                ax.set_ylabel(r'$\chi(q)$')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=7,
               title='Point labels: joint exceedances observed / expected under independence', title_fontsize=7)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _save(fig, filename); _show()


# ── notebook 06 ───────────────────────────────────────────────────────────────────────────────────────────────────────────


@recorded
def mcs_survivors(curves, flow_avg, wg, display_names, filename='mcs_survivors.png', flow_bands=None):
    """Kendall's tau(w) of the Acar and C-CopFlow MCS survivors, averaged over the folds.

    Parameters:
    curves (dict): Acar family key -> dict with arrays x, tau, tau_lo, tau_hi (fold-averaged curve and 95% band).
    flow_avg (dict): C-CopFlow model name -> fold-averaged tau on the grid wg.
    wg (np.ndarray): Common w grid of the C-CopFlow curves.
    display_names (dict): Acar family key -> display name.
    filename (str): Output file name. Default 'mcs_survivors.png'.
    flow_bands (dict or None): Model name of flow_avg -> (lo, hi, kind) on wg. kind 'ci' is a 95% bootstrap band and 'seed' the
        range over the training seeds. A model without an entry has no band. Default None.
    """
    doms = [(c['x'].min(), c['x'].max()) for c in curves.values()] + ([(wg.min(), wg.max())] if flow_avg else [])
    x_lo, x_hi = max(d[0] for d in doms), min(d[1] for d in doms)
    # every curve has its own colour (Okabe-Ito), line style and marker, so that it can be told apart in a grey-scale print
    STYLE = [(c, ls, mk) for c, ls, mk in zip(plotting_core.OKABE_ITO, plotting_core._LINESTYLES, 'osD^v<>P')]
    fig, ax = plt.subplots(figsize=plotting_core.print_figsize(3.4, 0.72))
    k_style = 0
    for fam, d in curves.items():
        disp = display_names.get(fam, fam.capitalize()); col, ls_, mk_ = STYLE[k_style % len(STYLE)]; k_style += 1
        m = (d['x'] >= x_lo) & (d['x'] <= x_hi)
        ax.plot(d['x'][m], d['tau'][m], color=col, ls=ls_, lw=1.4, marker=mk_, markevery=max(1, int(m.sum() // 6)), ms=3.5,
                label=f'{disp} (Acar)')
        ax.fill_between(d['x'][m], d['tau_lo'][m], d['tau_hi'][m], color=col, alpha=0.12, lw=0)
    mg = (wg >= x_lo) & (wg <= x_hi)
    for name, y in flow_avg.items():
        col, ls_, mk_ = STYLE[k_style % len(STYLE)]; k_style += 1
        lo_hi_kind = (flow_bands or {}).get(name)
        note = {None: 'no band', 'ci': '95% band', 'seed': 'seed range'}[lo_hi_kind[2] if lo_hi_kind else None]
        ax.plot(wg[mg], y[mg], color=col, ls=ls_, lw=1.4, marker=mk_, markevery=max(1, int(mg.sum() // 6)), ms=3.5,
                label=f'{name} ({note})')
        if lo_hi_kind:
            ax.fill_between(wg[mg], lo_hi_kind[0][mg], lo_hi_kind[1][mg], color=col, alpha=0.12, lw=0)
    ax.axhline(0, color='black', lw=0.6, ls=(0, (4, 3)), alpha=0.5)
    ax.set_xlabel('w (standardised divergence)')
    ax.set_ylabel("Kendall's $\\tau$")
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=6.5); ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); _save(fig, filename); _show()


@recorded
def chi_grid_model_implied(GRID_FULL, CURVES, QS, filename='chi_grid_model_implied.png'):
    """Model-implied co-exceedance chi_m(q) of the MCS survivors against the empirical band; upper tail on top, lower below.

    Parameters:
    GRID_FULL (pd.DataFrame): Empirical grid (columns group, tail, q, chi, chi_lo, chi_hi, independence).
    CURVES (dict): model name -> {'upper': {regime: array over QS}, 'lower': {regime: array over QS}}.
    QS (array-like): Exceedance levels of the curves.
    filename (str): Output file name. Default 'chi_grid_model_implied.png'.
    """
    REGIMES = (('all', 'all time steps'), ('low', 'low divergence'), ('high', 'high divergence'))
    # a group of survivors is one colour, line style and marker (Okabe-Ito, readable in grey-scale)
    GROUP = {'Unconditional': ('#009E73', 'o', (0, (5, 1.5))), 'Binary': ('#E69F00', 's', (0, (1, 1))), 'Acar': ('#0072B2', '^', '-')}

    def group_of(m):
        """Group of a model name: 'Acar', 'Binary' (divergence split) or 'Unconditional'.

        Parameters:
        m (str): Model name.

        Returns:
        str: The group.
        """
        return 'Acar' if m.startswith('Acar') else ('Binary' if m.startswith('Divergence split') else 'Unconditional')

    by_group = {}
    for m in CURVES:
        by_group.setdefault(group_of(m), []).append(m)
    fig, axes = plt.subplots(2, 3, figsize=plotting_core.print_figsize(5.2))
    outside = set()
    for ri, tail in enumerate(('upper', 'lower')):
        for ci, (reg, reg_label) in enumerate(REGIMES):
            ax = axes[ri, ci]
            g = GRID_FULL[(GRID_FULL.group == reg) & (GRID_FULL['tail'] == tail)].sort_values('q')
            ax.fill_between(g.q, g.chi_lo, g.chi_hi, color='#D55E00', alpha=0.18, lw=0, label='Empirical 95% block-bootstrap band')
            ax.plot(g.q, g.chi_lo, color='#D55E00', lw=0.7, ls='-', zorder=4)
            ax.plot(g.q, g.chi_hi, color='#D55E00', lw=0.7, ls='-', zorder=4)
            ax.plot(g.q, g.independence, 'k:', lw=1.6, label='Independence (1 - q)', zorder=5)
            ax.plot(g.q, g.chi, '-o', color='#D55E00', lw=1.8, ms=4, label='Empirical', zorder=6)
            lo_y = min(g.chi_lo.min(), g.independence.min()); hi_y = max(g.chi_hi.max(), g.independence.max())
            pad = 0.08 * (hi_y - lo_y)
            for gname, models in by_group.items():
                gcol = GROUP[gname][0]
                for j, m in enumerate(models):
                    y = CURVES[m][tail][reg]
                    if np.nanmin(y) < lo_y - pad or np.nanmax(y) > hi_y + pad:
                        outside.add(m)
                    ax.plot(QS, y, marker=GROUP[gname][1], ls=GROUP[gname][2], ms=2.8, lw=0.9, color=gcol, alpha=0.8, zorder=3,
                            label=f'{gname}: MCS survivors ({len(models)})' if (ri, ci, j) == (0, 0, 0) else None)
            ax.set_ylim(lo_y - pad, hi_y + pad)                  # window set by the empirical band and independence
            ax.set_title(f"{tail.capitalize()} tail, {reg_label}", fontweight='bold')
            ax.grid(True, alpha=0.25); ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlabel('$q$')
            if ci == 0:
                ax.set_ylabel(r'$\chi(q)$')
    if outside:
        print("Curves partly outside the plotted window (window = empirical band and independence):", sorted(outside))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    _save(fig, filename); _show()


# ── notebook 07 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
PART_COLOURS = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#F0E442', '#999999']      # Okabe-Ito, light enough for the numbers


def colour_parts(masks, lon, lat):
    """Greedy colouring of the parts so that neighbouring parts never share a colour (a handful of colours is then enough).

    Parameters:
    masks (list): Boolean pixel masks, one per part.
    lon (np.ndarray): Pixel longitudes.
    lat (np.ndarray): Pixel latitudes.

    Returns:
    list: Colour index of every part.
    """
    from scipy.spatial import cKDTree
    pts = np.column_stack([lon, lat]); step = np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])
    owner = np.full(len(lon), -1)
    for j, m in enumerate(masks):
        owner[m] = j
    tree = cKDTree(pts); nbrs = [set() for _ in masks]
    for a, b in tree.query_pairs(1.6 * step):
        if owner[a] != owner[b] and owner[a] >= 0 and owner[b] >= 0:
            nbrs[owner[a]].add(owner[b]); nbrs[owner[b]].add(owner[a])
    colour = {}
    for j in sorted(range(len(masks)), key=lambda j: -len(nbrs[j])):
        used = {colour[n] for n in nbrs[j] if n in colour}
        colour[j] = next(c for c in range(len(masks)) if c not in used)
    return [colour[j] for j in range(len(masks))]


def grid_numbers(masks, lon, lat):
    """Number of every part, counted from left to right and from top to bottom (1 2 3 / 4 5 6 / 7 8 9 for nine parts).

    The parts are cut into columns by longitude and into rows by latitude within each column (aggregation make_partitions), so the
    number follows from the position of the part: its column is the rank of its mean longitude and its row the rank of its mean
    latitude within the column, counted from the top. This is a display order only; the part ids of the results keep their own order.

    Parameters:
    masks (list): Boolean pixel masks, one per part (k = m * m of them).
    lon (np.ndarray): Pixel longitudes.
    lat (np.ndarray): Pixel latitudes.

    Returns:
    list: Number (starting at 1) of every part, in the order of masks. The order of masks if k is not a perfect square.
    """
    k = len(masks); m = int(round(np.sqrt(k)))
    if m * m != k:
        return list(range(1, k + 1))
    lon = np.asarray(lon); lat = np.asarray(lat)
    cx = np.array([lon[mk].mean() for mk in masks]); cy = np.array([lat[mk].mean() for mk in masks])
    num = [0] * k
    for c, idx in enumerate(np.array_split(np.argsort(cx, kind='stable'), m)):
        for r, j in enumerate(sorted(idx, key=lambda j: -cy[j])):
            num[j] = r * m + c + 1
    return num


@recorded
def spatial_partitions(parts, lon_lake, lat_lake, filename='spatial_partitions.png'):
    """Map of the lake pixels split into equal-size parts, one panel per number of parts.

    Parameters:
    parts (dict): number of parts -> list of boolean pixel masks.
    lon_lake (np.ndarray): Pixel longitudes.
    lat_lake (np.ndarray): Pixel latitudes.
    filename (str): Output file name. Default 'spatial_partitions.png'.
    """
    fig, axes = plt.subplots(1, 2, figsize=plotting_core.print_figsize(2.9, 1.0))
    for ax, (k, masks) in zip(axes, parts.items()):
        cols = colour_parts(masks, lon_lake, lat_lake)
        nums = grid_numbers(masks, lon_lake, lat_lake)
        for j, m in enumerate(masks):
            ax.scatter(lon_lake[m], lat_lake[m], s=5 if k == 36 else 7, color=PART_COLOURS[cols[j] % len(PART_COLOURS)], linewidths=0)
            ax.text(lon_lake[m].mean(), lat_lake[m].mean(), str(nums[j]), ha='center', va='center', fontsize=7 if k == 9 else 5.5,
                    fontweight='bold', color='black')
        ax.set_title(f"{k} parts"); ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_aspect('equal')
    fig.tight_layout(); _save(fig, filename); _show()


