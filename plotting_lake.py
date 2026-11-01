"""plotting_lake.py

Figure of the Lake Victoria grid in the report: the divergence of one day at four spatial scales (single pixels, the 6 x 6 and
3 x 3 partitions and the lake mean).

Figures are saved as PNG under plotting_core.IMAGES_DIR, which plotting_core.configure sets per notebook.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

import plotting_core as _core
from figure_replay import recorded


def _span_colorbar(cax, panels, narrower=0.75):
    """Stretch a colour bar over the full height of a stack of panels, whatever the size of the figure.

    The position is set whenever the figure is drawn (axes locator), after the layout has run and the equal-aspect panels have
    their final boxes, so the bar still spans exactly from the top of the highest to the bottom of the lowest panel when the figure
    is resized to its print width. The bar keeps the left edge the layout gave it and gets `narrower` times that width.

    Parameters:
    cax (matplotlib.axes.Axes): Colour bar axes, changed in place.
    panels (list): The panels the bar belongs to.
    narrower (float): Width relative to the width the layout gave the bar. Default 0.75.
    """
    from matplotlib.transforms import Bbox

    def locator(ax, renderer):
        """Axes locator that places the colour bar over the full height of the panels.

        Parameters:
        ax (matplotlib.axes.Axes): The colour bar axes.
        renderer: Renderer passed by matplotlib; not used.

        Returns:
        matplotlib.transforms.Bbox: Position of the colour bar.
        """
        ax.set_box_aspect(None)             # the layout sets a fixed aspect on a colour bar at every draw; the height comes from the panels
        boxes = []
        for p in panels:
            p.apply_aspect()                                   # the box the equal-aspect panel will really be drawn in
            boxes.append(p.get_position())
        top, bottom = max(b.y1 for b in boxes), min(b.y0 for b in boxes)
        own = ax.get_position(original=True)
        return Bbox.from_extents(own.x0, bottom, own.x0 + own.width * narrower, top)

    cax.set_axes_locator(locator)


@recorded
def plot_divergence_scale_example(lakepts, lon_sub, lat_sub, lake_sub, time_valid,
                                  partitions=None, date=None, hours=(4, 16), seed=0,
                                  cmap='RdBu_r', filename='divergence_scale_example.png'):
    """4 x 2 figure for one day: lake divergence at night and in the afternoon
    at four spatial scales (rows, finest to coarsest): no averaging (single
    pixels), averages over the 36 parts of the 6 x 6 partition, averages over
    the nine parts of the 3 x 3 partition, and the lake-wide average.

    Red is divergence, blue is convergence. All panels share one symmetric
    colour scale (98th percentile of the absolute pixel divergence), shown in a
    single colour bar spanning the four rows, so the shrinking range with
    averaging stays visible.

    Parameters:
    lakepts (dict): 'div' -> array (n_t, n_lake_pixels).
    lon_sub, lat_sub (array): 2D (or 1D) longitude / latitude of the lake box.
    lake_sub (numpy.ndarray): boolean 2D lake mask of the box; the pixels of
        lakepts are the True entries in C order.
    time_valid (array-like): time axis of lakepts.
    partitions (dict or None): {9: masks, 36: masks} with boolean masks over the
        lake pixels (spatial_aggregation.make_partitions). None builds them from
        the pixel coordinates.
    date (str or None): day to plot, e.g. '2021-03-10'. None draws a random day
        among those with a time step within two hours of both requested hours.
    hours (tuple): requested hours (night, day); the nearest available time
        step of the day is used and its time stamp is shown in the title.
    seed (int): seed of the random day.
    cmap (str): diverging colour map.
    filename (str): saved under _core.IMAGES_DIR.

    Returns:
    dict: date, the two time stamps and the lake-mean divergence at each.
    """
    tv = pd.DatetimeIndex(time_valid)
    div = np.asarray(lakepts['div'], float)
    lake_sub = np.asarray(lake_sub, bool)
    lon2, lat2 = np.asarray(lon_sub), np.asarray(lat_sub)
    if lon2.ndim == 1:
        lon2, lat2 = np.meshgrid(lon2, lat2)
    if partitions is None:
        from spatial_aggregation import make_partitions    # imported here: it pulls in the model code and is needed only now
        partitions = {k: make_partitions(lat2[lake_sub], lon2[lake_sub], k) for k in (9, 36)}

    def nearest(day, h):
        """Index of the time step on a day closest to an hour.

        Parameters:
        day (pd.Timestamp): Normalised date.
        h (int): Hour of day (UTC).

        Returns:
        int: Index of the step, or None if the day has no step within 2 hours or the step has no finite divergence.
        """
        idx = np.flatnonzero(tv.normalize() == day)
        if not len(idx):
            return None
        k = idx[np.argmin(np.abs(tv[idx].hour - h))]
        return k if abs(tv[k].hour - h) <= 2 and np.isfinite(div[k]).any() else None

    days = pd.DatetimeIndex(tv.normalize().unique())
    if date is None:
        valid = [d for d in days if all(nearest(d, h) is not None for h in hours)]
        day = valid[int(np.random.default_rng(seed).integers(len(valid)))]
    else:
        day = pd.Timestamp(date).normalize()
    ks = [nearest(day, h) for h in hours]
    if any(k is None for k in ks):
        raise ValueError(f"{day.date()} has no usable time step near hours {hours}")

    def to_grid(vals):
        """Place the values of the lake pixels on the 2D grid of the lake box.

        Parameters:
        vals (np.ndarray): One value per lake pixel.

        Returns:
        np.ndarray: Grid of the lake box, NaN outside the lake.
        """
        g = np.full(lake_sub.shape, np.nan)
        g[lake_sub] = vals
        return g

    def averaged(row, masks):
        """Replace the values of every part by the mean over that part.

        Parameters:
        row (np.ndarray): One value per lake pixel.
        masks (list): Boolean pixel masks of the parts.

        Returns:
        np.ndarray: Array shaped like row with the part means.
        """
        out = np.empty_like(row)
        for m in masks:
            out[m] = np.nanmean(row[m])
        return out

    # night / day from the local solar time of the lake (UTC hour + longitude / 15; night is 18:00 to 06:00), as in notebook 00
    solar_shift = float(np.nanmean(lon2[lake_sub])) / 15.0
    day_night = ['Night' if ((tv[k].hour + solar_shift) % 24 >= 18 or (tv[k].hour + solar_shift) % 24 < 6) else 'Day' for k in ks]
    n_pix = int(lake_sub.sum())
    rows = [('No spatial averaging', [div[k] for k in ks], None),
            ('Partial 6$\\times$6', [averaged(div[k], partitions[36]) for k in ks], 36),
            ('Partial 3$\\times$3', [averaged(div[k], partitions[9]) for k in ks], 9),
            ('Lake-wide average', [np.full(n_pix, np.nanmean(div[k])) for k in ks], 1)]
    means = [float(np.nanmean(div[k])) for k in ks]
    vmax = float(np.nanpercentile(np.abs(np.concatenate([div[k] for k in ks])), 98))

    fig, axes = plt.subplots(4, 2, figsize=(8.4, 13.2), sharex=True, sharey=True, constrained_layout=True)
    letters = 'abcdefgh'
    for r, (label, vals, n_parts) in enumerate(rows):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            ax.set_facecolor('#D9D9D9')
            im = ax.pcolormesh(lon2, lat2, np.ma.masked_invalid(to_grid(vals[c])), cmap=cmap,
                               vmin=-vmax, vmax=vmax, shading='auto')
            ax.set_aspect('equal')
            ax.text(0.03, 0.95, f"({letters[2 * r + c]})", transform=ax.transAxes, fontsize=10,
                    fontweight='bold', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))
            if r == 0:
                ax.set_title(f"{day_night[c]}\n{tv[k]:%Y-%m-%d %H:%M}", fontsize=11)
                share = float(np.mean(div[k][np.isfinite(div[k])] < 0)) * 100
                note = f"{share:.0f}% of pixels convergent"
            elif n_parts in (9, 36):
                masks = partitions[n_parts]
                conv = sum(float(np.nanmean(div[k][m])) < 0 for m in masks)
                note = f"{conv} of {n_parts} parts convergent"
            else:
                note = f"Lake mean = {means[c] * 1e5:+.2f}$\\times 10^{{-5}}$ s$^{{-1}}$"
            ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=8.5,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        axes[r, 0].set_ylabel(f"{label}\nLatitude", fontsize=10)
    # one bar for all four rows. The layout reserves its place (as before); when the figure is drawn, the bar is stretched from the
    # top of the top row to the bottom of the bottom row and made 25% narrower than the layout width (see _span_colorbar).
    cb = fig.colorbar(im, ax=axes, shrink=0.9, fraction=0.04, pad=0.02)
    _span_colorbar(cb.ax, list(axes[:, 0]) + list(axes[:, 1]), narrower=0.75)
    fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((0, 0))
    cb.formatter = fmt; cb.update_ticks()
    cb.set_label('Divergence (s$^{-1}$)')
    for ax in axes[3, :]:
        ax.set_xlabel('Longitude')
    plt.savefig(os.path.join(_core.IMAGES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    return dict(date=day, stamps=[tv[k] for k in ks], lake_mean=means)
