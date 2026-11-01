"""plotting_core.py

Shared plotting settings of the thesis figures: the matplotlib style, the print standard (every figure is saved as a PNG at the
size it is printed), the output folder (configure), the rc-context decorator with_rc, and the colour palettes of the regimes,
the copula families and the network variants.
"""
import os
import warnings

import matplotlib as mpl
import numpy as np

from copula_families import FAMILY_DISPLAY as _FAM_DISPLAY

mpl.rcParams['font.family']      = 'serif'
mpl.rcParams['font.serif']       = ['DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['text.usetex']      = False

# ── print standard of all figures ─────────────────────────────────────────────────────────────────────
# A figure is drawn at the size at which it is printed (TEXT_WIDTH_IN times the fraction of the text width it is included at),
# so a font size in the code is the font size on paper. The captions are written in the report, not in the figure: figure-level
# titles and sentences are removed when a figure is saved (see _print_savefig); titles of single panels stay.
TEXT_WIDTH_IN = 6.30          # \textwidth of the report (A4, 2.5 cm margins), in inches
PRINT_DPI = int(os.environ.get('THESIS_PRINT_DPI', '600'))   # resolution of every saved PNG; set THESIS_PRINT_DPI=300 for a quick draft
# Okabe-Ito palette (Okabe & Ito, 2008): distinguishable with the common forms of colour-vision deficiency. In the cycle every
# colour comes with its own line style, so lines can also be told apart in a grey-scale print.
OKABE_ITO = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000')
_LINESTYLES = ('-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (7, 2)))
mpl.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'legend.title_fontsize': 8, 'figure.titlesize': 9,
    'axes.prop_cycle': mpl.cycler(color=OKABE_ITO, linestyle=_LINESTYLES),
    'lines.linewidth': 1.3, 'lines.markersize': 4.5, 'axes.linewidth': 0.7,
    'savefig.dpi': PRINT_DPI, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(_ROOT, 'Images')


def _strip_figure_captions(fig):
    """Remove the figure-level title and figure-level sentences, which belong in the caption of the report.

    The suptitle goes; so does every horizontal figure-level text of 45 characters or more. Rotated texts (shared axis labels), short
    texts (shared x labels, panel letters) and the titles of the single panels are kept.

    Parameters:
    fig (matplotlib.figure.Figure): The figure, changed in place.
    """
    st = getattr(fig, '_suptitle', None)
    if st is not None:
        st.set_visible(False)
    for t in list(fig.texts):
        if abs(float(t.get_rotation())) < 1.0 and len(t.get_text().strip()) >= 45:
            t.set_visible(False)


# Share of the text width at which the report includes a figure (default 1). A figure is fitted to this printed width.
FIG_WIDTH_FRACTION = {'divergence_scale_example': 0.9, 'meandiurnalcycle': 0.8,
                      'spatial_partitions': 0.7, 'mcs_survivors': 0.72}
MIN_PRINT_PT = 7.0            # smallest text size on paper (tick labels, legends, annotations)


THIN_LINES = os.environ.get('THESIS_THIN_LINES', '1') == '1'    # '1': lines shrink with the figure (fine hairlines); '0': never thinner than 0.6 pt


def _scaled_linewidth(lw, k, dense=False):
    """Line width on paper after a figure is shrunk by the factor k.

    With THIN_LINES (the default) a line keeps its drawn proportion to the figure, as when a large figure is shrunk in the report:
    the width is lw * k, but not below 0.3 pt, and not below 0.12 pt for a dense line (more than 300 points, for example a time
    series), so that the detail of such a line stays visible. With THIN_LINES off, the width is lw * max(k, 0.6), at least 0.6 pt.

    Parameters:
    lw (float): Line width in the large figure, in points.
    k (float): Shrink factor of the canvas.
    dense (bool): True for a line with many points. Default False.

    Returns:
    float: Line width in points.
    """
    if THIN_LINES:
        return max(lw * k, 0.12 if dense else 0.3)
    return max(lw * max(k, 0.6), 0.6)


def fit_to_print(fig, fraction=1.0, min_pt=MIN_PRINT_PT):
    """Fit a figure to its printed width, so that no text is smaller than min_pt on paper.

    A figure drawn on a canvas wider than the printed width would be shrunk by the report, and its text with it. The canvas is set to
    the printed width (keeping the aspect ratio) and every text keeps its size on paper, except that texts that would end below
    min_pt are raised to min_pt. A figure that is already at most as wide as the printed width is left as it is.

    Parameters:
    fig (matplotlib.figure.Figure): The figure, changed in place.
    fraction (float): Share of the text width at which the report includes the figure. Default 1.0.
    min_pt (float): Smallest text size on paper. Default MIN_PRINT_PT.

    Returns:
    float: The shrink factor applied to the canvas (1.0 if the figure was left as it is).
    """
    width = TEXT_WIDTH_IN * float(fraction)
    w0, h0 = fig.get_size_inches()
    if w0 <= width * 1.02:
        return 1.0
    k = width / w0
    fig.canvas.draw()                                   # makes the tick labels exist
    for t in fig.findobj(mpl.text.Text):
        t.set_fontsize(max(t.get_fontsize() * k, min_pt))
    fig.set_size_inches(width, max(h0 * k, width / 3.0))     # a very wide row of panels gets some extra height instead of strips
    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(_scaled_linewidth(line.get_linewidth(), k, dense=len(line.get_xdata()) > 300))
            if line.get_marker() not in (None, 'None', '', ' ') and line.get_gid() != 'fixed_marker':
                line.set_markersize(max(line.get_markersize() * max(k, 0.6), 2.0))
        for coll in ax.collections:                     # scatter markers drawn large for the big canvas (small ones stay as they are)
            try:
                sizes = np.asarray(coll.get_sizes(), float)
                if sizes.size and sizes.max() >= 12 and isinstance(coll, mpl.collections.PathCollection):
                    coll.set_sizes(np.maximum(sizes * max(k, 0.5) ** 2, 4.0))
            except Exception:
                pass
    _rebuild_legends(fig, k, min_pt)
    if not fig.get_constrained_layout():
        _relayout(fig)
        _polish_text(fig)
        _relayout(fig)
    return k


def _relayout(fig):
    """Re-space the panels for the resized text; figures with manual layouts keep theirs.

    Parameters:
    fig (matplotlib.figure.Figure): The figure, changed in place.
    """
    if fig.get_constrained_layout():
        return
    rect = [0.0, 0.0, 1.0, 1.0]                         # room for legends that belong to the figure, at its top or bottom edge
    if fig.legends:
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for leg in fig.legends:
                bb = leg.get_window_extent(renderer).transformed(fig.transFigure.inverted())
                if bb.y0 > 0.5:
                    rect[3] = min(rect[3], bb.y0 - 0.01)
                else:
                    rect[1] = max(rect[1], bb.y1 + 0.01)
        except Exception:
            pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig.tight_layout(pad=0.4, rect=tuple(rect))
    except Exception:
        pass


def _rebuild_legends(fig, k, min_pt):
    """Give every legend the font size it now has, including its handle length, padding and spacing.

    A legend lays itself out in units of the font size it was created with. Shrinking only the text of a legend that was created
    with a large font leaves a frame, handles and spacing that are far too big. The legend box is therefore rebuilt with the
    shrunken font size (same handles, labels, title, location and columns).

    Parameters:
    fig (matplotlib.figure.Figure): The figure, changed in place.
    k (float): Shrink factor of the canvas.
    min_pt (float): Smallest text size on paper.
    """
    for leg in fig.findobj(mpl.legend.Legend):
        try:
            fs = max(float(leg._fontsize) * k, min_pt)
            title = leg.get_title()
            title_text, title_fs = title.get_text(), max(title.get_fontsize(), min_pt)
            labels = [t.get_text() for t in leg.texts]
            handles = list(leg.legend_handles)
            lw_k = max(k, 0.6)                          # marker size factor, as for the plotted lines
            for h in handles:
                if isinstance(h, mpl.lines.Line2D):
                    h.set_linewidth(_scaled_linewidth(h.get_linewidth(), k))     # the same width as the plotted lines, so that dashes match
                    if h.get_marker() not in (None, 'None', '', ' '):
                        h.set_markersize(max(h.get_markersize() * lw_k, 2.0))
            if any(isinstance(h, mpl.lines.Line2D) and h.get_linestyle() not in ('-', 'None', '') for h in handles):
                leg.handlelength = max(float(leg.handlelength), 3.2)      # long enough to show the dash pattern
            leg._fontsize = fs
            leg.prop.set_size(fs)
            leg._init_legend_box(handles, labels, markerfirst=getattr(leg, '_markerfirst', True))
            leg._legend_box.set_offset(leg._findoffset)           # the rebuilt box has to be placed by the legend again
            leg.set_title(title_text)
            leg.get_title().set_fontsize(title_fs)
        except Exception:
            continue                                    # a legend of an unusual kind keeps its shrunken text


def _wrap_in_middle(text):
    """Split a text into two lines at the space nearest to its middle.

    Parameters:
    text (str): A title or axis label.

    Returns:
    str: The text with one line break, or the text itself if it has no usable space.
    """
    if '\n' in text or ' ' not in text.strip():
        return text
    mid = len(text) / 2
    best = min((i for i, ch in enumerate(text) if ch == ' '), key=lambda i: abs(i - mid))
    return text[:best] + '\n' + text[best + 1:]


def _polish_text(fig):
    """Remove the typical overlaps of a figure that was shrunk to the print width.

    A panel title wider than its panel and a y label taller than its panel are wrapped onto two lines, and tick labels that touch
    the previous label are hidden (only unrotated labels, since the bounding box of a rotated label overstates its overlap).

    Parameters:
    fig (matplotlib.figure.Figure): The figure, changed in place.
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    changed = False
    for ax in fig.axes:
        pos = ax.get_window_extent(renderer)
        t = ax.title
        if t.get_text() and t.get_window_extent(renderer).width > pos.width * 1.08:
            wrapped = _wrap_in_middle(t.get_text())
            if wrapped != t.get_text():
                t.set_text(wrapped); changed = True
        yl = ax.yaxis.label
        if yl.get_text() and yl.get_window_extent(renderer).height > pos.height * 1.05:
            wrapped = _wrap_in_middle(yl.get_text())
            if wrapped != yl.get_text():
                yl.set_text(wrapped); changed = True
    if changed:
        _relayout(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for axis, horizontal in ((ax.xaxis, True), (ax.yaxis, False)):
            labels = [lab for lab in axis.get_ticklabels() if lab.get_visible() and lab.get_text()
                      and abs(lab.get_rotation()) < 1.0]
            if len(labels) < 2:
                continue
            last = None
            for lab in labels:
                bb = lab.get_window_extent(renderer)
                clash = last is not None and (bb.x0 < last.x1 + 1.0 if horizontal else bb.y0 < last.y1 + 1.0)
                if clash:
                    lab.set_visible(False)
                else:
                    last = bb


_ORIGINAL_SAVEFIG = mpl.figure.Figure.savefig


def _print_savefig(self, fname, *args, **kwargs):
    """Figure.savefig with the print standard: no figure-level caption, print resolution and tight box for PNG output.

    Only PNG files are written; no PDF is saved next to them.

    Parameters:
    fname (str or path-like): Output file.
    *args, **kwargs: As for Figure.savefig; an explicit dpi is overridden for PNG output.
    """
    _strip_figure_captions(self)
    name = os.fspath(fname) if isinstance(fname, (str, os.PathLike)) else None
    if name is not None and name.lower().endswith('.png'):
        stem = os.path.splitext(os.path.basename(name))[0]
        fit_to_print(self, FIG_WIDTH_FRACTION.get(stem, 1.0))
        kwargs['dpi'] = PRINT_DPI
        kwargs.setdefault('bbox_inches', 'tight')
    return _ORIGINAL_SAVEFIG(self, fname, *args, **kwargs)


if mpl.figure.Figure.savefig is _ORIGINAL_SAVEFIG:          # install once, also when the module is imported twice
    mpl.figure.Figure.savefig = _print_savefig


def design_title(tag):
    """Readable name of a synthetic design: 'gumbel_strong' becomes 'Gumbel, strong'; a tag without an underscore is kept.

    Parameters:
    tag (str): Design key or tag.

    Returns:
    str: Name starting with a capital.
    """
    tag = str(tag).strip()
    if '_' not in tag:
        return tag
    family, _, strength = tag.partition('_')
    return f"{family.capitalize()}, {strength.replace('_', ' ')}"


def print_figsize(height_in, fraction=1.0):
    """Figure size in inches for a figure printed at `fraction` of the text width.

    Parameters:
    height_in (float): Height in inches.
    fraction (float): Share of the text width at which the report includes the figure. Default 1.0.

    Returns:
    tuple: (width, height) in inches.
    """
    return (TEXT_WIDTH_IN * float(fraction), float(height_in))


def configure(window_label: str):
    """Route the figures of a notebook to its own folder 'Images <window_label>' and create it.

    Call once per notebook. The folder gets the suffix ' SMOKE' when THESIS_SMOKE is '1', so test output never mixes with
    real output.

    Parameters:
    window_label (str): Label identifying the calling notebook, used to build the folder name.
    """
    global IMAGES_DIR
    suffix = ' SMOKE' if os.environ.get('THESIS_SMOKE', '0') == '1' else ''     # smoke output never mixes with real output
    IMAGES_DIR = os.path.join(_ROOT, f'Images {window_label}{suffix}')
    os.makedirs(IMAGES_DIR, exist_ok=True)

# ── semantic colour palette (Okabe-Ito / Wong, colour-blind safe) ──────────────────────────────────
C_NIGHT = '#003366'   # navy: night regime
C_PRECIP       = '#56B4E9'   # sky blue: precipitation
C_WIND         = '#009E73'   # bluish green: wind speed
C_DIV_VAR      = '#E69F00'   # orange: divergence as a variable
C_DAY          = '#117733'   # dark green: day regime

# High/low-divergence regime pair (median split of x_std, see divergence_regime_masks_for_lake_plot). It is distinct from
# C_DIV_VAR, which colours divergence as a continuous variable (for example in marginal histograms). The red and blue are the
# Divergence / Convergence colours of plot_kendall_scale in notebook 06, although that figure splits on the raw sign. Checked with
# colorspacious as for the C-CopFlow variants: the pair is dE = 58 apart under normal vision and at least dE = 39 apart under
# simulated deuteranomaly, protanomaly and tritanomaly.
C_HIGH_DIV     = '#B2182B'   # red: high-divergence regime
C_LOW_DIV      = '#2166AC'   # blue: low-divergence regime

_FAMILY_COLORS = {
    'Gaussian': '#4477AA',
    't':        '#66CCEE',
    'Clayton':  '#228833',
    'Gumbel':   '#CCBB44',
    'Gumbel (180°)': '#DDAA33',   # darker gold: same family, mirrored tail
    'Frank':    '#EE6677',
    'Joe':      '#AA3377',
    'BB1':      '#BBBBBB',
    'BB6':      '#44AA99',
    'BB7':      '#332288',
    'BB8':      '#FF8800',
    'Tawn':     '#AA0000',
}

# The six C-CopFlow variants (conditioning mode x coupling type). The three colours teal, indigo and olive-brown were checked
# with colorspacious against every other colour on the same plots (the two hardcoded CODINE line colours, the black Acar
# reference line and the other three variant colours), under normal vision and simulated deuteranomaly, protanomaly and
# tritanomaly: the smallest pairwise distance is dE = 10.5, and both pairs below 15 involve the CODINE colours, not the three
# colours named above.
C_CCOPFLOW_VARIANTS = {
    'C-CopFlow (w only, affine)':                '#D55E00',  # vermillion
    'C-CopFlow (w only, spline coupling)':       '#006D77',  # teal
    'C-CopFlow (spline basis, affine)':          '#CC79A7',  # reddish purple
    'C-CopFlow (spline basis, spline coupling)': '#4B0092',  # indigo
    'C-CopFlow (FiLM, affine)':                  '#F0E442',  # yellow
    'C-CopFlow (FiLM, spline coupling)':         '#6B4C00',  # olive-brown
}

C_CCODINE_VARIANTS = {
    'C-CODINE (w only)':       '#002850',  # dark navy
    'C-CODINE (spline basis)': '#56B4E9',  # sky blue
    'C-CODINE (FiLM)':         '#E69F00',  # orange
}

# Acar's internal family keys are lowercase ('gaussian', 'student', ..., 'gumbel180'); this maps all seven to the display names
# used for the binary-split families and in _FAMILY_COLORS.
ACAR_TO_DISPLAY_FAMILY = {('student' if k == 't' else k): d for k, d in _FAM_DISPLAY.items()}
