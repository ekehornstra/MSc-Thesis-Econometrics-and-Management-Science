"""synthetic_report.py

Helpers shared by the synthetic-study notebooks AAA01-AAA10 and AAA_summary: quiet execution, the registry of the ten designs, the
five summary figures and the four tables of AAA_summary (overview, networks against the Gijbels benchmark, marginal uniformity).
Nothing here fits a model; it only reads results produced by synthetic_study.py.

References:
Gijbels, I., Veraverbeke, N., & Omelka, M. (2011). Conditional copulas, association measures and their applications.
    Computational Statistics & Data Analysis, 55(5), 1919-1932.
"""
import contextlib
import functools
import io
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.std import tqdm as _std_tqdm

import fold_pipeline as FP
import plotting_core
import reporting_tables
from synthetic_study import N_TRAIN, Q_TAIL
from figure_replay import recorded


# One switch for the classical benchmark of Gijbels et al. (2011) in every figure of this module (bar charts, tau curves and the
# summary pages of AAA_summary). False: the figures are drawn without it.
SHOW_GIJBELS = True
GIJ_COLOR = "#E8C9A0"
GIJ_NOTE = "Gijbels (kernel)"
# Switches for the other two reference estimators drawn next to the networks: the empirical checkerboard Bernstein copula of Lu and Ghosh
# (2024) and the Acar-type kernel estimator of the family that generated the data (the best-fitting of the seven families in the mixture
# designs, chosen by the lowest mean excess CVPE).
SHOW_ECBC = True
ECBC_COLOR = "#BFD8B8"
ECBC_NOTE = "ECBC (Bernstein)"
SHOW_ACAR_MATCHED = True
ACAR_COLOR = "#B5CDE3"
ACAR_NOTE = "Acar (kernel)"

GREYS = {"w only": "#D0D0D0", "spline basis": "#8A8A8A", "FiLM": "#2B3A4A"}      # light to dark


# ── quiet execution ────────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def quiet():
    """Silence progress bars, the 'thesis' logger and stray prints inside a with-block.

    Every tqdm bar created in the block is disabled, whichever way tqdm was imported (the class itself is patched, so names bound
    earlier by `from tqdm.auto import tqdm` are covered, and so is the notebook widget bar). Bars created in other processes are
    not covered unless the processes are forked inside the block. The remaining output is kept in a buffer and the last 4,000
    characters are written to the real stdout only if the block raises, so a failure is never hidden.

    Returns:
    generator: Context manager body; yields None, control to the body of the with-block.
    """
    orig_init = _std_tqdm.__init__

    @functools.wraps(orig_init)
    def disabled_init(self, *args, **kwargs):
        """Initialiser of the patched tqdm class that always disables the bar.

        Parameters:
        self: The tqdm instance.
        *args: Positional arguments of tqdm.
        **kwargs: Keyword arguments of tqdm; 'disable' is forced to True.
        """
        kwargs["disable"] = True
        orig_init(self, *args, **kwargs)
    _std_tqdm.__init__ = disabled_init
    log = logging.getLogger("thesis"); old = log.level; log.setLevel(logging.WARNING)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield
    except BaseException:
        sys.__stdout__.write(buf.getvalue()[-4000:])
        raise
    finally:
        _std_tqdm.__init__ = orig_init
        log.setLevel(old)


# ── Model Confidence Set within the C-CODINE variants and within the C-CopFlow variants ────────────


# ── grouped bar charts (one group per network family, three greys for the conditioning, unconditional CODINE unfilled) ───

FAMILIES = [("C-CODINE", [(f"C-CODINE ({c})", c) for c in ("w only", "spline basis", "FiLM")]),
            ("C-CopFlow (affine)", [(f"C-CopFlow ({c}, affine)", c) for c in ("w only", "spline basis", "FiLM")]),
            ("C-CopFlow (spline)", [(f"C-CopFlow ({c}, spline coupling)", c) for c in ("w only", "spline basis", "FiLM")])]


def _mean_ci(x):
    """Mean and the half-width of its 95% t-interval across draws (NaN values are ignored).

    Parameters:
    x (array-like): One value per data draw.

    Returns:
    tuple: (mean, half-width); (NaN, NaN) if there is no finite value and (value, 0.0) for a single one.
    """
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


# ── tau(w) figure: full range and close-ups of the two ends of the w axis ──────────────────────────


# ── report tables (one style for all: reporting_tables.df_to_latex) ────────────────────────────────


# ── the 10 synthetic notebooks: one registry shared by the overview table and the figures ──────────
DESIGN_ORDER = ("3a", "3b", "3c", "3d", "3e", "3f", "3g", "3h", "3i", "3j")        # internal DGP keys of synthetic_study.DGPS
FAMILY_LABEL = {"3a": "student", "3b": "student", "3c": "nonmonotonic", "3d": "nonmonotonic", "3e": "gumbel", "3f": "gumbel",
                "3g": "clayton", "3h": "clayton", "3i": "mixed", "3j": "mixed"}
STRENGTH_LABEL = {k: ("weak" if i % 2 == 0 else "strong") for i, k in enumerate(DESIGN_ORDER)}


def design_tag(dgp):
    """Name stem of notebook, artifact and files of a design, for example 'student_weak' for DGP 3a.

    Parameters:
    dgp (str): Internal design key.

    Returns:
    str: The tag.
    """
    return f"{FAMILY_LABEL[dgp]}_{STRENGTH_LABEL[dgp]}"


def all_designs():
    """The ten designs in notebook order AAA01-AAA10.

    Returns:
    list: One dict per design with number, dgp, n (training points per draw, synthetic_study.N_TRAIN), tag, notebook and artifact.
    """
    out = []
    for i, dgp in enumerate(DESIGN_ORDER, 1):
        tag = design_tag(dgp)
        out.append(dict(number=i, dgp=dgp, n=N_TRAIN, tag=tag, notebook=f"AAA{i:02d}_synthetic_{tag}", artifact=f"synthetic_{tag}"))
    return out


BENCHMARK_GIJBELS_COLS = ("excess", "excess_ces", "ces", "ces_calibrated", "tau_rmse", "tail_uu", "tail_ll", "tail_mixed",
                          "h", "g", "neff_min", "clipped_mass", "margin_error_max")
BENCHMARK_MODEL_COLS = ("excess", "excess_ces", "ces_calibrated", "tau_rmse", "tail_uu", "tail_ll", "tail_mixed")
BENCHMARK_MODELS = ("C-CODINE (w only)", "C-CODINE (spline basis)", "C-CODINE (FiLM)",
                    "C-CopFlow (w only, affine)", "C-CopFlow (spline basis, affine)", "C-CopFlow (FiLM, affine)",
                    "C-CopFlow (w only, spline coupling)", "C-CopFlow (spline basis, spline coupling)",
                    "C-CopFlow (FiLM, spline coupling)", "Unconditional: CODINE")


def benchmark_numbers_table(load_artifact):
    """One row per design with the numbers quoted in the thesis text: the Gijbels et al. (2011) benchmark next to the networks.

    Reads the saved artifacts only, nothing is refitted. Designs without a saved Gijbels frame are skipped.

    Parameters:
    load_artifact (callable): run_support.load_artifact, called with the artifact name of a design.

    Returns:
    pd.DataFrame: One row per design. Columns are ``design``, ``n_draws``, the mean and SD across draws of each Gijbels column
        (``<col>_mean``, ``<col>_sd``) and, for each network and the
        unconditional CODINE, the mean over draws of the error measures (``<model>|<col>``).
    """
    rows = []
    for d in all_designs():
        art = load_artifact(d["artifact"])
        g = art.get("gijbels")
        if g is None or len(g) == 0:
            continue
        cols = [c for c in BENCHMARK_GIJBELS_COLS if c in g.columns]
        m = g[cols].mean(); sd = g[cols].std(ddof=1)
        row = dict(design=d["tag"], n_draws=len(g))
        for c in cols:
            row[f"{c}_mean"] = m[c]; row[f"{c}_sd"] = sd[c]
        e = art.get("ecbc")
        if e is not None and len(e):
            for c in [c for c in BENCHMARK_GIJBELS_COLS if c in e.columns and c not in ("h", "g")]:
                row[f"ECBC|{c}_mean"] = e[c].mean(); row[f"ECBC|{c}_sd"] = e[c].std(ddof=1)
        S = art["summary"]                      # mean over draws, one row per model
        am = matched_acar_model(d["dgp"], art["summary_by_draw"])
        if am is not None:
            row["matched_acar"] = am
            for c in BENCHMARK_MODEL_COLS:
                if c in S.columns:
                    row[f"matched_acar|{c}"] = S.loc[am, c]
        for name in BENCHMARK_MODELS:
            if name in S.index:
                for c in BENCHMARK_MODEL_COLS:
                    if c in S.columns:
                        row[f"{name}|{c}"] = S.loc[name, c]
        rows.append(row)
    return pd.DataFrame(rows)


# ── AAA_summary: one error measure over all 10 designs, as one full-page figure (rows: family, columns: weak / strong tau) ───
ROW_LABELS = {"student": "Student-t", "nonmonotonic": "Student-t, nonmonotonic", "gumbel": "Gumbel", "clayton": "Clayton", "mixed": "Gumbel-Clayton mixture"}
ROW_ORDER = ("student", "nonmonotonic", "gumbel", "clayton", "mixed")
SUMMARY_PLOTS = {   # key: (metric column, y label, models shown)
    "cvpe": ("excess", "Excess CVPE over the oracle", "all"),
    "tau": ("tau_rmse", r"$\tau(w)$ RMSE", "all"),
    "lower_tail": ("tail_ll", rf"Lower-tail RMSE, $\chi(q{{=}}{Q_TAIL:g})$", "all"),
    "upper_tail": ("tail_uu", rf"Upper-tail RMSE, $\chi(q{{=}}{Q_TAIL:g})$", "all"),
    "ces": ("excess_ces", "Excess Copula Energy Score over the oracle", "all"),
}


GIJ_KEY = "Gijbels et al. (2011), kernel"        # key of the classical benchmark in the panel statistics
GIJ_METRICS = ("excess", "excess_ces", "tau_rmse", "tail_ll", "tail_uu")      # all four error measures exist in the Gijbels frame


def _gij_stats(gij_df, metric):
    """Mean and 95% t-interval across data draws of one metric of the saved Gijbels frame.

    Parameters:
    gij_df (pd.DataFrame or None): Saved Gijbels frame.
    metric (str): Metric column; only the metrics in GIJ_METRICS exist in the frame.

    Returns:
    dict: {GIJ_KEY: (mean, half-width)}, or {} if the frame or the metric is unavailable.
    """
    if gij_df is None or metric not in GIJ_METRICS or metric not in gij_df or len(gij_df) == 0:
        return {}
    d = gij_df
    x = d.groupby("rep")[metric].mean() if "rep" in d.columns else d[metric]
    return {GIJ_KEY: _mean_ci(x)}


ECBC_KEY = "Lu and Ghosh (2024), ECBC"
ACAR_KEY = "Acar (kernel), matched family"
ACAR_MATCHED = {"3a": "student", "3b": "student", "3c": "student", "3d": "student", "3e": "gumbel", "3f": "gumbel",
                "3g": "clayton", "3h": "clayton"}       # designs 3i and 3j (mixtures) have no matching family: the best-fitting one is used


_REF_HATCH = {GIJ_KEY: "//", ECBC_KEY: "\\\\", ACAR_KEY: "xx"}


def _ecbc_stats(ecbc_df, metric):
    """Mean and 95% t-interval across data draws of one metric of the saved ECBC frame (same layout as the Gijbels frame).

    Parameters:
    ecbc_df (pd.DataFrame or None): Saved ECBC frame.
    metric (str): Metric column; only the metrics in GIJ_METRICS exist in the frame.

    Returns:
    dict: {ECBC_KEY: (mean, half-width)}, or {} if the frame or the metric is unavailable.
    """
    return {ECBC_KEY: v for v in _gij_stats(ecbc_df, metric).values()}


def matched_acar_model(dgp, long_df):
    """Name of the Acar model shown as the matched-family reference of one design.

    Designs 3a to 3h use the family of the data-generating copula (Student-t, Gumbel or Clayton). The mixture designs 3i and 3j have no
    matching family; the Acar model with the lowest mean excess CVPE over the data draws of the design is used, which makes the bar a best case.

    Parameters:
    dgp (str): Internal design key ('3a' to '3j').
    long_df (pd.DataFrame): summary_by_draw of the design (rows: model x draw).

    Returns:
    str or None: Model name as in the frame, or None if the frame has no Acar rows.
    """
    if long_df is None:
        return None
    if dgp in ACAR_MATCHED:
        name = FP.acar_name(ACAR_MATCHED[dgp])
        return name if name in set(long_df["model"]) else None
    acar = long_df[long_df["model"].str.startswith("Acar")]
    if acar.empty:
        return None
    return acar.groupby("model")["excess"].mean().idxmin()


def _acar_stats(dgp, long_df, metric):
    """Mean and 95% t-interval across data draws of one metric of the matched Acar model.

    Parameters:
    dgp (str): Internal design key.
    long_df (pd.DataFrame or None): summary_by_draw of the design.
    metric (str): Metric column.

    Returns:
    dict: {ACAR_KEY: (mean, half-width)}, or {} if the model or the metric is unavailable.
    """
    name = matched_acar_model(dgp, long_df)
    if name is None or metric not in long_df.columns:
        return {}
    x = long_df[long_df["model"] == name].groupby("rep")[metric].mean()
    return {ACAR_KEY: _mean_ci(x)} if x.notna().any() else {}


def _tag_to_dgp(tag):
    """Internal design key of a design tag such as 'student_weak'.

    Parameters:
    tag (str): Design tag.

    Returns:
    str: The design key.
    """
    return next(d for d in DESIGN_ORDER if design_tag(d) == tag)


def _bar_layout(subset, with_gij=False, with_ecbc=False, with_acar=False):
    """Bar positions, colours and tick labels of one panel of the summary figures.

    Parameters:
    subset (str): 'ccodine' shows the C-CODINE family only; anything else shows all three network families.
    with_gij (bool): Add the Gijbels bar. Default False.
    with_ecbc (bool): Add the ECBC bar. Default False.
    with_acar (bool): Add the matched-family Acar bar. Default False.

    Returns:
    tuple: (layout, ticks), with layout a list of (model, colour, x, unfilled) and ticks a list of (x, label).
    """
    fams = FAMILIES[:1] if subset == "ccodine" else FAMILIES
    layout, x, ticks = [], 0.0, []
    for fam, members in fams:
        start = x
        for model, cond in members:
            layout.append((model, GREYS[cond], x, False)); x += 1
        ticks.append(((start + x - 1) / 2, fam)); x += 0.7
    x += 0.4
    layout.append((FP.UNCOND_CODINE, "white", x, True)); ticks.append((x, "CODINE\n(uncond.)"))
    if with_acar:
        x += 2.3
        layout.append((ACAR_KEY, ACAR_COLOR, x, False)); ticks.append((x, "Acar"))
    if with_gij:
        x += 2.0 if with_acar else 2.3
        layout.append((GIJ_KEY, GIJ_COLOR, x, False)); ticks.append((x, "Gijbels"))
    if with_ecbc:
        x += 2.0 if (with_acar or with_gij) else 2.3
        layout.append((ECBC_KEY, ECBC_COLOR, x, False)); ticks.append((x, "ECBC"))
    return layout, ticks


def _panel_stats(long_df, metric, layout):
    """Mean and 95% interval across draws of one metric for every model of the layout that is in the frame.

    Parameters:
    long_df (pd.DataFrame): Long frame with the columns 'rep', 'model' and the metric.
    metric (str): Column to summarise.
    layout (list): (model, ...) tuples of the models to show.

    Returns:
    dict: Model -> output of _mean_ci.
    """
    wide = long_df.pivot_table(index="rep", columns="model", values=metric)
    return {m: _mean_ci(wide[m]) for m, _, _, _ in layout if m in wide}


def summary_limits(layout_subset_stats):
    """Common y-limits over every bar (mean +- interval) of every panel of one summary figure.

    Parameters:
    layout_subset_stats (iterable): Per panel, a dict model -> (mean, half-width).

    Returns:
    tuple: (lower, upper) with headroom at the top for the row labels.
    """
    lo, hi = 0.0, 0.0
    for st in layout_subset_stats:
        for m, ci in st.values():
            if np.isfinite(m):
                lo = min(lo, m - (ci if np.isfinite(ci) else 0)); hi = max(hi, m + (ci if np.isfinite(ci) else 0))
    span = hi - lo if hi > lo else 1.0
    return (lo - 0.04 * span if lo < 0 else 0.0), hi + 0.22 * span     # headroom for the row labels inside the panels


@recorded
def summary_page_figures(long_by_tag, key, filename_stem, figsize=None, gij_by_tag=None, ecbc_by_tag=None):
    """One full-page figure for one error measure: 5 rows (designs) x 2 columns (weak, strong tau(w)) with one common y-axis.

    Missing designs stay empty. The Gijbels bar shows an excess CVPE that is the estimator-minus-oracle difference on a 32 x 32
    cell discretisation, not the exact CVPE of the networks, so that bar is a reference level and not a like-for-like comparison.

    Parameters:
    long_by_tag (dict): Design tag -> summary_by_draw DataFrame (rows: model x draw), or None.
    key (str): Key of SUMMARY_PLOTS.
    filename_stem (str): File name without extension, saved as png in plotting_core.IMAGES_DIR.
    figsize (tuple or None): Figure size in inches. Default None: full text width, plotting_core.print_figsize(9.0).
    gij_by_tag (dict or None): Design tag -> saved Gijbels frame. If given and SHOW_GIJBELS is True, a hatched Gijbels bar is added.
    ecbc_by_tag (dict or None): Design tag -> saved ECBC frame. If given and SHOW_ECBC is True, an ECBC bar is added.
    The matched-family Acar bar (SHOW_ACAR_MATCHED) is read from long_by_tag.

    Returns:
    tuple: (figure, (ymin, ymax)).

    Raises:
    KeyError: If the metric is missing from every saved frame.
    """
    metric, ylabel, subset = SUMMARY_PLOTS[key]
    if not any(df is not None and metric in df.columns for df in long_by_tag.values()):
        raise KeyError(f"column '{metric}' is not in the saved summary_by_draw frames; the figure '{key}' cannot be drawn without a rerun")
    with_gij = SHOW_GIJBELS and bool(gij_by_tag) and metric in GIJ_METRICS
    with_ecbc = SHOW_ECBC and bool(ecbc_by_tag) and metric in GIJ_METRICS
    with_acar = SHOW_ACAR_MATCHED
    layout, ticks = _bar_layout(subset, with_gij=with_gij, with_ecbc=with_ecbc, with_acar=with_acar)
    stats = {tag: _panel_stats(df, metric, layout) for tag, df in long_by_tag.items() if df is not None}
    for tag in list(stats):
        if with_gij:
            stats[tag].update(_gij_stats(gij_by_tag.get(tag), metric))
        if with_ecbc:
            stats[tag].update(_ecbc_stats(ecbc_by_tag.get(tag), metric))
        if with_acar:
            stats[tag].update(_acar_stats(_tag_to_dgp(tag), long_by_tag[tag], metric))
    ylim = summary_limits(stats.values()) if stats else (0, 1)
    fig, axes = plt.subplots(5, 2, figsize=figsize or plotting_core.print_figsize(9.0), sharey=True)
    for r, fam in enumerate(ROW_ORDER):
        for c, strength in enumerate(("weak", "strong")):
            ax = axes[r, c]
            dgp = [d for d in DESIGN_ORDER if FAMILY_LABEL[d] == fam and STRENGTH_LABEL[d] == strength][0]
            st = stats.get(design_tag(dgp))
            if st is None:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#555555")
            else:
                for m, col, x, unfilled in layout:
                    if m not in st:
                        continue
                    mean, ci = st[m]
                    ax.bar(x, mean, 0.85, facecolor=col, edgecolor="black", linewidth=1.2 if unfilled else 0.6,
                           hatch=_REF_HATCH.get(m))
                    ax.errorbar(x, mean, yerr=ci, color="black", capsize=1.5, lw=0.8)
            ax.set_ylim(*ylim)
            ax.axhline(0, color="black", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xticks([t for t, _ in ticks])
            ax.set_xticklabels([l.replace("\n", " ") for _, l in ticks] if r == 4 else [], fontsize=6.5, rotation=40, ha="right", rotation_mode="anchor")
            if c == 1:
                ax.tick_params(axis="y", labelleft=False)
            if r == 0:
                ax.set_title(f"{strength} $\\tau(w)$", fontsize=9, pad=4)
            if c == 0:
                ax.text(0.02, 0.97, ROW_LABELS[fam], transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    fig.supylabel(ylabel + " (mean $\\pm$ 95% CI across draws)", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GREYS[c], edgecolor="black", lw=0.6) for c in GREYS]
    names = list(GREYS)
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", lw=1.2)); names.append("unconditional CODINE")
    extra = ((with_acar, ACAR_COLOR, ACAR_KEY, ACAR_NOTE + ", DGP family (best of 7: mixtures)"),
             (with_gij, GIJ_COLOR, GIJ_KEY, GIJ_NOTE + (", 32-cell" if metric == "excess" else "")),
             (with_ecbc, ECBC_COLOR, ECBC_KEY, ECBC_NOTE + (", 32-cell" if metric == "excess" else "")))
    for on, col, k, label in extra:
        if on:
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="black", lw=0.6, hatch=_REF_HATCH[k])); names.append(label)
    fig.legend(handles, names, title="Conditioning", loc="lower center", ncol=3 if (with_gij or with_ecbc or with_acar) else 4, frameon=False, fontsize=7, title_fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(rect=(0.02, 0.09 if (with_gij or with_ecbc or with_acar) else 0.06, 1, 1), h_pad=0.5, w_pad=0.5)
    fig.savefig(os.path.join(plotting_core.IMAGES_DIR, f"{filename_stem}.png"))
    return fig, ylim


# ── redraw the figures from the saved artifacts (no model is refitted) ─────────────────────────────


# ── LaTeX tables of the synthetic study (written by AAA_summary and by notebook 06) ──────────────────────────────────────
_DESIGN_NAME = {"student": "Student-$t$", "nonmonotonic": "Student-$t$ nonmonotonic", "gumbel": "Gumbel", "clayton": "Clayton",
                "mixed": "Mixture"}


def design_label(tag, strength=True):
    """Row label of a design in the tables: its number, family and, optionally, the strength of the dependence.

    Parameters:
    tag (str): Design tag, for example 'student_weak'.
    strength (bool): Append ', weak' or ', strong'. Default True.

    Returns:
    str: LaTeX text, for example '1 Student-$t$, weak'.
    """
    d = next(x for x in all_designs() if x["tag"] == tag)
    fam, strg = tag.rsplit("_", 1)
    return f"{d['number']} {_DESIGN_NAME[fam]}" + (f", {strg}" if strength else "")


def _variants(S, prefix, contains=None):
    """Names of the models in a summary frame that start with a prefix (and contain a text).

    Parameters:
    S (pd.DataFrame): Summary frame indexed by model name.
    prefix (str): Required start of the name.
    contains (str): Text the name must contain. Default None.

    Returns:
    list: The matching model names.
    """
    return [m for m in S.index if str(m).startswith(prefix) and (contains is None or contains in str(m))]


def synthetic_overview_table(load_artifact, filename="synthetic_overview_combined.tex", label="tab:synthetic_overview"):
    """Table of the ten synthetic designs: excess CVPE and tau(w) RMSE of each model group, lowest per row in bold.

    Groups: unconditional CODINE, C-CODINE, C-CopFlow with affine and with spline coupling, and the two nonparametric benchmarks, the
    kernel estimator of Gijbels et al. (2011) and the ECBC of Lu and Ghosh (2024). For each network group the best (lowest) variant is
    shown, so the variant can differ between columns. The CVPE of the two benchmarks is computed from 32 x 32 cell masses and not by
    quadrature of a density, so it is marked with a dagger and is only approximately on the scale of the other columns.

    Parameters:
    load_artifact (callable): run_support.load_artifact.
    filename (str): File name in the table folder. Default 'synthetic_overview_combined.tex'.
    label (str): LaTeX label.

    Returns:
    str or None: The LaTeX text, or None if no design has a saved summary.
    """
    rt = reporting_tables
    groups = [("Unconditional: CODINE", None), ("C-CODINE", None), ("C-CopFlow", "affine)"), ("C-CopFlow", "spline coupling)")]

    def bench_mean(frame, col):
        """Mean over the draws of one column of a saved benchmark frame, NaN if the frame or the column is missing.

        Parameters:
        frame (pd.DataFrame): Benchmark frame, or None.
        col (str): Column name.

        Returns:
        float: The mean, or NaN.
        """
        return float(frame[col].mean()) if frame is not None and len(frame) and col in frame.columns else np.nan

    rows, data = {}, []
    cols = ["Uncond-CODINE", "C-CODINE", "C-CopFlow-affine", "C-CopFlow-spline", "Kernel", "ECBC"]
    n_draws = None
    for d in all_designs():
        art = load_artifact(d["artifact"])
        if not art or art.get("summary") is None:
            continue
        S = art["summary"]
        ex, tau = [], []
        for prefix, contains in groups:
            mods = _variants(S, prefix, contains)
            ex.append(float(S.loc[mods, "excess"].min()) * 1e3 if mods else np.nan)
            tau.append(float(S.loc[mods, "tau_rmse"].min()) if mods else np.nan)
        g, e = art.get("gijbels"), art.get("ecbc")
        ex += [bench_mean(g, "excess") * 1e3, bench_mean(e, "excess") * 1e3]
        tau += [bench_mean(g, "tau_rmse"), bench_mean(e, "tau_rmse")]
        rows[d["tag"]] = (ex, tau)
        n_draws = int(art["mc_reps"]) if "mc_reps" in art else n_draws
    if not rows:
        return None
    n_words = {10: "ten"}.get(n_draws, str(n_draws))
    head = (r' & \multicolumn{6}{c}{Excess CVPE $\times10^{3}$} & \multicolumn{6}{c}{$\tau(w)$ RMSE} \\' '\n'
            r'\cmidrule(lr){2-7}\cmidrule(lr){8-13}' '\n'
            r'Design & \shortstack[r]{Uncond.\\CODINE} & \shortstack[r]{C-\\CODINE} & \shortstack[r]{CopFlow\\affine} & '
            r'\shortstack[r]{CopFlow\\spline} & \shortstack[r]{Kernel\\$^{\dagger}$} & \shortstack[r]{ECBC\\$^{\dagger}$} & '
            r'\shortstack[r]{Uncond.\\CODINE} & \shortstack[r]{C-\\CODINE} & '
            r'\shortstack[r]{CopFlow\\affine} & \shortstack[r]{CopFlow\\spline} & Kernel & ECBC \\')
    short = {"nonmonotonic": "Student-$t$ nonmon."}
    lines = []
    for block, name in (("weak", "Weak dependence"), ("strong", "Strong dependence")):
        if lines:
            lines.append(r'\midrule')
        lines.append(rf'\multicolumn{{13}}{{l}}{{\emph{{{name}}}}} \\')
        for d in all_designs():
            tag = d["tag"]
            if not tag.endswith(block) or tag not in rows:
                continue
            ex, tau = rows[tag]
            fam = tag.rsplit("_", 1)[0]
            lab = f"{d['number']} " + short.get(fam, _DESIGN_NAME[fam])
            cells = rt.tex_bold_min(ex, lambda x: rt.tex_num(x, 2)) + rt.tex_bold_min(tau, lambda x: rt.tex_num(x, 3))
            lines.append(lab + " & " + " & ".join(cells) + r' \\')
            data.append(dict(zip(["design"] + [f"excess_cvpe_{g}" for g in cols] + [f"tau_rmse_{g}" for g in cols], [d["tag"]] + list(ex) + list(tau))))
    n_tex = f"{N_TRAIN // 1000}{{,}}{N_TRAIN % 1000:03d}"
    cap = (rf'Error of each model group on the ten synthetic designs ($n={n_tex}$, mean over {n_words} draws; for each network '
           r'group the lowest value over its variants is shown; lowest per row and measure in bold). Left: excess CVPE over the '
           r"oracle, times $10^{3}$. Right: RMSE of the fitted Kendall's $\tau(w)$ on 15 values of $w$. Kernel: estimator of "
           r'\citet{gijbels2011conditional}; ECBC: estimator of \citet{lu2024nonparametric}. $^{\dagger}$The CVPE of the kernel '
           r'estimator and the ECBC is computed from $32\times32$ cell masses, that of the other models by quadrature of the density, '
           r'so the two are only approximately on the same scale (Subsection~\ref{subsubsec:resultsgijbels}).')
    return rt.write_raw_table(filename, cap, label, "l" + "r" * 12, head, "\n".join(lines), data=pd.DataFrame(data),
                              size=r'\footnotesize\setlength{\tabcolsep}{2pt}')


def networks_vs_gijbels_table(load_artifact, filename="networks_vs_gijbels.tex", label="tab:networks_vs_gijbels"):
    """Table of the best network of each family against the kernel benchmark: tau(w) RMSE and excess Copula Energy Score.

    Parameters:
    load_artifact (callable): run_support.load_artifact.
    filename (str): File name in the table folder. Default 'networks_vs_gijbels.tex'.
    label (str): LaTeX label.

    Returns:
    str or None: The LaTeX text, or None if no design has a saved Gijbels frame.
    """
    rt = reporting_tables
    lines, data = [], []
    n_draws = None
    for d in all_designs():
        art = load_artifact(d["artifact"])
        g = None if not art else art.get("gijbels")
        if g is None or len(g) == 0:
            continue
        S = art["summary"]
        n_draws = int(art.get("mc_reps", len(g)))
        tau, ces = [], []
        for prefix in ("C-CODINE", "C-CopFlow"):
            mods = _variants(S, prefix)
            tau.append(float(S.loc[mods, "tau_rmse"].min()) if mods else np.nan)
            ces.append(float(S.loc[mods, "excess_ces"].min()) * 1e3 if mods else np.nan)
        tau.append(float(g["tau_rmse"].mean()))
        ces.append(float(g["excess_ces"].mean()) * 1e3)
        cells = rt.tex_bold_min(tau, lambda x: rt.tex_num(x, 3)) + rt.tex_bold_min(ces, lambda x: rt.tex_num(x, 2))
        lines.append(design_label(d["tag"]) + " & " + " & ".join(cells) + r' \\')
        data.append(dict(zip(["design", "tau_rmse_C-CODINE", "tau_rmse_C-CopFlow", "tau_rmse_Kernel",
                              "excess_ces_x1e3_C-CODINE", "excess_ces_x1e3_C-CopFlow", "excess_ces_x1e3_Kernel"], [d["tag"]] + tau + ces)))
    if not lines:
        return None
    n_words = {10: "ten"}.get(n_draws, str(n_draws))
    head = (r' & \multicolumn{3}{c}{$\tau(w)$ RMSE} & \multicolumn{3}{c}{Excess CES $\times10^{3}$} \\' '\n'
            r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}' '\n'
            r'Design & C-CODINE & C-CopFlow & Kernel & C-CODINE & C-CopFlow & Kernel \\')
    cap = (r'Best network of each family and the kernel estimator of \citet{gijbels2011conditional} on the ten synthetic designs '
           rf'($n={N_TRAIN // 1000}{{,}}{N_TRAIN % 1000:03d}$, mean over {n_words} draws; lowest value of each row block in bold). $\tau(w)$ RMSE: RMSE of the fitted '
           r"Kendall's $\tau(w)$ on 15 values of $w$. Excess CES: Copula Energy Score minus that of the true copula, times "
           r'$10^{3}$, computed on the same $64\times64$ quadrature for all estimators (raw score, so margins that are not exactly '
           r'uniform are penalised). For each family the lowest value over its variants is shown, so the variant can differ between '
           r"columns. Excess CVPE is not shown here; Table~\ref{tab:synthetic_overview} gives it, with the benchmark's CVPE, computed on a "
           r'coarser grid, marked as only approximately comparable (Subsection~\ref{subsubsec:resultsgijbels}).')
    return rt.write_raw_table(filename, cap, label, "lrrrrrr", head, "\n".join(lines), data=pd.DataFrame(data),
                              size=r'\footnotesize', short_caption="Networks and the kernel benchmark")


_FAMILY_ORDER = ("CODINE", "C-CODINE", "C-CopFlow")


def _uniformity_frame(ks):
    """Rows of the stored-network uniformity frame that were checked, with the design strength and a readable design label.

    Parameters:
    ks (pd.DataFrame): Uniformity frame with the columns 'n_tests' and 'design'.

    Returns:
    pd.DataFrame: The checked rows with the added columns 'strength' and 'label'.
    """
    k = ks[ks.n_tests > 0].copy()
    k["strength"] = k.design.str.rsplit("_", n=1).str[1]
    k["label"] = k.design.map(design_label)
    return k


def synthetic_uniformity_table(ks, filename="synthetic_uniformity.tex", label="tab:synthetic_uniformity"):
    """Table of the marginal-uniformity screen over the stored synthetic networks, by network family.

    Parameters:
    ks (pd.DataFrame): The frame KS_SYN of AAA_summary (columns design, rep, model, family, d_max, n_tests, passed).
    filename (str): File name in the table folder. Default 'synthetic_uniformity.tex'.
    label (str): LaTeX label.

    Returns:
    str: The LaTeX text.
    """
    k = _uniformity_frame(ks)

    def frac(s):
        """'passing/total (share%)' of a sub-frame.

        Parameters:
        s (pd.DataFrame): Sub-frame with the boolean column 'passed'.

        Returns:
        str: LaTeX text, or '--' if the frame is empty.
        """
        return rf"{int(s.passed.sum())}/{len(s)} ({100 * s.passed.mean():.0f}\%)" if len(s) else "--"

    lines, data = [], []
    for fam in _FAMILY_ORDER:
        s = k[k.family == fam]
        if s.empty:
            continue
        lines.append(" & ".join([fam, frac(s[s.strength == "weak"]), frac(s[s.strength == "strong"]), frac(s),
                                 f"{s.d_max.median():.3f}", f"{s.d_max.max():.3f}"]) + r' \\')
        w, st = s[s.strength == "weak"], s[s.strength == "strong"]
        data.append(dict(family=fam, networks_weak=len(w), passing_weak=int(w.passed.sum()), networks_strong=len(st),
                         passing_strong=int(st.passed.sum()), networks_all=len(s), passing_all=int(s.passed.sum()),
                         median_dmax=float(s.d_max.median()), largest_dmax=float(s.d_max.max())))
    head = r'Family & Weak & Strong & All & Median $D_{\max}$ & Largest $D_{\max}$ \\'
    cap = (r'Marginal-uniformity screen applied to the networks of the synthetic study, whose true margins are exactly uniform. '
           r'A network passes if its largest KS distance $D_{\max}$ is at most the threshold of the lake screen '
           r'(Subsection~\ref{subsubsec:marginaluniformitychecks}). Entries are passing networks out of all networks (one per draw and '
           r'variant, ten draws) in the five weak designs, the five strong designs and all ten. CODINE comprises the unconditional '
           r'network and the two halves of the split network. Median and largest $D_{\max}$ are over all networks of the family.')
    return reporting_tables.write_raw_table(filename, cap, label, "lrrrrr", head, "\n".join(lines), data=pd.DataFrame(data),
                                            size=r'\footnotesize\setlength{\tabcolsep}{4pt}')


def synthetic_uniformity_design_table(ks, filename="synthetic_uniformity_design.tex", label="tab:synthetic_uniformity_design"):
    """Long table of the marginal-uniformity screen for every design and network family.

    Parameters:
    ks (pd.DataFrame): The frame KS_SYN of AAA_summary.
    filename (str): File name in the table folder. Default 'synthetic_uniformity_design.tex'.
    label (str): LaTeX label.

    Returns:
    str: The LaTeX text.
    """
    k = _uniformity_frame(ks)
    order = {d["tag"]: d["number"] for d in all_designs()}
    lines, data = [], []
    for tag in sorted(k.design.unique(), key=order.get):
        for fam in _FAMILY_ORDER:
            s = k[(k.design == tag) & (k.family == fam)]
            if s.empty:
                continue
            lines.append(" & ".join([design_label(tag), fam, str(len(s)), str(int(s.passed.sum())),
                                     f"{s.d_max.median():.3f}", f"{s.d_max.max():.3f}"]) + r' \\')
            data.append(dict(design=tag, family=fam, networks=len(s), passing=int(s.passed.sum()),
                             median_dmax=float(s.d_max.median()), largest_dmax=float(s.d_max.max())))
    head = r'Design & Family & Networks & Passing & Median $D_{\max}$ & Largest $D_{\max}$ \\'
    cap = (r'Marginal-uniformity screen for the networks of each synthetic design (ten draws, seed 0): number of networks, number '
           r'passing, and median and largest $D_{\max}$. CODINE comprises the unconditional network and the two halves of the split '
           r'network. The screen and its threshold are those of Table~\ref{tab:synthetic_uniformity}.')
    return reporting_tables.write_raw_table(filename, cap, label, "llrrrr", head, "\n".join(lines), data=pd.DataFrame(data),
                                            size=r'\footnotesize\setlength{\tabcolsep}{4pt}',
                                            short_caption="Marginal-uniformity screen by design", longtable=True)


_METRIC_TEX = {"excess CVPE": "Excess CVPE", "tau(w) RMSE": r"$\tau(w)$ RMSE",
               "lower-tail RMSE": rf"Lower-tail RMSE, $\chi({Q_TAIL:g})$", "upper-tail RMSE": rf"Upper-tail RMSE, $\chi({Q_TAIL:g})$",
               "excess CES (raw)": "Excess CES (raw)"}


def synthetic_uniformity_accuracy_table(ks_acc, cells, filename="synthetic_uniformity_accuracy.tex",
                                        label="tab:synthetic_uniformity_accuracy"):
    """Table of the within-design rank correlation between a network's largest KS distance and its error.

    Parameters:
    ks_acc (pd.DataFrame): The frame KS_ACC of AAA_summary (columns family, metric, runs, failing, spearman_within, p_value).
    cells (dict): Number of (design, network variant) cells per family, for the caption, for example {'C-CODINE': 30, 'C-CopFlow': 60}.
    filename (str): File name in the table folder. Default 'synthetic_uniformity_accuracy.tex'.
    label (str): LaTeX label.

    Returns:
    str: The LaTeX text.
    """
    rt = reporting_tables
    lines = []
    for fam in ("C-CODINE", "C-CopFlow"):
        for r in ks_acc[ks_acc.family == fam].itertuples():
            lines.append(" & ".join([fam, _METRIC_TEX.get(r.metric, rt._esc_latex(r.metric)), str(int(r.runs)), str(int(r.failing)),
                                     rt.tex_num(r.spearman_within, 2), rt.tex_p(r.p_value)]) + r' \\')
    head = r'Family & Error measure & Networks & Failing & Spearman & $p$-value \\'
    cap = (r'Marginal uniformity and accuracy of the conditional networks on the synthetic designs. For each error measure, the '
           r'Spearman rank correlation between the largest KS distance $D_{\max}$ of a network and its error, after subtracting the '
           rf'mean of its design and network variant (ten draws each, {cells.get("C-CODINE", "--")} cells for C-CODINE and '
           rf'{cells.get("C-CopFlow", "--")} for C-CopFlow). A positive value means that a larger $D_{{\max}}$ goes with a larger error. '
           r'Failing: number of networks above the threshold of the screen (Table~\ref{tab:synthetic_uniformity}). The $p$-values are '
           r'not corrected for the ten comparisons, and the correlations do not establish that non-uniform margins cause the errors.')
    return rt.write_raw_table(filename, cap, label, "llrrrr", head, "\n".join(lines),
                              data=ks_acc[ks_acc.family.isin(["C-CODINE", "C-CopFlow"])])
