"""Every recorded figure can be redrawn from its stored call alone (fake data, temporary artifact folder)."""
import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix='figreplay_')
os.environ['THESIS_ARTIFACT_DIR'] = os.path.join(_TMP, 'artifacts')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['THESIS_ALLOW_UNCALIBRATED_IMPORT'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figure_recipes as FR  # noqa: E402
import figure_replay  # noqa: E402
import plotting_core  # noqa: E402

OUT = os.path.join(_TMP, 'Images test')
os.makedirs(OUT, exist_ok=True)
plotting_core.IMAGES_DIR = OUT
rng = np.random.default_rng(0)


def _grid():
    """Fake empirical tail grid for the figure tests.

    Returns:
    pd.DataFrame: Rows for every regime, tail and threshold.
    """
    q = np.array([0.8, 0.85, 0.9, 0.95, 0.975])
    rows = []
    for g in ('all', 'low', 'high'):
        for t in ('upper', 'lower'):
            c = 0.2 + 0.1 * rng.random(len(q))
            rows.append(pd.DataFrame(dict(group=g, tail=t, q=q, chi=c, chi_lo=c - .05, chi_hi=c + .05, independence=1 - q,
                                          chi_fixed_lo=c - .04, chi_fixed_hi=c + .04, chi_fixed_denominator=c,
                                          n_joint=rng.integers(1, 30, len(q)), expected_joint_indep=rng.random(len(q)) * 10)))
    return pd.concat(rows, ignore_index=True)


def _redraw(key, name):
    """Redraw a recorded figure from its stored call and check that the PNG is written.

    Parameters:
    key (str): Figure key.
    name (str): Output file stem.
    """
    assert key in figure_replay.list_figures(), (key, figure_replay.list_figures())
    p = os.path.join(OUT, name + '.png')
    if os.path.exists(p):
        os.remove(p)
    figure_replay.replay(key, images_dir=OUT)
    assert os.path.getsize(os.path.join(OUT, name + '.png')) > 1000


def test_chi_grid_empirical_and_preview():
    """The empirical chi figure can be redrawn from the stored call."""
    G = _grid()
    FR.chi_grid_empirical(G)
    _redraw('chi_grid_empirical', 'chi_grid_empirical')


def test_penalty_calibration():
    """The penalty calibration figure can be redrawn from the stored call."""
    lam = [0, 1, 10]
    idx = pd.MultiIndex.from_product([['C-CODINE', 'C-CopFlow'], lam])
    G = pd.DataFrame(dict(mean=rng.random(6), se=rng.random(6) * .1), index=idx)
    best = pd.DataFrame(dict(family=['C-CODINE'] * 5 + ['C-CopFlow'] * 5, best_lam=[0, 1, 1, 10, 10] * 2))
    FR.penalty_calibration(G, best, lam, {'C-CODINE': 1, 'C-CopFlow': 10})
    _redraw('penalty_calibration', 'penalty_calibration')


def test_mcs_survivors():
    """The MCS survivor figure can be redrawn from the stored call."""
    x = np.linspace(-2, 2, 30)
    curves = {'gaussian': dict(x=x, tau=0.1 * x, tau_lo=0.1 * x - .05, tau_hi=0.1 * x + .05)}
    wg = np.linspace(-1.8, 1.8, 50)
    FR.mcs_survivors(curves, {'C-CopFlow (FiLM, affine)': 0.08 * wg}, wg, {'gaussian': 'Gaussian'})
    _redraw('mcs_survivors', 'mcs_survivors')


def test_spatial_partitions():
    """The partition map can be redrawn from the stored call."""
    lon, lat = rng.random(200), rng.random(200)
    parts = {9: [np.arange(200) % 9 == j for j in range(9)], 36: [np.arange(200) % 36 == j for j in range(36)]}
    FR.spatial_partitions(parts, lon, lat)
    _redraw('spatial_partitions', 'spatial_partitions')


def test_axes_arguments_are_not_recorded():
    """A call that receives a matplotlib Axes is not recorded."""
    import matplotlib.pyplot as plt
    called = []

    @figure_replay.recorded
    def draw_on(ax):
        """Drawing function that takes an Axes.

        Parameters:
        ax (matplotlib.axes.Axes): Axes to draw on.
        """
        called.append(1)
    fig, ax = plt.subplots()
    draw_on(ax)
    assert called and 'draw_on' not in figure_replay.list_figures()
    plt.close(fig)
