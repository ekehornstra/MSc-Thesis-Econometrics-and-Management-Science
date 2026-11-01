"""data_split.py

Chronological data-splitting and block-resampling utilities shared across every
notebook, matching the evaluation procedure described in Sections 4.8 and 4.9
of the thesis.

- ``stationary_bootstrap_indices`` / ``block_length_joint`` implement the
  stationary block bootstrap (Politis & Romano, 1994) used for confidence
  intervals on Kendall's tau and tail dependence, with the
  shared block length selected via the spectral-density plug-in method of
  Politis and White (2004), corrected by Patton, Politis, and White (2009).
- ``train_test_split_chrono`` / ``purged_chrono_folds`` implement the purged
  chronological train/test split used for the cross-validated prediction
  error, removing a leakage-protection gap L from the split
  boundary to reduce temporal dependence between training and test windows.

The leakage gap L (max of the Politis-White block length and the longest AR lag, computed in NB00) is deliberately separate
from the stationary-bootstrap mean block length b, which controls inference.

References:
Patton, A., Politis, D. N., & White, H. (2009). Correction to "Automatic block-length selection for the dependent
bootstrap" by D. Politis and H. White. Econometric Reviews, 28(4), 372-375.
Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical Association,
89(428), 1303-1313.
Politis, D. N., & White, H. (2004). Automatic block-length selection for the dependent bootstrap. Econometric
Reviews, 23(1), 53-70.
"""

import numpy as np
from arch.bootstrap import optimal_block_length
from run_support import get_logger
log = get_logger()


def stationary_bootstrap_indices(n, b, rng):
    """Politis and Romano (1994) stationary bootstrap index sequence.

    Draws circular blocks of GEOMETRIC(1/b) length (mean length = b) rather
    than a fixed block length, so no single block-boundary choice can bias
    the resample. Wraps around the end of the series (circular) rather than
    truncating the last block, which is the standard treatment and keeps
    every block the same distribution regardless of where it starts.

    Vectorized per-run (not per-index) -- cost is O(n/b) python-level steps,
    not O(n), so this stays cheap even for n_boot in the hundreds.

    Source: Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical
    Association, 89(428), 1303-1313.

    Parameters:
    n (int): Length of the series to resample.
    b (float): Mean block length for this stationary bootstrap. This is
        not the outer cross-validation purge length L.
    rng (np.random.Generator): Pass one per bootstrap call for reproducible,
        independent replicates -- do not share a single rng across a
        vectorized batch of replicates without accounting for draw order.

    Returns:
    np.ndarray: (n,) int array of resampled positions into the original series.
    """
    p = 1.0 / b
    idx = np.empty(n, dtype=int)
    pos = 0
    while pos < n:
        start = rng.integers(0, n)
        run_len = min(int(rng.geometric(p)), n - pos)
        idx[pos:pos + run_len] = (start + np.arange(run_len)) % n
        pos += run_len
    return idx


def regime_stationary_bootstrap_indices(n_full, regime_mask, b, rng):
    """Stationary bootstrap of a regime-masked, non-contiguous subsequence,
    respecting real time-adjacency.

    Draws a Politis and Romano (1994) stationary-bootstrap resample of the
    FULL chronological index (length n_full, via stationary_bootstrap_indices
    above), then keeps only the resampled positions whose original,
    pre-resample position is a regime member.

    This is not the same thing as resampling the regime-restricted
    subsequence directly (block=False / i.i.d. treatment of
    u1[regime_mask], u2[regime_mask]). A block drawn here is a genuine
    contiguous run of real time, so a block spanning e.g. several hours can
    contain both regime and non-regime timesteps, and only the regime ones
    survive into the returned index -- this is what lets the resample capture
    real within-regime temporal clustering (e.g. a multi-hour high-divergence
    spell), which treating regime membership as already scattering
    observations independently through time cannot.

    Parameters:
    n_full (int): Length of the ORIGINAL (unmasked) chronological series.
    regime_mask (np.ndarray): Boolean, length n_full, True for regime
        members.
    b (float): Mean block length for the underlying stationary bootstrap
        -- use the inference b, distinct from the leakage purge L, in
        this pipeline.
    rng (np.random.Generator): Per-replicate RNG, same contract as
        stationary_bootstrap_indices.

    Returns:
    np.ndarray: Indices into the ORIGINAL n_full series (not into a
        regime-restricted subarray), all regime members. Length varies
        replicate to replicate depending on how much of each drawn block
        happened to land in-regime -- close to regime_mask.sum() in
        expectation but not fixed to it exactly. That variability is the
        correct behaviour for a block bootstrap of a masked subsequence,
        not something to pad or truncate away.
    """
    idx_full = stationary_bootstrap_indices(n_full, b, rng)
    return idx_full[regime_mask[idx_full]]


def block_length_joint(*series, variant="stationary"):
    """Largest plug-in optimal block length across the pooled series.

    Selected via the spectral-density plug-in method of Politis and White
    (2004), with the correction of Patton, Politis, and White (2009), as
    implemented by arch.bootstrap.optimal_block_length. Both the level series and their squared demeaned values
    (volatility) are checked; the returned block length is the larger of
    the two across all series, so it is long enough to capture the
    slowest-decaying dependence among all series and both moments.

    Source: Politis, D. N., & White, H. (2004). Automatic block-length selection for the dependent bootstrap.
    Econometric Reviews, 23(1), 53-70. Patton, A., Politis, D. N., & White, H. (2009). Correction to "Automatic
    block-length selection for the dependent bootstrap" by D. Politis and H. White. Econometric Reviews, 28(4),
    372-375.

    Parameters:
    *series (np.ndarray): One or more equal-length series to pool (e.g.
        precipitation, wind, divergence); all three are pooled
        so one shared block length applies everywhere.
    variant (str): Which optimal_block_length estimate to use ('stationary'
        or 'circular'). Default 'stationary'.

    Returns:
    int: The shared block length b, floored at 2.
    """
    x = np.column_stack(series)
    x2 = np.column_stack([(s - np.mean(s))**2 for s in series])
    res_level = optimal_block_length(x)
    res_vol   = optimal_block_length(x2)
    b = int(np.ceil(max(res_level[variant].max(), res_vol[variant].max())))
    return max(b, 2)


def contiguous_runs(mask):
    """List of index arrays, one per maximal run of consecutive True positions in `mask`.

    Parameters:
    mask (array-like): Boolean mask over time-ordered rows.

    Returns:
    list: One integer index array per maximal run of consecutive True positions (empty list if none).
    """
    pos = np.flatnonzero(np.asarray(mask, bool))
    if not len(pos):
        return []
    breaks = np.flatnonzero(np.diff(pos) > 1) + 1
    return np.split(pos, breaks)


def rank_to_unit(x):
    """Ranks of x scaled to (0, 1): rank / (n + 1). Ties are broken by order (no ties expected).

    Parameters:
    x (array-like): Values to rank.

    Returns:
    np.ndarray: Ranks divided by (n + 1), same length as x.
    """
    x = np.asarray(x, float)
    return (np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort") + 1.0) / (len(x) + 1.0)


def pw_block_length_segments(series, mask, min_len, variant="stationary"):
    """Politis-White block length of the pooled rank-transformed series over the rows in `mask`.

    DIAGNOSTIC: the length is computed on the rank-transformed (pseudo-observation scale)
    series, separately inside every contiguous run of `mask` with at least `min_len` rows (`min_len` is the longest
    retained AR lag of precipitation and wind, no default; so a training
    set made of two pieces, as in fold B1, is never glued together), and the largest value over the runs is
    returned. It is reported next to the leakage length L and is used only for descriptive bootstrap
    intervals; it does not drive any purge or cross-validation.

    Source: Politis, D. N., & White, H. (2004), Econometric Reviews, 23(1), 53-70, via block_length_joint.

    Parameters:
    series (list): Full-length arrays (e.g. precipitation residual, wind residual, standardised divergence).
    mask (np.ndarray): Boolean mask of the rows to use.
    min_len (int): Minimum run length; shorter contiguous runs are skipped.
    variant (str): 'stationary' (mean block length of the stationary bootstrap) or 'circular' (fixed block length of the
        circular bootstrap, larger by about (3/2)^(1/3)). Default 'stationary'.

    Returns:
    int: Block length, at least 2.
    """
    mask = np.asarray(mask, bool)
    ranked = [np.full(len(mask), np.nan) for _ in series]
    for k, x in enumerate(series):
        ranked[k][mask] = rank_to_unit(np.asarray(x, float)[mask])
    best = 2
    for run in contiguous_runs(mask):
        if len(run) >= min_len:
            best = max(best, block_length_joint(*[r[run] for r in ranked], variant=variant))
    return int(best)


VAL_FRAC = 0.25   # share of a fold's training rows used as the early-stopping validation set


def train_test_split_chrono(n, test_frac=0.2, purge=0, clamp=True, one_sided=False):
    """Chronological train/test split with a purge gap removed from the boundary.

    Splits the first (1 - test_frac) fraction of indices into train and the
    remainder into test, then removes a gap of length `purge` from the
    boundary, split evenly on either side, so that observations immediately
    adjacent to the split point -- still temporally dependent under the
    residual autocorrelation -- are excluded
    from both sets rather than leaking across it. This is the CVPE split. The purge length L is not required
    to equal the stationary-bootstrap mean block length b.

    Parameters:
    n (int): Length of the series to split.
    test_frac (float): Fraction of the series assigned to the test set.
        Default 0.2.
    purge (int): Total gap length removed from the boundary, split evenly
        on either side. Default 0.
    clamp (bool): If True (default), silently reduce an oversized purge to
        the largest value the split can support (leaving at least 20% of
        the smaller side as actual train/test data) and log a warning,
        instead of raising. Set False to raise a ValueError instead.
    one_sided (bool): If True the whole gap is taken from the TRAIN side, so the test (validation) set keeps
        every row after the boundary. Used for the early-stopping validation split: the validation rows are
        then the full last `test_frac` of the data. Default False (gap split evenly, as for the outer folds).

    Returns:
    tuple: (train_mask, test_mask), each a boolean array of length n.
    """
    n_train = int(np.floor(n * (1 - test_frac)))
    n_test = n - n_train
    max_safe_purge = int(0.8 * min(n_train, n_test))

    if purge > max_safe_purge:
        if not clamp:
            raise ValueError(
                f"purge={purge} too large relative to n={n} and test_frac={test_frac}; "
                f"max safe purge is {max_safe_purge}"
            )
        log.info(f"WARNING: purge={purge} too large for n={n}, test_frac={test_frac} "
              f"— clamped to {max_safe_purge}")
        purge = max_safe_purge

    half_gap_before = purge if one_sided else purge // 2
    half_gap_after = 0 if one_sided else purge - half_gap_before
    train_end = max(n_train - half_gap_before, 1)
    test_start = min(n_train + half_gap_after, n - 1)

    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[:train_end] = True
    test_mask[test_start:] = True
    return train_mask, test_mask

def purged_chrono_folds(n, n_splits=5, purge=0):
    """Purged/blocked K-fold CV over chronological blocks.

    Splits the series into n_splits contiguous chronological blocks. For
    each fold, the held-out block is the test set and every other index is
    the training set, except for a purge gap of length `purge` removed
    from each side of the held-out block's boundary. Unlike a
    rolling-origin/walk-forward split, each fold's training set includes
    both the data before AND after its held-out block -- appropriate here
    because the pseudo-observations are treated as approximately
    stationary post-whitening, not as a forecasting
    problem where only the past may be used.

    Parameters:
    n (int): Length of the series to split.
    n_splits (int): Number of folds. Default 5.
    purge (int): Gap length removed from each side of the held-out block's
        boundary. Default 0.

    Returns:
    generator: Yields (train_mask, test_mask) for each fold, each a boolean array of length n.
    """
    edges = np.linspace(0, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        lo, hi = edges[k], edges[k + 1]
        test_mask = np.zeros(n, dtype=bool)
        test_mask[lo:hi] = True
        train_mask = ~test_mask
        train_mask[max(lo - purge, 0):min(hi + purge, n)] = False
        yield train_mask, test_mask

# ─────────────────────────────────────────────────────────────────────────────
# Four purged blocked folds (cross-validation design)
# ─────────────────────────────────────────────────────────────────────────────

def make_blocked_folds(n, purge):
    """Four purged blocked out-of-time folds over positions 0..n-1.

    Quarter edges e = linspace(0, n, 5). The purge is split into
    floor(purge / 2) steps before and ceil(purge / 2) steps after each
    boundary. This retains the full requested gap for odd purge lengths.

    Folds:
      A1: train first half,  test second half
      A2: train second half, test first half
      B1: train outer quarters (two segments), test middle half
      B2: train middle half, test outer quarters (two segments)

    Parameters:
    n (int): Length of the usable series (common_idx), e.g. 2884.
    purge (int): Total gap at each boundary, split evenly on both sides. REQUIRED (no default): the caller passes the
        leakage length L computed in NB00 (``purge_final``: max of the Politis-White b and the longest AR lag, raised until
        every fold's own b is at most L), so that the value is derived from the data and never hardcoded here.

    Returns:
    list: Four dicts with keys 'name', 'train', 'test' (boolean arrays of
        length n). B1's train set and B2's test set are non-contiguous.
    """
    e = np.linspace(0, n, 5).astype(int)
    gp_before = purge // 2
    gp_after = purge - gp_before
    pos = np.arange(n)
    H1 = pos < e[2] - gp_before
    H2 = pos >= e[2] + gp_after
    mid = (pos >= e[1] + gp_after) & (pos < e[3] - gp_before)
    outer = (pos < e[1] - gp_before) | (pos >= e[3] + gp_after)
    return [dict(name="A1", train=H1, test=H2),
            dict(name="A2", train=H2, test=H1),
            dict(name="B1", train=outer, test=mid),
            dict(name="B2", train=mid, test=outer)]


def check_blocked_folds(folds, n, purge, expected_test_sizes=None):
    """Assert the fold-design invariants (disjoint train/test, purge gaps, coverage) and return coverage counts.

    For every fold: train/test disjoint; minimum absolute distance between
    any train and any test position >= purge; optional test size match. Over
    folds: the union of test masks covers every position.

    Parameters:
    folds (list): Output of make_blocked_folds.
    n (int): Series length.
    purge (int): Required minimum train-test distance (the same L that was passed to make_blocked_folds).
    expected_test_sizes (dict): Optional {fold_name: size}.

    Returns:
    dict: {'min_gap': {name: int}, 'n_train', 'n_test', 'times_tested':
        np.ndarray (n,) count of folds testing each position}.
    """
    out = {'min_gap': {}, 'n_train': {}, 'n_test': {}}
    times = np.zeros(n, dtype=int)
    for f in folds:
        tr, te = f['train'], f['test']
        assert tr.shape == (n,) and te.shape == (n,), f"{f['name']}: wrong mask length"
        assert not (tr & te).any(), f"{f['name']}: train and test overlap"
        tr_pos, te_pos = np.flatnonzero(tr), np.flatnonzero(te)
        # nearest train position for each test position (sorted arrays)
        j = np.searchsorted(tr_pos, te_pos)
        left = np.where(j > 0, te_pos - tr_pos[np.maximum(j - 1, 0)], np.iinfo(int).max)
        right = np.where(j < len(tr_pos), tr_pos[np.minimum(j, len(tr_pos) - 1)] - te_pos,
                         np.iinfo(int).max)
        gap = int(np.minimum(left, right).min())
        assert gap >= purge, f"{f['name']}: min train-test distance {gap} < purge {purge}"
        if expected_test_sizes is not None and f['name'] in expected_test_sizes:
            assert int(te.sum()) == expected_test_sizes[f['name']], \
                f"{f['name']}: test size {int(te.sum())} != {expected_test_sizes[f['name']]}"
        out['min_gap'][f['name']] = gap
        out['n_train'][f['name']] = int(tr.sum())
        out['n_test'][f['name']] = int(te.sum())
        times += te.astype(int)
    assert (times >= 1).all(), f"{int((times == 0).sum())} positions never tested"
    out['times_tested'] = times
    return out


def fold_val_purge(n_fit_rows, max_selected_ar_lag, val_frac=VAL_FRAC):
    """One-sided gap (steps) between fit rows and validation rows for the neural early-stopping split:
    the largest selected AR lag, reduced only if train_test_split_chrono would clamp it. The gap is taken from the fit side,
    so the validation set is the full last `val_frac` of a fold's training rows (for example 25% of a fold's training rows;
    the counts depend on n and on the leakage length L, see notebook 00).

    Parameters:
    n_fit_rows (int): Number of training rows of the fold.
    max_selected_ar_lag (int): Largest selected AR lag; must be a positive integer.
    val_frac (float): Share of the training rows used for validation. Default VAL_FRAC.

    Returns:
    int: The gap in steps.
    """
    n_tr = int(np.floor(n_fit_rows * (1 - val_frac)))
    n_va = n_fit_rows - n_tr
    lag = int(max_selected_ar_lag)
    if lag < 1:
        raise ValueError("max_selected_ar_lag must be a positive integer")
    gap = int(min(lag, 0.8 * min(n_tr, n_va)))
    if gap < lag:
        raise ValueError(f"validation gap {gap} < selected AR lag {lag}; insufficient training rows")
    return gap


def train_estimation_order(train_mask):
    """Order in which a fold's training rows are concatenated for AR / GARCH
    estimation.

    A contiguous training set (A1, A2, B2) keeps its time order. When the
    training set is two pieces, one at the very start and one at the very
    end of the series (B1: outer quarters), the pieces are joined END piece
    first, START piece second. The record spans about one calendar year, so
    the end of the record (e.g. June-August) is seasonally followed by its
    start (e.g. September-November); the time-ordered join would instead
    glue November to June, a half-year seasonal jump at the junction.

    Only the estimation sees this order. Residuals are always computed on the
    real, time-ordered full series with the estimated parameters fixed.

    Parameters:
    train_mask (np.ndarray): Boolean training mask over time-ordered rows.

    Returns:
    np.ndarray: Integer positions of the training rows in estimation order.
    """
    m = np.asarray(train_mask, dtype=bool)
    pos = np.flatnonzero(m)
    if len(pos) == 0:
        return pos
    breaks = np.flatnonzero(np.diff(pos) > 1) + 1
    segs = np.split(pos, breaks)
    if len(segs) == 2 and segs[0][0] == 0 and segs[1][-1] == len(m) - 1:
        return np.concatenate([segs[1], segs[0]])
    return pos

def moving_block_indices(n, block, rng):
    """Circular moving-block bootstrap indices (fixed block length).

    Parameters:
    n (int): Length of the series to resample.
    block (int): Fixed block length.
    rng (np.random.Generator): Random number generator.

    Returns:
    np.ndarray: (n,) int array of resampled positions.
    """
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, nb)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def block_bootstrap_mean(x, block, n_boot=2000, seed=0, level=0.95):
    """Mean of a series with a STATIONARY (Politis-Romano) bootstrap interval; `block` is the mean
    block length. NaNs are dropped first, with a logged warning giving their number; the time order of the
    remaining steps is kept.

    Source: Politis, D. N., & Romano, J. P. (1994), Journal of the American Statistical Association, 89(428),
    1303-1313, via stationary_bootstrap_indices.

    Parameters:
    x (array-like): Series.
    block (float): Mean block length of the stationary bootstrap.
    n_boot (int): Number of bootstrap replicates. Default 2000.
    seed (int): Seed of the random number generator. Default 0.
    level (float): Coverage of the percentile interval. Default 0.95.

    Returns:
    tuple: (mean, lo, hi).
    """
    x = np.asarray(x, float)
    n_dropped = int((~np.isfinite(x)).sum())
    if n_dropped:
        log.warning(f"block_bootstrap_mean: dropped {n_dropped} of {len(x)} non-finite steps before bootstrapping")
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    means = np.array([x[stationary_bootstrap_indices(len(x), block, rng)].mean() for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def masked_stationary_indices(n_full, positions, b, rng, lookup=None):
    """Chronological-index stationary bootstrap of a (possibly non-contiguous) training subset.

    Draws a Politis-Romano stationary-bootstrap resample of the FULL timeline (length n_full) and keeps
    only the drawn positions that belong to the subset. A block is therefore a real contiguous run of
    time: no block is stitched across the join of two training segments. The length of the returned
    resample varies from replicate to replicate (close to len(positions) in expectation).

    Source: Politis, D. N., & Romano, J. P. (1994), Journal of the American Statistical Association, 89(428),
    1303-1313, via stationary_bootstrap_indices.

    Parameters:
    n_full (int): Length of the full timeline.
    positions (np.ndarray): Sorted integer time positions of the subset rows.
    b (float): Mean block length of the stationary bootstrap.
    rng (np.random.Generator): Random number generator.
    lookup (np.ndarray): Optional precomputed full-length map time position -> row (-1 for non-members).

    Returns:
    np.ndarray: Row indices into the compact subset array (0..len(positions)-1).
    """
    if lookup is None:
        lookup = np.full(n_full, -1, dtype=np.int64)
        lookup[np.asarray(positions)] = np.arange(len(positions))
    rows = lookup[stationary_bootstrap_indices(n_full, b, rng)]
    return rows[rows >= 0]
