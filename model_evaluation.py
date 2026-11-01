"""model_evaluation.py

Model evaluation helpers: block-bootstrap confidence intervals for the cross-validated prediction error (CVPE),
the marginal-uniformity screen of the neural copulas, and a wrapper around the Model Confidence Set.

CVPE follows Acar, Craiu and Yao (2011, eq. 4): the squared error of the conditional expectation of each
variable given the other (and the covariate), summed over both directions. The bootstrap is the stationary
bootstrap of Politis and Romano (1994). The Model Confidence Set is that of Hansen, Lunde and Nason (2011).

References:
Acar, E. F., Craiu, R. V., & Yao, F. (2011). Dependence calibration in conditional copulas: A nonparametric
    approach. Biometrics, 67(2), 445-453.
Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.
Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical
    Association, 89(428), 1303-1313.
"""
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import kstest
from tqdm.auto import tqdm

import gate_config as GC
from copula_families import QUAD_N, expectations_pair
from data_split import stationary_bootstrap_indices


def cvpe_ci_all_models(loss_matrix, model_names, b, n_boot=5000, seed=0, level=0.95,
                        show_progress=True):
    """Stationary block-bootstrap confidence interval of the pooled CVPE of every model, from a loss matrix.

    The data are one fixed series, so the interval comes from a stationary bootstrap (Politis & Romano, 1994)
    over the test rows, not from retraining. Use the same mean block length b as for the Model Confidence Set, so
    that the interval and the MCS p-values describe temporal dependence at the same timescale.

    The rows of loss_matrix must be in their original time order. A matrix filtered to a scattered subset of rows
    (for example a joint-tail mask) is not suitable: the bootstrap resamples contiguous runs, which only reflects
    the temporal dependence if adjacent rows are adjacent in time.

    Parameters:
    loss_matrix (np.ndarray): (n, n_models) per-step losses, one column per model, rows in time order.
    model_names (sequence): Model names, aligned with the columns of loss_matrix.
    b (float): Mean block length of the stationary bootstrap.
    n_boot (int): Number of bootstrap replicates. Default 5000.
    seed (int): Bootstrap seed. Default 0.
    level (float): Confidence level. Default 0.95.
    show_progress (bool): Show a tqdm bar over the replicates. Default True.

    Returns:
    pd.DataFrame: Indexed by model name ('variant'), with columns n, mean_loss, ci_lo and ci_hi.
    """
    n, n_models = loss_matrix.shape
    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_boot, n_models))
    iterator = range(n_boot)
    if show_progress:
        iterator = tqdm(iterator, desc='CVPE block bootstrap (all admissible models)')
    for i in iterator:
        idx = stationary_bootstrap_indices(n, b, rng)
        boot_means[i] = loss_matrix[idx].mean(axis=0)

    means = loss_matrix.mean(axis=0)
    alpha = (1 - level) / 2
    los = np.percentile(boot_means, alpha * 100, axis=0)
    his = np.percentile(boot_means, (1 - alpha) * 100, axis=0)
    rows = [{'variant': name, 'n': n, 'mean_loss': float(means[i]),
             'ci_lo': float(los[i]), 'ci_hi': float(his[i])}
            for i, name in enumerate(model_names)]
    return pd.DataFrame(rows).set_index('variant')


def cvpe_only(density_fn, u1_arr, u2_arr, w_arr, grid_n=QUAD_N):
    """CVPE of a density on a set of points, without Monte Carlo sampling.

    Cheap enough to evaluate at every checkpoint during early stopping. It uses the same Gauss-Legendre
    quadrature (copula_families.expectations_pair) and the same loss, (u1 - e1)^2 + (u2 - e2)^2 averaged over the
    points, as the loss matrix of the Model Confidence Set. A checkpoint chosen by this criterion is therefore
    chosen for the quantity the MCS later compares, unlike the negative log-likelihood that training minimises.

    Parameters:
    density_fn (callable): density_fn(pts, w) -> (m,) density array.
    u1_arr (np.ndarray): First variable's pseudo-observations (a validation split during early stopping,
        never the final test set).
    u2_arr (np.ndarray): Second variable's pseudo-observations.
    w_arr (np.ndarray): Covariate values, same length as u1_arr.
    grid_n (int): Number of Gauss-Legendre nodes. Default copula_families.QUAD_N.

    Returns:
    float: Mean squared error of the conditional-expectation predictions, both directions, over all points.
    """
    u1_arr = np.asarray(u1_arr, float); u2_arr = np.asarray(u2_arr, float)
    e1, e2 = expectations_pair(density_fn, u1_arr, u2_arr, w_arr, n=grid_n)
    return float(np.mean((u1_arr - e1) ** 2 + (u2_arr - e2) ** 2))


def neural_conditional_predictions(density_fn, u1_te, u2_te, w_te, grid_n=QUAD_N):
    """Conditional-expectation predictions and CVPE of a continuously conditioned neural copula (C-CODINE or C-CopFlow).

    The expectations are computed by Gauss-Legendre integration of the model's own density
    (copula_families.expectations_pair), the same rule that scores the unconditional and split CODINE and the
    parametric families. The per-point arrays are returned so that the caller can build the loss matrix of the MCS.

    Parameters:
    density_fn (callable): density_fn(pts, w) -> (m,) density array, e.g.
        lambda pts, wv: eval_conditional_density(net, pts, wv) or lambda pts, wv: eval_flow_density(flow, pts, wv).
    u1_te (np.ndarray): First variable's test pseudo-observations.
    u2_te (np.ndarray): Second variable's test pseudo-observations.
    w_te (np.ndarray): Test covariate values.
    grid_n (int): Number of Gauss-Legendre nodes. Default copula_families.QUAD_N.

    Returns:
    tuple: (e1, e2, cvpe), where e1 and e2 are (n_test,) arrays of conditional expectations and cvpe is the mean
        squared error over both directions.
    """
    e1, e2 = expectations_pair(density_fn, u1_te, u2_te, w_te, n=grid_n)

    cvpe = float(np.mean((u1_te - e1) ** 2 + (u2_te - e2) ** 2))
    return e1, e2, cvpe


def marginal_uniformity_check(sample_fn, w_values, n_samples=1_000, seed=0):
    """Kolmogorov-Smirnov test of each coordinate's marginal against Uniform(0,1), at each of several w values.

    The margins of a copula are exactly uniform, but a joint density that integrates to one does not guarantee
    this, and a network trained by maximum likelihood has no structural guarantee that its generated margins are
    uniform. The self-consistency checks of the networks verify normalisation only.

    The admissibility gate of the conditional neural variants is built on this table through ks_gate_summary: a
    seed passes if the largest KS statistic D_max over all covariate values and both coordinates is at most
    the threshold in gate_config. The p-values and the rejection rate of marginal_uniformity_summary are
    diagnostics only and never gate.

    Parameters:
    sample_fn (callable): sample_fn(w, n, seed) -> (n, 2) samples, e.g.
        lambda wv, n, s: sample_from_flow(flow, wv, n_samples=n, seed=s).
    w_values (iterable): Covariate values to check at.
    n_samples (int): Number of draws per w value. Default 1000.
    seed (int): Base seed; the j-th w value uses seed + j. Default 0.

    Returns:
    pd.DataFrame: One row per w, with columns w, ks_stat_u1, ks_p_u1, ks_stat_u2 and ks_p_u2.
    """
    rows = []
    for j, wv in enumerate(w_values):
        samples = sample_fn(float(wv), n_samples, seed + j)
        stat1, p1 = kstest(samples[:, 0], 'uniform')
        stat2, p2 = kstest(samples[:, 1], 'uniform')
        rows.append({'w': float(wv), 'ks_stat_u1': stat1, 'ks_p_u1': p1,
                     'ks_stat_u2': stat2, 'ks_p_u2': p2})
    return pd.DataFrame(rows)


def marginal_uniformity_summary(df_check, alpha=0.05):
    """Bonferroni-corrected rejection rate of the marginal-uniformity KS tests (diagnostic only).

    The rate and the underlying p-values do not decide admissibility; ks_gate_summary (an effect-size gate on
    D_max) does. The maximum and mean KS statistic are reported next to the rejection rate: the rate says whether a
    deviation from exact uniformity was detected, the statistic says how large it was.

    Parameters:
    df_check (pd.DataFrame): Output of marginal_uniformity_check.
    alpha (float): Nominal significance level before the Bonferroni correction. Default 0.05.

    Returns:
    dict: {'n_tests': int, 'alpha_corrected': float, 'rejection_rate': float, 'max_ks_stat': float,
        'mean_ks_stat': float}.
    """
    p_values = np.concatenate([df_check['ks_p_u1'].values, df_check['ks_p_u2'].values])
    ks_stats = np.concatenate([df_check['ks_stat_u1'].values, df_check['ks_stat_u2'].values])
    n_tests = len(p_values)
    alpha_corrected = alpha / n_tests
    rejection_rate = float(np.mean(p_values < alpha_corrected))
    return {
        'n_tests': n_tests,
        'alpha_corrected': alpha_corrected,
        'rejection_rate': rejection_rate,
        'max_ks_stat': float(np.max(ks_stats)),
        'mean_ks_stat': float(np.mean(ks_stats)),
    }


def ks_gate_summary(df_check, grid=False):
    """Gate decision for one trained network from a marginal_uniformity_check table.

    D_max is the largest KS statistic over all covariate values (rows) and both coordinates. The seed passes if
    D_max <= gate_config.ks_threshold(n_tests, grid): an effect-size rule without p-value. The threshold is
    gate_config.KS_D_MAX for a two-test check and slightly larger for a check with more tests or the grid sampler.
    A table with no rows or a non-finite D_max fails.

    Parameters:
    df_check (pd.DataFrame): Output of marginal_uniformity_check.
    grid (bool): True if the samples come from the CODINE grid sampler. Default False.

    Returns:
    dict: {'d_max': float, 'n_tests': int, 'pass': bool}.
    """
    stats = np.concatenate([np.asarray(df_check['ks_stat_u1'], float), np.asarray(df_check['ks_stat_u2'], float)])
    dm = float(np.max(stats)) if stats.size else float('nan')
    ok = GC.ks_seed_pass(dm, n_tests=stats.size, grid=grid)
    return {'d_max': dm, 'n_tests': int(stats.size), 'pass': bool(ok)}


# ── Model Confidence Set ─────────────────────────────────────────────────────
@njit
def _seed_numba_rng(seed):
    """Reseed Numba's internal random number generator from inside jitted code.

    np.random.seed called from plain Python does not reseed the random draws of a jitted function: Numba keeps
    a separate RNG state that is only reset by np.random.seed called inside jitted code. The bootstrap routines of
    the model_confidence_set package are jitted, so run_mcs calls this function to make its draws reproducible.

    Parameters:
    seed (int): Seed for Numba's RNG.
    """
    np.random.seed(seed)


def run_mcs(loss_matrix, model_names, block_len, alpha=0.10, n_boot=5000, seed=0, method="R"):
    """Run the Model Confidence Set (Hansen, Lunde & Nason, 2011) on a loss matrix, with a stationary bootstrap.

    Thin wrapper around model_confidence_set.ModelConfidenceSet. The losses are passed as a named DataFrame so
    that the results report model names instead of integer indices.

    The package eliminates the model with the largest average excess loss at each step (the rule e_max of
    Hansen et al., 2011) and not the rule e_R that the paper pairs with T_R.

    Parameters:
    loss_matrix (np.ndarray): (n_test, n_models) loss matrix.
    model_names (list): Model names, one per column of loss_matrix.
    block_len (int): Mean block length of the stationary bootstrap, one value for the whole matrix (all models
        are compared on one shared test index).
    alpha (float): MCS level; models with a p-value below alpha are eliminated. The p-values do not depend on
        alpha. Default 0.10.
    n_boot (int): Number of bootstrap replicates. Default 5000.
    seed (int): Seed for Numba's RNG, set (via _seed_numba_rng) after the object is constructed and immediately
        before compute(), so that the bootstrap draws depend only on this value. Default 0.
    method (str): Test statistic of the package. "R" is the range statistic T_R = max |t_ij| of Hansen et al.
        (2011, Section 3.1.2); "SQ" is the sum of the squared standardised pairwise differences, an option of
        the package that is not one of the statistics of that paper. Default "R".

    Returns:
    model_confidence_set.ModelConfidenceSet: The computed MCS object.
    """
    # Imported here so that the rest of this module works without the model_confidence_set package.
    from model_confidence_set import ModelConfidenceSet

    losses_df = pd.DataFrame(loss_matrix, columns=model_names)
    mcs = ModelConfidenceSet(
        losses_df, n_boot=n_boot, alpha=alpha,
        block_len=block_len, bootstrap_variant="stationary", method=method,
    )
    # The constructor runs one jitted warm-up bootstrap that advances Numba's RNG state, so the
    # state is set after construction, immediately before compute().
    _seed_numba_rng(seed)
    mcs.compute()
    return mcs
