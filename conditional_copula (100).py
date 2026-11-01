"""conditional_copula.py

Acar-type conditional copulas: a single-parameter calibration curve theta(w) fitted by kernel-weighted local
likelihood (local-linear, Epanechnikov kernel), with cross-validated bandwidth choice and a bootstrap
confidence band for Kendall's tau(w).

The calibration framework follows Acar, Craiu and Yao (2011): theta(w) = g^-1(eta(w)), with eta estimated by
local-linear (p = 1) local likelihood and an Epanechnikov kernel. Differences from that paper: the bandwidth is chosen by purged chronological K-fold cross-validation
instead of leave-one-out (serial dependence; Racine, 2000), and the confidence band is a bootstrap band (the paper suggests
the bootstrap when the marginals are estimated nonparametrically).

The last section gives block-bootstrap confidence bands for the fitted calibration curves tau(w) of every blocked fold
(fold_averaged_curve). Only the local-likelihood step is repeated, on the
fold training rows with the fold's own bandwidth h and block length b (the same inputs as fold_pipeline.acar_job). No
network is retrained and no stored result is changed. The bands are percentile bands of a stationary bootstrap of the
local-likelihood calibration estimator.

References:
Acar, E. F., Craiu, R. V., & Yao, F. (2011). Dependence calibration in conditional copulas: A nonparametric
    approach. Biometrics, 67(2), 445-453.
Demarta, S., & McNeil, A. J. (2005). The t copula and related copulas. International Statistical Review,
    73(1), 111-129.
Genest, C., & Rivest, L.-P. (1993). Statistical inference procedures for bivariate Archimedean copulas.
    Journal of the American Statistical Association, 88(423), 1034-1043.
Joe, H. (2014). Dependence modeling with copulas. CRC Press.
Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical Association,
    89(428), 1303-1313.
Racine, J. (2000). Consistent cross-validatory model-selection for dependent data: hv-block cross-validation.
    Journal of Econometrics, 99(1), 39-61.
"""
import warnings

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm as sp_norm, t as t_dist
from tqdm.auto import tqdm

from copula_families import fit_family, theta_col_of
from data_split import purged_chrono_folds, stationary_bootstrap_indices, train_test_split_chrono
from run_support import get_logger, resolve_pool_workers
import os

log = get_logger()


# ── bandwidth selection policy ───────────────────────────────────────────────
def choose_acar_bandwidths(cv_table, families, min_coverage=0.95):
    """Choose one bandwidth per family from the cross-validation table, with an explicit fallback policy.

    Primary rule: the highest cross-validation score among candidates that meet the predeclared coverage rule
    (min_coverage) and have a finite score on every inner fold.

    If no candidate of a family qualifies, the fallback prefers the highest observed validation coverage
    (not the mean score on a more favourable subset), then the number of scored folds, then the raw mean
    fold log-likelihood, then the bandwidth as a deterministic tie-break. Each such choice is reported as a
    fallback. When every candidate has zero finite validation likelihoods, the widest tested bandwidth is used and
    marked UNVALIDATED. The fallback never adds test data and never relaxes the primary coverage rule.

    Parameters:
    cv_table (pd.DataFrame): One row per (family, bandwidth), as built by select_model (columns family,
        bandwidth, cv_score, raw_cv_score, n_valid, n_attempted, n_scored_folds, coverage, eligible).
    families (iterable): Family names to choose a bandwidth for.
    min_coverage (float): Minimum share of attempted validation points with a finite log-likelihood.
        Default 0.95.

    Returns:
    dict: {family: dict} with keys family, bandwidth, selection_status ('primary',
        'low_coverage_fallback' or 'unvalidated_widest_fallback'), selection_reason, coverage, n_valid,
        n_attempted, n_scored_folds, cv_score, raw_cv_score and min_required_coverage.
    """
    decisions = {}
    for family in families:
        subset = cv_table.loc[cv_table['family'] == family].copy()
        if subset.empty:
            raise ValueError(f'Acar CV grid has no rows for requested family {family!r}')
        subset['bandwidth'] = pd.to_numeric(subset['bandwidth'], errors='coerce')
        if not ((subset['bandwidth'] > 0) & np.isfinite(subset['bandwidth'])).all():
            raise ValueError(f'Acar CV grid contains invalid bandwidths for {family!r}')
        covered = subset.loc[subset['eligible'].fillna(False).astype(bool)
                             & np.isfinite(subset['cv_score'])]
        if not covered.empty:
            best = covered.sort_values(['cv_score', 'coverage', 'bandwidth'],
                                       ascending=[False, False, False], kind='stable').iloc[0]
            status = 'primary'
            rationale = f'met {min_coverage:.0%} minimum coverage and finite score on every inner fold'
        else:
            # cv_score is deliberately -inf for ineligible candidates. The
            # diagnostic raw_cv_score is the mean of the finite fold means.
            usable = subset.loc[(subset['n_valid'] > 0)
                                & np.isfinite(subset['raw_cv_score'])]
            if not usable.empty:
                usable = usable.copy()
                usable['n_scored_folds'] = pd.to_numeric(
                    usable.get('n_scored_folds', pd.Series(0, index=usable.index)),
                    errors='coerce').fillna(0)
                best = usable.sort_values(
                    ['coverage', 'n_scored_folds', 'raw_cv_score', 'bandwidth'],
                    ascending=[False, False, False, False], kind='stable').iloc[0]
                status = 'low_coverage_fallback'
                rationale = ('no candidate met the validation coverage rule; '
                             'maximize coverage, then scored folds, then available CV score')
            else:
                best = subset.sort_values('bandwidth', ascending=False, kind='stable').iloc[0]
                status = 'unvalidated_widest_fallback'
                rationale = ('NO candidate produced a finite validation score; '
                             'widest tested bandwidth is an emergency default, NOT CV-validated')
        decisions[family] = {
            'family': str(family), 'bandwidth': float(best['bandwidth']),
            'selection_status': status, 'selection_reason': rationale,
            'coverage': float(best['coverage']),
            'n_valid': int(best['n_valid']),
            'n_attempted': int(best['n_attempted']),
            'n_scored_folds': int(best.get('n_scored_folds', 0)),
            'cv_score': float(best['cv_score']) if np.isfinite(best['cv_score']) else None,
            'raw_cv_score': (float(best['raw_cv_score'])
                             if np.isfinite(best['raw_cv_score']) else None),
            'min_required_coverage': float(min_coverage),
        }
    return decisions


# ── kernel and log-densities ─────────────────────────────────────────────────
def epanechnikov(t):
    """Epanechnikov kernel, K(t) = 0.75 * (1 - t^2) for |t| <= 1, else 0 (Acar et al., 2011, Section 2.1).

    Parameters:
    t (np.ndarray): Standardised distance values.

    Returns:
    np.ndarray: Kernel weights, same shape as t.
    """
    return np.where(np.abs(t) <= 1, 0.75 * (1 - t**2), 0.0)

def kern_weights(x, x0, h):
    """Epanechnikov kernel weights for local-likelihood fitting at covariate value x0.

    Parameters:
    x (np.ndarray): Covariate values for all observations.
    x0 (float): The covariate value to center the kernel on.
    h (float): Bandwidth.

    Returns:
    np.ndarray: Kernel weight per observation.
    """
    return epanechnikov((x - x0) / h)

def _safe(u):
    """Clip pseudo-observations away from the [0, 1] boundary for numerical stability.

    Parameters:
    u (np.ndarray): Pseudo-observations in [0, 1].

    Returns:
    np.ndarray: u clipped to [1e-6, 1 - 1e-6].
    """
    return np.clip(u, 1e-6, 1 - 1e-6)

def _log_dens_gaussian(u, v, rho):
    """Log-density of the bivariate Gaussian copula, the copula of the bivariate normal distribution.

    Closed form as in Joe (2014, Chapter 4); the copula of the multivariate normal distribution
    (Demarta & McNeil, 2005, Section 2.2).

    Parameters:
    u (np.ndarray): First variable's pseudo-observations.
    v (np.ndarray): Second variable's pseudo-observations.
    rho (np.ndarray): Correlation parameter (per-observation or scalar).

    Returns:
    np.ndarray: Log-density values.
    """
    rho = np.clip(rho, -0.9999, 0.9999)
    qu  = sp_norm.ppf(_safe(u))
    qv  = sp_norm.ppf(_safe(v))
    r2  = rho**2
    return (-0.5 * np.log(1 - r2)
            - (r2 * (qu**2 + qv**2) - 2 * rho * qu * qv) / (2 * (1 - r2)))

def _log_dens_clayton(u, v, theta):
    """Log-density of the bivariate Clayton copula (Joe, 2014, Chapter 4).

    Parameters:
    u (np.ndarray): First variable's pseudo-observations.
    v (np.ndarray): Second variable's pseudo-observations.
    theta (np.ndarray): Dependence parameter (per-observation or scalar).

    Returns:
    np.ndarray: Log-density values.
    """
    theta = np.maximum(theta, 1e-6)
    u, v  = _safe(u), _safe(v)
    S     = u**(-theta) + v**(-theta) - 1.0
    return (np.log(theta + 1)
            + (-theta - 1) * (np.log(u) + np.log(v))
            + (-1.0 / theta - 2) * np.log(np.maximum(S, 1e-300)))

def _log_dens_gumbel(u, v, theta):
    """Log-density of the bivariate Gumbel copula (Joe, 2014, Chapter 4).

    Parameters:
    u (np.ndarray): First variable's pseudo-observations.
    v (np.ndarray): Second variable's pseudo-observations.
    theta (np.ndarray): Dependence parameter (per-observation or scalar).

    Returns:
    np.ndarray: Log-density values.
    """
    theta = np.maximum(theta, 1.0001)
    u, v  = _safe(u), _safe(v)
    lu    = -np.log(u)
    lv    = -np.log(v)
    S     = lu**theta + lv**theta
    S1    = S**(1.0 / theta)
    return (-S1
            + (1.0 / theta - 2.0) * np.log(np.maximum(S, 1e-300))
            + np.log(np.maximum(S1 + theta - 1, 1e-300))
            + (theta - 1) * (np.log(np.maximum(lu, 1e-300)) + np.log(np.maximum(lv, 1e-300)))
            + lu + lv)


def _log_dens_gumbel180(u, v, theta):
    """Log-density of the Gumbel copula rotated by 180 degrees: c_G(1-u, 1-v; theta).

    The rotation maps the Gumbel's upper-tail dependence into the lower tail (lambda_L = 2 - 2**(1/theta),
    lambda_U = 0) and leaves the parameter range, the link function and Kendall's tau unchanged.

    Parameters:
    u (np.ndarray): First variable's pseudo-observations.
    v (np.ndarray): Second variable's pseudo-observations.
    theta (np.ndarray): Dependence parameter (per-observation or scalar).

    Returns:
    np.ndarray: Log-density values.
    """
    return _log_dens_gumbel(1.0 - np.asarray(u, float), 1.0 - np.asarray(v, float), theta)


def _log_dens_frank(u, v, theta):
    """Log-density of the bivariate Frank copula (Joe, 2014, Chapter 4).

    Parameters:
    u (np.ndarray): First variable's pseudo-observations.
    v (np.ndarray): Second variable's pseudo-observations.
    theta (np.ndarray): Dependence parameter (per-observation or scalar).

    Returns:
    np.ndarray: Log-density values.
    """
    u, v  = _safe(u), _safe(v)
    theta = np.where(np.abs(theta) < 1e-6, 1e-6, theta)
    et    = np.exp(-theta)
    D     = (et - 1) + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
    return (np.log(np.abs(theta))
            + np.log(np.maximum(np.abs(et - 1), 1e-300))
            - theta * u - theta * v
            - 2 * np.log(np.maximum(np.abs(D), 1e-300)))

def _log_dens_student(u, v, rho, nu):
    """Log-density of the bivariate Student-t copula (Demarta and McNeil, 2005, eq. 6: the joint t density at the
    t quantiles divided by the product of the marginal t densities).

    Falls back to the Gaussian log-density for nu > 49, the upper clip
    used by the predictors (so both agree on the Student-t range).

    Parameters:
    u (np.ndarray): First variable's pseudo-observations.
    v (np.ndarray): Second variable's pseudo-observations.
    rho (np.ndarray): Correlation parameter (per-observation or scalar).
    nu (float): Degrees of freedom.

    Returns:
    np.ndarray: Log-density values.
    """
    nu = float(nu)
    if nu > 49:
        return _log_dens_gaussian(u, v, rho)
    rho = np.clip(rho, -0.9999, 0.9999)
    u, v = _safe(np.asarray(u, dtype=float)), _safe(np.asarray(v, dtype=float))
    x = t_dist.ppf(u, df=nu)
    y = t_dist.ppf(v, df=nu)
    r2 = rho**2
    Q  = x**2 - 2*rho*x*y + y**2
    return (gammaln((nu + 2) / 2) + gammaln(nu / 2) - 2*gammaln((nu + 1) / 2)
            - 0.5 * np.log(1 - r2)
            - ((nu + 2) / 2) * np.log(1 + Q / (nu * (1 - r2)))
            + ((nu + 1) / 2) * np.log(1 + x**2 / nu)
            + ((nu + 1) / 2) * np.log(1 + y**2 / nu))

def _log_dens_joe(u, v, theta):
    """Log-density of the bivariate Joe copula.

    Joe (2014, Chapter 4). The density is the sum of two positive terms, log_t1 and log_t2, which are combined in
    log space with the log-sum-exp trick for numerical stability.

    Parameters:
    u (np.ndarray): First variable's pseudo-observations.
    v (np.ndarray): Second variable's pseudo-observations.
    theta (np.ndarray): Dependence parameter (per-observation or scalar).

    Returns:
    np.ndarray: Log-density values.
    """
    theta = np.maximum(theta, 1.0001)
    u, v  = _safe(u), _safe(v)
    ub    = 1.0 - u
    vb    = 1.0 - v
    A     = ub**theta
    B     = vb**theta
    S     = A + B - A * B

    log_S    = np.log(np.maximum(S, 1e-300))
    log_ub   = np.log(np.maximum(ub, 1e-300))
    log_vb   = np.log(np.maximum(vb, 1e-300))
    log_1mA  = np.log(np.maximum(1.0 - A, 1e-300))
    log_1mB  = np.log(np.maximum(1.0 - B, 1e-300))

    # term1: (1/theta)(1 - 1/theta) * S^(1/theta-2) * dSdu * dSdv
    log_t1 = (np.log(1.0/theta) + np.log(np.maximum(1.0 - 1.0/theta, 1e-300))
              + (1.0/theta - 2) * log_S
              + np.log(theta) + (theta-1)*log_ub + log_1mB
              + np.log(theta) + (theta-1)*log_vb + log_1mA)

    # term2: (1/theta) * S^(1/theta-1) * d2S/dudv  [positive: minus times minus]
    log_t2 = (np.log(1.0/theta)
              + (1.0/theta - 1) * log_S
              + 2*np.log(theta) + (theta-1)*log_ub + (theta-1)*log_vb)

    # Both terms positive, combine in log space
    log_max = np.maximum(log_t1, log_t2)
    return log_max + np.log(np.maximum(
        np.exp(log_t1 - log_max) + np.exp(log_t2 - log_max), 1e-300))

def _signed(f_pos):
    """Log-density of a signed family from the log-density ``f_pos`` of its own rotation.

    The primary parameter is SIGNED: ``theta >= 0`` uses ``f_pos(u, v, |theta|)``, ``theta < 0`` the family
    turned by 90 degrees, ``f_pos(1 - u, v, |theta|)`` (Clayton, Gumbel, Joe: pyvinecopulib rotation 90;
    Gumbel (180 degrees): rotation 270).

    Parameters:
    f_pos (callable): Log-density f_pos(u, v, theta) of the family's own rotation, for theta >= 0.

    Returns:
    callable: ld(u, v, theta) with a signed theta.
    """
    def ld(u, v, theta):
        """Log-density of the signed family.

        Parameters:
        u (array-like): First pseudo-observation.
        v (array-like): Second pseudo-observation.
        theta (array-like): Signed parameter; a negative value selects the turned branch with parameter |theta|.

        Returns:
        np.ndarray: Log-density at each point.
        """
        th = np.asarray(theta, float)
        u = np.asarray(u, float); v = np.asarray(v, float)
        a = np.abs(th)
        return np.where(th >= 0, f_pos(u, v, a), f_pos(1.0 - u, v, a))
    return ld


def _signed_link(par0, t_max=50.0):
    """Link functions of a signed family.

    eta in R is the signed excess over independence: theta = sign(eta) * (par0 + |eta|), so eta = 0 is
    independence and the two branches join there.

    Parameters:
    par0 (float): Parameter value at independence (0 for Clayton, 1 for Gumbel and Joe).
    t_max (float): Clip bound for |eta| in the inverse link. Default 50.0.

    Returns:
    tuple: (link_inv, link_fwd), the maps eta -> theta and theta -> eta.
    """
    def inv(eta):
        """Map eta to the signed parameter theta = sign(eta) * (par0 + |eta|), with |eta| clipped at t_max.

        Parameters:
        eta (array-like): Signed excess over independence.

        Returns:
        np.ndarray: The signed parameter theta.
        """
        e = np.clip(eta, -t_max, t_max)
        return np.where(e >= 0, par0 + e, -(par0 + np.abs(e)))
    def fwd(theta):
        """Map the signed parameter theta back to eta (the inverse of the inverse link).

        Parameters:
        theta (array-like): Signed parameter.

        Returns:
        np.ndarray: Signed excess over independence.
        """
        th = np.asarray(theta, float)
        return np.where(th >= 0, th - par0, -(np.abs(th) - par0))
    return inv, fwd


# ── FAMILY_INFO ──────────────────────────────────────────────────────────────
# link_inv/link_fwd/analytical are used by the single-parameter local-linear (BFGS) path in fit_at_point.
# Only the family's first pyvinecopulib parameter varies locally with the covariate in that path; any remaining
# parameters (Student-t's degrees of freedom) are held fixed at their unconditional (global) MLE via fixed_params.
_C_INV, _C_FWD = _signed_link(0.0)
_G_INV, _G_FWD = _signed_link(1.0)
FAMILY_INFO = {
    'gaussian': dict(pv_fam=pv.BicopFamily.gaussian,
                     link_inv=np.tanh,
                     link_fwd=lambda theta: np.arctanh(np.clip(theta, -0.9999, 0.9999)),
                     analytical=_log_dens_gaussian),
    'clayton':  dict(pv_fam=pv.BicopFamily.clayton, signed=True, neg_rotation=90, par0=0.0,
                     link_inv=_C_INV, link_fwd=_C_FWD,
                     analytical=_signed(_log_dens_clayton)),
    'gumbel':   dict(pv_fam=pv.BicopFamily.gumbel, signed=True, neg_rotation=90, par0=1.0,
                     link_inv=_G_INV, link_fwd=_G_FWD,
                     analytical=_signed(_log_dens_gumbel)),
    'gumbel180': dict(pv_fam=pv.BicopFamily.gumbel, rotation=180, signed=True, neg_rotation=270, par0=1.0,
                      link_inv=_G_INV, link_fwd=_G_FWD,
                      analytical=_signed(_log_dens_gumbel180)),
    'frank':    dict(pv_fam=pv.BicopFamily.frank,
                     link_inv=lambda t: t,
                     link_fwd=lambda theta: theta,
                     analytical=_log_dens_frank),
    'student':  dict(pv_fam=pv.BicopFamily.student,
                     link_inv=np.tanh,
                     link_fwd=lambda theta: np.arctanh(np.clip(theta, -0.9999, 0.9999)),
                     analytical=_log_dens_student),
    'joe':      dict(pv_fam=pv.BicopFamily.joe, signed=True, neg_rotation=90, par0=1.0,
                     link_inv=_G_INV, link_fwd=_G_FWD,
                     analytical=_signed(_log_dens_joe)),
}
for _info in FAMILY_INFO.values():
    _info.setdefault('rotation', 0)          # own rotation (the signed families are turned for tau < 0)
    _info.setdefault('signed', False)

def fit_global(u1, u2, family):
    """Fit a copula family unconditionally (maximum pseudo-likelihood) and split its parameters into primary and fixed.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations.
    u2 (np.ndarray): Second variable's pseudo-observations.
    family (str): Family name, a key in FAMILY_INFO.

    Returns:
    tuple: (primary, fixed, cop), where primary is the first pyvinecopulib
        parameter (float), fixed is a list of any remaining parameters,
        and cop is the fitted pv.Bicop object.
    """
    data        = np.column_stack([u1, u2])
    cop         = fit_family(family, data)      # family and rotation; signed families keep the better branch
    params_flat = cop.parameters.ravel()
    primary     = float(theta_col_of(cop))      # SIGNED primary parameter (negative = turned branch)
    fixed       = [float(p) for p in params_flat[1:]]
    return primary, fixed, cop

def log_density(u, v, theta_primary, family, fixed_params=None):
    """Evaluate a copula's log-density at each observation, given a per-observation primary parameter.

    Uses the closed-form analytical log-density in FAMILY_INFO (all seven families have one; there is no
    pyvinecopulib fallback, which cannot take per-observation parameters).

    Parameters:
    u (np.ndarray): First variable's pseudo-observations.
    v (np.ndarray): Second variable's pseudo-observations.
    theta_primary (np.ndarray or float): Primary dependence parameter,
        either a scalar (broadcast to every observation) or one value per
        observation.
    family (str): Family name, a key in FAMILY_INFO.
    fixed_params (list): Any remaining (non-primary) parameters, held
        fixed across all observations. Optional.

    Returns:
    np.ndarray: Log-density at each observation, with non-finite values
        replaced by -500.0.
    """
    info    = FAMILY_INFO[family]
    u_s     = np.asarray(u, dtype=float)
    v_s     = np.asarray(v, dtype=float)
    n       = len(u_s)
    theta_a = np.broadcast_to(np.asarray(theta_primary, dtype=float), (n,))

    if info['analytical'] is not None:
        with np.errstate(over='ignore', invalid='ignore'):
            ld = (info['analytical'](u_s, v_s, theta_a, *fixed_params)
                  if fixed_params else
                  info['analytical'](u_s, v_s, theta_a))
        return np.where(np.isfinite(ld), ld, -500.0)

    raise NotImplementedError(
        f"log_density: family {family!r} has no closed-form log-density in FAMILY_INFO, and pyvinecopulib "
        f"cannot take per-observation parameters.")

def fit_at_point(u1, u2, x, x0, h, family, fixed_params, beta_init=None, bfgs_options=None):
    """Fit the single-parameter local-linear calibration model at one covariate value x0.

    theta(w) = link_inv(beta_0 + beta_1 * (w - x0)) is fitted by kernel-weighted local likelihood
    (Epanechnikov kernel, bandwidth h; Acar et al., 2011), with Student-t's degrees of
    freedom held fixed at its global (unconditional) MLE. The kernel-weighted log-likelihood is maximised with BFGS.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations.
    u2 (np.ndarray): Second variable's pseudo-observations.
    x (np.ndarray): Covariate values.
    x0 (float): The covariate value to fit at.
    h (float): Kernel bandwidth.
    family (str): Family name, a key in FAMILY_INFO.
    fixed_params (list): Non-primary parameters, held fixed at this
        family's global (unconditional) MLE.
    beta_init (np.ndarray): Starting point [beta_0, beta_1] for BFGS. If
        None, initialised from the global fit. Optional.
    bfgs_options (dict): Passed to scipy.optimize.minimize. Optional.

    Returns:
    np.ndarray: Fitted [beta_0, beta_1], or beta_init unchanged if fewer
        than 3 observations have non-negligible kernel weight at x0.
    """
    if beta_init is None:
        g_fwd = FAMILY_INFO[family]['link_fwd']
        primary, _, _ = fit_global(u1, u2, family)
        beta_init = np.array([float(g_fwd(primary)), 0.0])

    # Quantities that do not depend on beta are computed once, outside the objective.
    g_inv  = FAMILY_INFO[family]['link_inv']
    w      = kern_weights(x, x0, h)
    active = w > 1e-10
    if active.sum() < 3:
        return beta_init                       # too few observations with kernel weight: no fit, beta_init returned

    w_a  = w[active]
    u1_a = u1[active]
    u2_a = u2[active]
    dx_a = x[active] - x0

    def _neg_loglik(beta):
        """Negative kernel-weighted local log-likelihood at the local-linear coefficients.

        Parameters:
        beta (np.ndarray): Intercept and slope of the linear predictor of eta in the centred covariate.

        Returns:
        float: The negative weighted log-likelihood.
        """
        theta_i = g_inv(beta[0] + beta[1] * dx_a)
        ld = log_density(u1_a, u2_a, theta_i, family, fixed_params)
        return -float(np.dot(w_a, ld))

    result = minimize(
        _neg_loglik,
        x0=beta_init,
        method='BFGS',
        options=bfgs_options or {'gtol': 1e-4, 'maxiter': 200},
    )
    return result.x                            # last iterate, also if BFGS reports no convergence

def traintest_cv_score(u1, u2, x, h, family, test_frac=0.2,
                        fixed_params=None, beta_init=None,
                        test_idx=None, bfgs_options=None,
                        train_mask=None, test_mask=None, purge=0):
    """Mean out-of-sample log-likelihood for a single-parameter calibration model.

    Refits theta(w) at each test point via fit_at_point, using only the
    training portion, then scores the held-out points under log_density.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations.
    u2 (np.ndarray): Second variable's pseudo-observations.
    x (np.ndarray): Covariate values.
    h (float): Kernel bandwidth.
    family (str): Family name, a key in FAMILY_INFO.
    test_frac (float): Used only if train_mask/test_mask are not provided.
        Default 0.2.
    fixed_params (list): Non-primary parameters, held fixed. If None,
        fit from the training portion. Optional.
    beta_init (np.ndarray): Starting point for fit_at_point's BFGS. If
        None, initialised from the global fit. Optional.
    test_idx (np.ndarray): Optional fixed array of test-set positions (as
        returned by np.where(test_mask)[0]) to score on, instead of the
        full test set. Pass the same test_idx across every (family, h)
        call in one sweep: this speeds up the ranking pass and makes every
        h score an identical set of points, rather than whatever survives
        its own kernel-weight threshold. Optional.
    bfgs_options (dict): Passed through to fit_at_point. A loose setting
        (e.g. {'gtol': 1e-3, 'maxiter': 50}) is enough for a ranking sweep
        across many bandwidths. None uses the default (gtol=1e-4,
        maxiter=200), as for the final calibration curve. Optional.
    train_mask (np.ndarray): If provided, used directly instead of
        test_frac. Optional.
    test_mask (np.ndarray): As above. Optional.
    purge (int): Used only if train_mask/test_mask are not provided.
        Default 0.

    Returns:
    tuple: (mean_log_likelihood, n_valid), the mean log-density over the
        successfully scored test points and their number.
    """
    n = len(u1)
    if train_mask is None or test_mask is None:
        train_mask, test_mask = train_test_split_chrono(n, test_frac, purge=purge)
    g_inv = FAMILY_INFO[family]['link_inv']
    g_fwd = FAMILY_INFO[family]['link_fwd']
    if fixed_params is None or beta_init is None:
        primary, fixed_params, _ = fit_global(u1[train_mask], u2[train_mask], family)
        beta_init = np.array([float(g_fwd(primary)), 0.0])
    eval_idx = test_idx if test_idx is not None else np.where(test_mask)[0]
    ll_total, n_valid = 0.0, 0
    for i in eval_idx:
        if kern_weights(x[train_mask], x[i], h).sum() < 3:
            continue
        try:
            beta = fit_at_point(u1[train_mask], u2[train_mask], x[train_mask],
                                 x[i], h, family, fixed_params, beta_init=beta_init,
                                 bfgs_options=bfgs_options)
            theta_i = float(g_inv(beta[0]))
            ld_i = log_density(np.array([u1[i]]), np.array([u2[i]]),
                                np.array([theta_i]), family, fixed_params)[0]
            if np.isfinite(ld_i):
                ll_total += ld_i
                n_valid += 1
        except Exception:
            continue
    return (ll_total / n_valid if n_valid else -np.inf), n_valid

def select_model(u1, u2, x,
                 families=tuple(FAMILY_INFO.keys()),
                 bandwidths=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5),
                 bfgs_options=None,
                 n_splits=3, purge=0, min_fold_train=50, n_jobs=1,
                 max_eval_points=None, eval_seed=0, min_coverage=0.95):
    """Rank (family, bandwidth) combinations by purged K-fold cross-validated log-likelihood.

    For each family and bandwidth, refits the local-linear calibration
    model (traintest_cv_score) across n_splits purged chronological folds
    (data_split.purged_chrono_folds) and averages the per-fold mean
    log-likelihood. Runs the (family, bandwidth) grid in parallel via
    joblib.

    Acar et al. (2011, Section 2.2) choose the bandwidth by the leave-one-out cross-validated local likelihood
    and caution that likelihoods are not comparable across copula families (they compare families with the
    CVPE). Here, purged folds replace leave-one-out, and the bandwidth chosen per family is the result of
    interest; the cross-family winner in the return value is a convenience that the callers in this
    codebase do not use.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations.
    u2 (np.ndarray): Second variable's pseudo-observations.
    x (np.ndarray): Covariate values.
    families (tuple): Family names to evaluate. Default all keys of FAMILY_INFO.
    bandwidths (tuple): Kernel bandwidths to evaluate.
    bfgs_options (dict): Passed through to traintest_cv_score. Optional.
    n_splits (int): Number of purged chronological folds. Default 3.
    purge (int): Purge gap length for purged_chrono_folds. Default 0.
    min_fold_train (int): Skip a fold if its training portion has fewer
        than this many observations. Default 50.
    n_jobs (int): Parallel jobs for joblib.Parallel. Default 1.
    max_eval_points (int): Cap on the number of test-set points scored per
        fold and (family, h). A small number (e.g. 60-100) suffices for an
        exploratory ranking sweep: every (family, h) is scored on the same
        subsample, so the ranking is stable, and the cost per fit no longer
        depends on the fold size. None scores the full test set. Optional.
    eval_seed (int): Seed of the subsample drawn when max_eval_points is set; the subsample is the
        same for every (family, h). Default 0.
    min_coverage (float): Minimum share of attempted validation points that must return a finite log-likelihood.
        In addition, every inner fold must have a finite score. Default 0.95.

    Returns:
    tuple: (df, best_family, best_bandwidth), where df is a DataFrame with
        one row per (family, bandwidth) combination (columns: family,
        bandwidth, cv_score, cv_score_std, raw_cv_score, n_scored_folds, n_valid, n_attempted, coverage, eligible, ...), and
        best_family/best_bandwidth identify the row with the highest
        cv_score. Candidates that fail the coverage rule are retained with their raw (partial) validation
        score for an explicitly labelled fallback; a family for which every candidate failed receives the
        widest tested bandwidth with an UNVALIDATED warning, rather than halting the outer-fold run.
    """
    from joblib import Parallel, delayed
    # Nuisance parameters and starting values are fitted on each inner training sample only, so that no
    # inner validation observation enters them.
    inner_folds = []
    for fold_id, (tr_mask, te_mask) in enumerate(purged_chrono_folds(
            len(u1), n_splits=n_splits, purge=purge)):
        if tr_mask.sum() < min_fold_train:
            continue
        fold_test_idx = np.flatnonzero(te_mask)
        if max_eval_points is not None and len(fold_test_idx) > max_eval_points:
            rng = np.random.default_rng(eval_seed + fold_id)
            fold_test_idx = np.sort(rng.choice(fold_test_idx, size=max_eval_points, replace=False))
        inner_folds.append((tr_mask, te_mask, fold_test_idx))
    if not inner_folds:
        raise ValueError("No valid inner folds for Acar bandwidth selection")
    init_by_family = {}
    for fam in families:
        init_by_family[fam] = []
        for tr_mask, _, _ in inner_folds:
            primary, fixed, _ = fit_global(u1[tr_mask], u2[tr_mask], fam)
            g_fwd = FAMILY_INFO[fam]['link_fwd']
            init_by_family[fam].append((fixed, np.array([float(g_fwd(primary)), 0.0])))

    def _score_one(fam, h):
        """Purged cross-validated score of one (family, bandwidth) combination over the inner folds.

        Parameters:
        fam (str): Family name.
        h (float): Kernel bandwidth.

        Returns:
        dict: Scores and coverage of the combination (cv_score, cv_score_std, raw_cv_score, n_valid, n_attempted, coverage, eligible
            and related fields).
        """
        fold_scores, fold_nvalid, fold_ntest = [], [], []
        for (tr_mask, te_mask, fold_test_idx), (fixed_params, beta_init) in zip(
                inner_folds, init_by_family[fam]):
            try:
                score, n_valid = traintest_cv_score(
                    u1, u2, x, h, fam,
                    fixed_params=fixed_params, beta_init=beta_init,
                    bfgs_options=bfgs_options,
                    train_mask=tr_mask, test_mask=te_mask,
                    test_idx=fold_test_idx)
            except Exception:
                score, n_valid = -np.inf, 0
            fold_scores.append(score)
            fold_nvalid.append(n_valid)
            fold_ntest.append(len(fold_test_idx))
        coverage = np.sum(fold_nvalid) / max(int(np.sum(fold_ntest)), 1)
        # A candidate must not improve its average by silently dropping hard points: it is eligible only if
        # coverage and every inner fold are sufficient.
        finite_scores = np.asarray(fold_scores, dtype=float)
        valid_fold_scores = finite_scores[np.isfinite(finite_scores)]
        eligible = bool(coverage >= min_coverage and len(valid_fold_scores) == len(inner_folds))
        return {
            'family': fam, 'bandwidth': h,
            'cv_score': float(np.mean(fold_scores)) if eligible else -np.inf,
            'cv_score_std': float(np.std(fold_scores)) if eligible else np.nan,
            # Kept for the low-coverage fallback only; never a substitute for the primary score.
            'raw_cv_score': float(np.mean(valid_fold_scores)) if len(valid_fold_scores) else np.nan,
            'n_scored_folds': int(len(valid_fold_scores)),
            'n_valid': int(np.sum(fold_nvalid)),
            'n_attempted': int(np.sum(fold_ntest)),
            'coverage': float(coverage),
            'eligible': eligible,
        }
    jobs = [(fam, h) for fam in families for h in bandwidths]
    log.info(f"Model selection: {len(jobs)} (family, h) combos, n_jobs={n_jobs}, "
          f"max_eval_points={max_eval_points if max_eval_points is not None else 'ALL (slow)'}")
    n_jobs = (1 if n_jobs == 1 else resolve_pool_workers(
        n_tasks=len(jobs), max_workers=n_jobs,
        worker_gib=float(os.getenv('THESIS_ACAR_WORKER_GIB', '2.0'))))
    records = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_score_one)(fam, h) for fam, h in jobs
    )
    df = pd.DataFrame(records)
    decisions = choose_acar_bandwidths(df, families, min_coverage=min_coverage)
    df['selected'] = False
    df['selection_status'] = ''
    for fam, choice in decisions.items():
        mask = (df['family'] == fam) & (df['bandwidth'] == choice['bandwidth'])
        df.loc[mask, 'selected'] = True
        df.loc[mask, 'selection_status'] = choice['selection_status']
        message = (f" {fam}: h={choice['bandwidth']:.2f}, "
                   f"coverage={choice['coverage']:.1%} "
                   f"({choice['n_valid']}/{choice['n_attempted']} points), "
                   f"status={choice['selection_status']}")
        if choice['selection_status'] == 'primary':
            log.info(message)
        else:
            log.warning('ACAR BANDWIDTH FALLBACK: ' + message + '; ' + choice['selection_reason'])
    # The per-family selections are marked in the dataframe (columns selected, selection_status).
    eligible = df[df['eligible']]
    if not eligible.empty:
        best = eligible.loc[eligible['cv_score'].idxmax()]
        return df, str(best['family']), float(best['bandwidth'])
    # No family met the coverage rule: prefer a scored fallback over an unvalidated one.
    ranked = sorted(decisions.values(), key=lambda v: (
        v['n_valid'] > 0, v['coverage'], v['n_scored_folds'],
        -np.inf if v['raw_cv_score'] is None else v['raw_cv_score'], v['bandwidth']),
        reverse=True)
    return df, ranked[0]['family'], ranked[0]['bandwidth']

def theta_to_tau(theta, family):
    """Convert the primary copula parameter to Kendall's tau.

    Gaussian and Student-t: tau = (2 / pi) arcsin(rho) (Demarta & McNeil, 2005, eq. 9). Clayton: theta / (theta + 2).
    Gumbel (also rotated by 180 degrees): 1 - 1 / theta. The Archimedean values follow from the generator identity of
    Genest and Rivest (1993), tau = 1 + 4 * integral of phi(t) / phi'(t) over [0, 1]. Frank and Joe: pyvinecopulib.

    For the signed families (Clayton, Gumbel, Gumbel (180 degrees), Joe) ``theta`` is the signed parameter:
    a negative value is the turned branch with parameter ``abs(theta)`` and Kendall's tau is negative.

    Parameters:
    theta (np.ndarray or float): The single copula parameter (signed for the signed families).
    family (str): Family name.

    Returns:
    np.ndarray: Kendall's tau for each parameter value given.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    if family in ('gaussian', 'student'):
        return (2.0 / np.pi) * np.arcsin(theta)
    sign = np.where(theta >= 0, 1.0, -1.0)
    a = np.abs(theta)
    if family == 'clayton':
        return sign * a / (a + 2.0)
    if family in ('gumbel', 'gumbel180'):        # the 180-degree rotation leaves Kendall's tau unchanged
        return sign * (1.0 - 1.0 / a)
    if family in ('frank', 'joe'):
        cop = pv.Bicop(family=FAMILY_INFO[family]['pv_fam'])
        # pyvinecopulib converts one parameter vector per call and returns a float, so loop over theta.
        if family == 'frank':
            return np.array([float(cop.parameters_to_tau(np.array([[t]]))) for t in theta])
        return np.array([sg * float(cop.parameters_to_tau(np.array([[t]]))) for sg, t in zip(sign, a)])
    raise ValueError(
        f"theta_to_tau: unhandled family '{family}'; add a branch rather "
        f"than letting this fall through silently."
    )


def _calibration_bootstrap_one(bi, idx, u1, u2, x, x_eval, h, family,
                                fixed_params, beta_init_grid):
    """One bootstrap replicate of the confidence-band loop in calibration_bootstrap.

    Defined at module level so that ProcessPoolExecutor can pickle it. The resample indices are drawn
    sequentially by the caller before dispatch, not here.

    Parameters:
    bi (int): Replicate index, returned unchanged so the caller can place
        this result back into tau_boot[bi] regardless of completion order.
    idx (np.ndarray): This replicate's precomputed resample indices.
    u1, u2, x (np.ndarray): Full pseudo-observation / covariate series.
    x_eval (np.ndarray): Covariate points the curve is evaluated at.
    h (float): Kernel bandwidth.
    family (str): Family name.
    fixed_params (list): This family's fixed (non-primary) parameters from fit_global.
    beta_init_grid (list): Per-x_eval warm-start beta values from the point-estimate fit.

    Returns:
    tuple: (bi, tau_row), where tau_row has shape (len(x_eval),) and is NaN at every x_eval point whose fit failed.
    """
    u1b, u2b, xb = u1[idx], u2[idx], x[idx]
    tau_row = np.full(len(x_eval), np.nan)

    g_inv = FAMILY_INFO[family]['link_inv']
    for j, x0 in enumerate(x_eval):
        try:
            beta = fit_at_point(u1b, u2b, xb, x0, h, family, fixed_params,
                                 beta_init=beta_init_grid[j])
            tau_row[j] = float(theta_to_tau(g_inv(beta[0]), family)[0])
        except Exception:
            pass
    return bi, tau_row


def calibration_bootstrap(u1, u2, x, family, h, x_eval, ci_boot=30, seed=42, block=False, b=None, n_jobs=1,
                          all_idx=None, progress=True):
    """Local-likelihood calibration curve on a given grid and its bootstrap replicates.

    The point estimate is the local-linear fit at every grid point. Each replicate refits the calibration model on a resample of
    the rows, with the family's fixed parameters and the start values taken from the point estimate. The resample is i.i.d.,
    or a stationary bootstrap (Politis & Romano, 1994) if block is True, or given by the caller through all_idx.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations.
    u2 (np.ndarray): Second variable's pseudo-observations.
    x (np.ndarray): Covariate values.
    family (str): Family name.
    h (float): Kernel bandwidth.
    x_eval (np.ndarray): Grid of covariate values.
    ci_boot (int): Number of bootstrap replicates. Default 30. Ignored if all_idx is given.
    seed (int): Bootstrap seed. Default 42. Ignored if all_idx is given.
    block (bool): Resample with the stationary bootstrap of mean block length b instead of i.i.d. Default False.
    b (float): Mean block length, required if block is True. Optional.
    n_jobs (int): The replicates run in parallel via ProcessPoolExecutor when n_jobs > 1. All resample indices are drawn
        sequentially before dispatch, so a parallel run reproduces a sequential run. The point estimate is always
        computed sequentially. Default 1.
    all_idx (list): One index array into u1, u2 and x per replicate, used instead of drawing the resamples. Optional.
    progress (bool): Show a progress bar over the replicates. Default True.

    Returns:
    tuple: (theta_hat, tau_hat, tau_boot), where theta_hat and tau_hat have shape (len(x_eval),) and tau_boot has shape
        (number of replicates, len(x_eval)). A grid point whose local fit fails is NaN in the estimate, and in each replicate
        that fails there. A RuntimeWarning gives the number of failed fits and the fewest valid replicates at a grid point;
        the number of valid replicates per grid point is np.isfinite(tau_boot).sum(axis=0).

    Raises:
    ValueError: If block is True and no block length b is given.
    """
    if all_idx is None and block and b is None:
        raise ValueError("block=True requires a block length b (use the "
                          "same b as the purge gap / MCS bootstrap).")
    n_eval = len(x_eval)
    g_inv = FAMILY_INFO[family]['link_inv']
    g_fwd = FAMILY_INFO[family]['link_fwd']
    primary, fixed_params, _ = fit_global(u1, u2, family)
    global_beta_init = np.array([float(g_fwd(primary)), 0.0])
    theta_hat = np.full(n_eval, np.nan)
    beta_init_grid = []
    for j, x0 in enumerate(x_eval):
        try:
            beta = fit_at_point(u1, u2, x, x0, h, family, fixed_params,
                                 beta_init=global_beta_init)
            theta_hat[j] = float(g_inv(beta[0]))
            beta_init_grid.append(beta)
        except Exception:
            beta_init_grid.append(None)
    tau_hat = theta_to_tau(theta_hat, family)

    # All resample indices are drawn sequentially before dispatch (cheap), so that a parallel run uses the
    # same replicates as a sequential one.
    if all_idx is None:
        rng = np.random.default_rng(seed)
        n = len(u1)
        all_idx = []
        for bi in range(ci_boot):
            if block:
                idx = stationary_bootstrap_indices(n, b, rng)
            else:
                idx = rng.integers(0, n, size=n)
            all_idx.append(idx)
    n_rep = len(all_idx)
    tau_boot = np.full((n_rep, n_eval), np.nan)

    n_jobs = (1 if n_jobs == 1 else resolve_pool_workers(
        n_tasks=n_rep, max_workers=n_jobs,
        worker_gib=float(os.getenv('THESIS_ACAR_WORKER_GIB', '2.0'))))
    if n_jobs == 1:
        for bi in (tqdm(range(n_rep), desc='Bootstrap CI', leave=False) if progress else range(n_rep)):
            tau_boot[bi] = _calibration_bootstrap_one(bi, all_idx[bi], u1, u2, x, x_eval, h, family,
                                                      fixed_params, beta_init_grid)[1]
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_calibration_bootstrap_one, bi, all_idx[bi], u1, u2, x, x_eval,
                                 h, family, fixed_params, beta_init_grid): bi
                for bi in range(n_rep)
            }
            done = as_completed(futures)
            if progress:
                done = tqdm(done, total=n_rep, desc=f'Bootstrap CI (n_jobs={n_jobs})')
            for future in done:
                bi, tau_row = future.result()
                tau_boot[bi] = tau_row

    n_fail_point = int(np.isnan(theta_hat).sum())
    n_valid = np.isfinite(tau_boot).sum(axis=0)
    n_fail_boot = int(tau_boot.size - n_valid.sum())
    if n_fail_point or n_fail_boot:
        warnings.warn(f"calibration curve ({family}): the local fit failed at {n_fail_point} of {n_eval} grid points and in "
                      f"{n_fail_boot} of {tau_boot.size} replicate fits (fewest valid replicates at a grid point: "
                      f"{int(n_valid.min())} of {n_rep}). Failed fits are NaN, and the percentile band uses the remaining "
                      f"replicates.", RuntimeWarning, stacklevel=2)
    return theta_hat, tau_hat, tau_boot


def estimate_calibration_curve(u1, u2, x, family, h,
                                n_eval=40, ci_boot=30,
                                seed=42, x_eval=None,
                                block=False, b=None, n_jobs=1):
    """Estimate the local-likelihood calibration curve theta(w) and tau(w) with a bootstrap confidence band.

    The curve and the replicates come from calibration_bootstrap; the band is the 2.5th and 97.5th percentile of the replicates.
    Failed local fits are NaN, are reported in a RuntimeWarning, and the column n_boot gives the number of valid replicates at
    each grid point.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations.
    u2 (np.ndarray): Second variable's pseudo-observations.
    x (np.ndarray): Covariate values.
    family (str): Family name.
    h (float): Kernel bandwidth.
    n_eval (int): Number of covariate points to evaluate the curve at.
        Default 40. Used only if x_eval is None.
    ci_boot (int): Number of bootstrap replicates for the confidence
        interval. Default 30.
    seed (int): Bootstrap seed. Default 42.
    x_eval (np.ndarray): Covariate points to evaluate at. If None, an
        evenly-spaced grid between the 2nd and 98th percentile of x is
        used. Optional.
    block (bool): If True, the bootstrap resamples with the stationary bootstrap of Politis and Romano
        (1994) (circular geometric blocks of mean length b) instead of i.i.d., because u1, u2 and x are
        resampled jointly as one chronologically contiguous series. Use the same b as for the purge gap
        and the MCS bootstrap. The cost is dominated by ci_boot and n_eval (the number of local-likelihood
        refits), not by the resampling scheme. Default False (i.i.d.).
    b (float): Mean block length, required if block=True. Optional.
    n_jobs (int): The bootstrap replicates run in parallel via ProcessPoolExecutor when n_jobs > 1. All
        resample indices are drawn sequentially before dispatch, so a parallel run reproduces the replicates of a
        sequential run with the same seed. The point-estimate curve is always computed sequentially.
        Default 1.

    Returns:
    pd.DataFrame: Columns x, tau, tau_lo, tau_hi (2.5th and 97.5th bootstrap percentile), theta, n_boot (valid replicates
        at the grid point) and on_boundary (always False; kept for interface stability). attrs['boundary_frac'] is always
        0.0.

    Raises:
    ValueError: If block is True and no block length b is given.
    """
    if block and b is None:
        raise ValueError("block=True requires a block length b (use the "
                          "same b as the purge gap / MCS bootstrap).")

    if x_eval is None:
        x_eval = np.linspace(np.percentile(x, 2), np.percentile(x, 98), n_eval)
    n_eval = len(x_eval)
    theta_hat, tau_hat, tau_boot = calibration_bootstrap(u1, u2, x, family, h, x_eval, ci_boot=ci_boot, seed=seed,
                                                         block=block, b=b, n_jobs=n_jobs)

    if ci_boot > 0:
        with warnings.catch_warnings():
            # a grid point without a valid replicate is NaN and has already been reported by calibration_bootstrap
            warnings.simplefilter("ignore", RuntimeWarning)
            tau_lo = np.nanpercentile(tau_boot, 2.5, axis=0)
            tau_hi = np.nanpercentile(tau_boot, 97.5, axis=0)
    else:
        tau_lo = np.full(n_eval, np.nan)
        tau_hi = np.full(n_eval, np.nan)

    boundary_flags = np.zeros(n_eval, dtype=bool)
    out = {'x': x_eval, 'tau': tau_hat, 'tau_lo': tau_lo, 'tau_hi': tau_hi,
           'on_boundary': boundary_flags, 'theta': theta_hat, 'n_boot': np.isfinite(tau_boot).sum(axis=0)}
    df_out = pd.DataFrame(out)
    df_out.attrs['boundary_frac'] = float(boundary_flags.mean())
    return df_out


def acar_curve_boot(arrays, family, h, x_eval, ci_boot=200, seed=42, n_jobs=1, all_idx=None):
    """Point curve and bootstrap replicates of tau(w) on a given grid, which is needed to average over folds.

    The fit and the stationary bootstrap (fold block length b; Politis & Romano, 1994) are those of
    estimate_calibration_curve, which uses the same function (calibration_bootstrap), but the replicate
    matrix is returned instead of only its 2.5% and 97.5% percentiles. A grid point whose local fit fails is NaN in the
    curve and in the replicates that fail there.

    Parameters:
    arrays (dict): Fold arrays with the keys u1, u2, w, train (bool mask) and b (mean block length).
    family (str): Key of FAMILY_INFO.
    h (float): Fold bandwidth.
    x_eval (np.ndarray): Grid of w values.
    ci_boot (int): Bootstrap replicates. Default 200. Ignored if all_idx is given.
    seed (int): Bootstrap seed. Default 42. Ignored if all_idx is given.
    n_jobs (int): Worker processes for the replicates. Default 1.
    all_idx (list): One array of positions among the fold's training rows per replicate, used instead of drawing the
        resamples. Optional.

    Returns:
    tuple: (tau_hat of shape (len(x_eval),), tau_boot of shape (number of replicates, len(x_eval))).
    """
    tr = np.asarray(arrays["train"], bool)
    u1, u2 = np.asarray(arrays["u1"])[tr], np.asarray(arrays["u2"])[tr]
    x = np.asarray(arrays["w"], float)[tr]
    _, tau_hat, boot = calibration_bootstrap(u1, u2, x, family, float(h), x_eval, ci_boot=ci_boot, seed=seed, block=True,
                                             b=float(arrays["b"]), n_jobs=n_jobs, all_idx=all_idx, progress=False)
    return np.asarray(tau_hat, float), boot


def fold_averaged_curve(ARR, ACAR, folds, family, n_eval=30, ci_boot=200, n_jobs=1, seed=42):
    """tau(w) of one Acar family averaged over the folds, with a 95% band.

    The curve is the mean of the fold curves on a common grid (the overlap of the folds' 2%-98% training ranges of w).

    For the band, replicate r is one stationary bootstrap draw (Politis & Romano, 1994) over the whole series, shared by all
    folds. Fold k refits on the rows of that draw that lie in its training set. Because the training sets of the blocked
    folds overlap, the fold curves of a replicate are correlated as they are in the data, and the band of the average is not
    understated, as it would be if every fold were resampled independently. The mean block length of the shared draw is the
    mean of the folds' block lengths b. The replicate's fold curves are averaged, and the 2.5% and 97.5% percentiles are
    taken over r. The fold-averaged boxplots (plotting_core.plot_family_comparison) follow the same scheme.

    Parameters:
    ARR (dict): Fold name -> fold arrays (see fold_pipeline.fold_arrays); all folds are over the same series.
    ACAR (dict): (fold, family) -> Acar result of the fold_models artifact; the key 'h' holds the bandwidth.
    folds (iterable): Fold names.
    family (str): Key of FAMILY_INFO.
    n_eval (int): Number of grid points. Default 30.
    ci_boot (int): Bootstrap replicates. Default 200.
    n_jobs (int): Worker processes per fold. Default 1.
    seed (int): Seed of the shared bootstrap draws. Default 42.

    Returns:
    pd.DataFrame: Columns x, tau, tau_lo, tau_hi and n_boot (replicates with a valid fold mean at the grid point). Failed
        local fits are NaN and are skipped in the means, and calibration_bootstrap reports their number
        in a RuntimeWarning for every fold.

    Raises:
    ValueError: If the folds are not over series of the same length.
    """
    folds = list(folds)
    lo = max(np.percentile(np.asarray(ARR[f]["w"], float)[np.asarray(ARR[f]["train"], bool)], 2) for f in folds)
    hi = min(np.percentile(np.asarray(ARR[f]["w"], float)[np.asarray(ARR[f]["train"], bool)], 98) for f in folds)
    x_eval = np.linspace(lo, hi, n_eval)
    lengths = {len(ARR[f]["w"]) for f in folds}
    if len(lengths) != 1:
        raise ValueError(f"the folds are over series of different lengths: {sorted(lengths)}")
    n = lengths.pop()
    rng = np.random.default_rng(seed)
    b = float(np.mean([ARR[f]["b"] for f in folds]))
    draws = [stationary_bootstrap_indices(n, b, rng) for _ in range(ci_boot)]
    taus, boots = [], []
    for f in tqdm(folds, desc=f'Acar {family}: folds', leave=False):
        tr = np.asarray(ARR[f]["train"], bool)
        pos = np.cumsum(tr) - 1                                    # position of a series row among the fold's training rows
        idx_f = [pos[d[tr[d]]] for d in draws]                     # the rows of the shared draw that lie in the training set
        t, bt = acar_curve_boot(ARR[f], family, ACAR[(f, family)]["h"], x_eval, n_jobs=n_jobs, all_idx=idx_f)
        taus.append(t); boots.append(bt)
    with warnings.catch_warnings():
        # a grid point without a valid fit is NaN and has already been reported by calibration_bootstrap
        warnings.simplefilter("ignore", RuntimeWarning)
        tau = np.nanmean(np.vstack(taus), axis=0)
        avg = np.nanmean(np.stack(boots), axis=0)                  # (ci_boot, n_eval): replicate-wise fold mean
        band_lo, band_hi = np.nanpercentile(avg, [2.5, 97.5], axis=0)
    return pd.DataFrame(dict(x=x_eval, tau=tau, tau_lo=band_lo, tau_hi=band_hi, n_boot=np.isfinite(avg).sum(axis=0)))
