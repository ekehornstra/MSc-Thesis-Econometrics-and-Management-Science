"""copula_families.py

Single source of truth for the seven parametric copula families of the thesis:

    Gaussian, Student-t, Clayton, Gumbel, Gumbel rotated by 180 degrees, Frank, Joe

The second half of the module holds the helpers built on the families: Gauss-Legendre quadrature of conditional
expectations, safe construction of fitted copulas, the CVPE of a fitted copula and the bootstrap summaries of
Kendall's tau and of tail dependence.

Every place that builds a ``pyvinecopulib.Bicop`` from a family must go through
:func:`make_bicop`, so the rotation of the rotated Gumbel can never be dropped by
a bare ``pv.Bicop(family=...)`` call (a bare call silently fits an unrotated
Gumbel to data whose dependence sits in the lower tail).

Keys
----
``'gaussian' 't' 'clayton' 'gumbel' 'gumbel180' 'frank' 'joe'``. The aliases
``'student'`` (= ``'t'``) and the display names (``'Gaussian'``, ``'t'``, ``'Clayton'``,
``'Gumbel'``, ``'Gumbel (180°)'``, ``'Frank'``, ``'Joe'``) are accepted by :func:`resolve`.

Signed families
---------------
Clayton, Gumbel, Gumbel (180°) and Joe can only describe tau >= 0 at their own rotation. Each of
them is therefore a *signed* family with ONE parameter: for tau >= 0 it uses its own rotation
(0° or 180°), for tau < 0 the rotation turned by 90° (90° or 270°) with the same parameter range.
At independence both branches coincide (Clayton theta -> 0, Gumbel/Joe theta -> 1), so the family is
continuous in tau and has the same number of parameters; the candidate count does not change.
Gaussian, Student-t and Frank describe both signs without a rotation.

The *signed primary parameter* ``theta_col`` used by the Acar calibration curves is the pyvinecopulib
parameter with the sign of the branch: ``+theta`` on the own rotation, ``-theta`` on the turned one.
(Independence is ``+-par0``: 0 for Clayton, 1 for Gumbel/Joe.) :func:`make_bicop_signed` and
:func:`theta_col_of` convert in both directions; :func:`fit_family` fits both branches and keeps the
one with the higher log-likelihood.

The Gumbel (180°) copula has density ``c_Gumbel(1-u, 1-v; theta)``. It shares the
parameter range ``[1, inf)``, the inverse link ``1 + exp(eta)`` and Kendall's tau
with the Gumbel copula; its tail dependence is lower, ``lambda_L = 2 - 2**(1/theta)``,
``lambda_U = 0``.

References:
Demarta, S., & McNeil, A. J. (2005). The t copula and related copulas. International Statistical Review, 73(1),
111-129.
Joe, H. (1997). Multivariate models and dependence concepts. Chapman and Hall.
Nelsen, R. B. (2006). An introduction to copulas (2nd ed.). Springer.
"""
import os
import logging
import re
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pyvinecopulib as pv
import torch
from scipy.stats import t as _t_dist
from tqdm.auto import tqdm

from data_split import regime_stationary_bootstrap_indices, stationary_bootstrap_indices, train_test_split_chrono
from run_support import resolve_max_workers, resolve_pool_workers

F = pv.BicopFamily
log = logging.getLogger("thesis")        # the logger configured by run_support.get_logger
_LOGGED_BRANCH_FAILURES = set()          # each distinct branch failure is logged once per process


@dataclass(frozen=True)
class FamilySpec:
    """Static description of one candidate family.

    Attributes:
    key (str): Machine key, e.g. ``'gumbel180'``.
    display (str): Name used in tables, figures and result rows.
    pv_family (pv.BicopFamily): pyvinecopulib family enum.
    rotation (int): 0 or 180.
    n_params (int): Number of copula parameters (2 for Student-t, else 1).
    symmetric (bool): Radially symmetric (unchanged by u -> 1-u).
    neg_rotation (int or None): Rotation used for tau < 0 (signed families), None if the family
        already covers both signs.
    par0 (float): Parameter value at independence (0 for Clayton, 1 for Gumbel and Joe).
    """
    key: str
    display: str
    pv_family: object
    rotation: int
    n_params: int
    symmetric: bool
    neg_rotation: object = None
    par0: float = 0.0


FAMILIES = OrderedDict((s.key, s) for s in (
    FamilySpec('gaussian',  'Gaussian',      F.gaussian, 0,   1, True),
    FamilySpec('t',         't',             F.student,  0,   2, True),
    FamilySpec('clayton',   'Clayton',       F.clayton,  0,   1, False, 90, 0.0),
    FamilySpec('gumbel',    'Gumbel',        F.gumbel,   0,   1, False, 90, 1.0),
    FamilySpec('gumbel180', 'Gumbel (180°)', F.gumbel,   180, 1, False, 270, 1.0),
    FamilySpec('frank',     'Frank',         F.frank,    0,   1, True),
    FamilySpec('joe',       'Joe',           F.joe,      0,   1, False, 90, 1.0),
))

FAMILY_KEYS = tuple(FAMILIES)                       # the seven keys, in the order used in all tables
N_FAMILIES = len(FAMILY_KEYS)                       # 7
FAMILY_DISPLAY = OrderedDict((k, s.display) for k, s in FAMILIES.items())

_ALIASES = {'student': 't', 'student-t': 't', 'gumbel_180': 'gumbel180',
            'gumbel (180°)': 'gumbel180', 'gumbel(180)': 'gumbel180', 'gumbel (180)': 'gumbel180',
            'rotated gumbel': 'gumbel180'}
_BY_DISPLAY = {s.display.lower(): k for k, s in FAMILIES.items()}


def resolve(name):
    """Return the :class:`FamilySpec` for a key, alias or display name (case-insensitive).

    Parameters:
    name (str or FamilySpec): Family key, alias or display name; a FamilySpec is returned unchanged.

    Returns:
    FamilySpec: The matching family description.

    Raises:
    KeyError: If the name is not one of the seven families (never a silent default).
    """
    if isinstance(name, FamilySpec):
        return name
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    key = _BY_DISPLAY.get(key, key)
    if key not in FAMILIES:
        raise KeyError(f"unknown copula family {name!r}; expected one of {list(FAMILY_KEYS)} "
                       f"(or their display names {list(FAMILY_DISPLAY.values())})")
    return FAMILIES[key]


def make_bicop(name, parameters=None, rotation=None):
    """Build a ``pv.Bicop`` for a family WITH its rotation.

    Parameters:
    name (str): Key, alias or display name accepted by :func:`resolve`.
    parameters (array-like): Parameter vector in pyvinecopulib's own order
        (``[rho, nu]`` for the Student-t, ``[theta]`` otherwise). If None, an unfitted
        copula of that family is returned (to be fitted with ``.fit``). Optional.
    rotation (int): Override of the rotation (only the family's own rotation or, for signed
        families, its ``neg_rotation`` are allowed). Default: the family's own rotation. Optional.

    Returns:
    pv.Bicop: The copula, fitted or unfitted depending on ``parameters``.
    """
    s = resolve(name)
    rot = s.rotation if rotation is None else int(rotation)
    if rot not in _allowed_rotations(s):
        raise ValueError(f"make_bicop: rotation {rot} is not allowed for {s.display} (allowed: {_allowed_rotations(s)})")
    if parameters is None:
        return pv.Bicop(family=s.pv_family, rotation=rot)
    p = np.asarray(parameters, float).reshape(-1, 1)
    return pv.Bicop(family=s.pv_family, rotation=rot, parameters=p)


def _allowed_rotations(s):
    """Rotations a family may take: its own and, for signed families, the turned one.

    Parameters:
    s (FamilySpec): Family description.

    Returns:
    tuple: Allowed rotations in degrees.
    """
    return (s.rotation,) if s.neg_rotation is None else (s.rotation, s.neg_rotation)


def is_signed(name):
    """True for the families that switch to a turned rotation when tau < 0 (Clayton, Gumbel, Gumbel (180°), Joe).

    Parameters:
    name (str): Family key, alias or display name.

    Returns:
    bool: Whether the family is a signed family.
    """
    return resolve(name).neg_rotation is not None


def make_bicop_signed(name, parameters):
    """Build a ``pv.Bicop`` from a SIGNED primary parameter (see the module docstring).

    Parameters:
    name (str): Family key, alias or display name.
    parameters (array-like): ``[theta_col]`` or, for the Student-t, ``[rho, nu]``. For a signed family a
        negative first value selects the turned rotation with parameter ``abs(theta_col)``; for the
        other families the vector is used as it is.

    Returns:
    pv.Bicop: The copula with the rotation implied by the sign of the first parameter.
    """
    s = resolve(name)
    p = np.asarray(parameters, float).ravel().copy()
    rot = s.rotation
    if s.neg_rotation is not None:
        if p[0] < 0:
            rot = s.neg_rotation
        p[0] = abs(p[0])
    return pv.Bicop(family=s.pv_family, rotation=rot, parameters=p.reshape(-1, 1))


def theta_col_of(cop):
    """Signed primary parameter of a fitted ``pv.Bicop`` (inverse of :func:`make_bicop_signed` for one-parameter families).

    Parameters:
    cop (pv.Bicop): A fitted copula.

    Returns:
    float: The first parameter, negative if the copula is on the turned branch of a signed family.
    """
    s = FAMILIES[key_of(cop)]
    th = float(np.asarray(cop.parameters, float).ravel()[0])
    return -th if (s.neg_rotation is not None and int(cop.rotation) == s.neg_rotation) else th


def signed_params_of(cop):
    """Full parameter vector of a fitted copula with the sign of the branch on the first entry.

    Parameters:
    cop (pv.Bicop): A fitted copula.

    Returns:
    np.ndarray: Parameter vector whose first entry is ``theta_col_of(cop)``.
    """
    p = np.asarray(cop.parameters, float).ravel().copy()
    p[0] = theta_col_of(cop)
    return p


def branch_of(cop):
    """'own' or 'turned' for a signed family (turned = rotation + 90 degrees, i.e. tau < 0), '-' for Gaussian,
    Student-t and Frank, which need no turned branch.

    Parameters:
    cop (pv.Bicop): A fitted copula.

    Returns:
    str: 'own', 'turned' or '-'.
    """
    s = FAMILIES[key_of(cop)]
    if s.neg_rotation is None:
        return '-'
    return 'turned' if int(cop.rotation) == s.neg_rotation else 'own'


def theta_col_to_eta(name, theta_col):
    """Continuous signed excess of ``theta_col``: ``sign * (abs(theta_col) - par0)`` for signed families
    (0 at independence), the value itself for the others. Interpolate in this scale, never in ``theta_col``
    (which jumps from -par0 to +par0 at independence for Gumbel and Joe).

    Parameters:
    name (str): Family key, alias or display name.
    theta_col (array-like): Signed primary parameter value(s).

    Returns:
    np.ndarray: The signed excess over independence.
    """
    s = resolve(name)
    th = np.asarray(theta_col, float)
    if s.neg_rotation is None:
        return th
    return np.where(th >= 0, th - s.par0, -(np.abs(th) - s.par0))


def eta_to_theta_col(name, eta):
    """Inverse of :func:`theta_col_to_eta` (``eta >= 0`` selects the own rotation).

    Parameters:
    name (str): Family key, alias or display name.
    eta (array-like): Signed excess over independence.

    Returns:
    np.ndarray: Signed primary parameter value(s).
    """
    s = resolve(name)
    e = np.asarray(eta, float)
    if s.neg_rotation is None:
        return e
    return np.where(e >= 0, s.par0 + e, -(s.par0 + np.abs(e)))


def interp_theta_col(name, x, xs, theta_cols):
    """Linear interpolation of a signed calibration curve at ``x``, done in the continuous excess scale.

    Parameters:
    name (str): Family key, alias or display name.
    x (array-like): Covariate values to interpolate at.
    xs (array-like): Covariate grid of the curve, increasing.
    theta_cols (array-like): Signed primary parameter on that grid.

    Returns:
    np.ndarray: Signed primary parameter at ``x``.
    """
    return eta_to_theta_col(name, np.interp(x, np.asarray(xs, float), theta_col_to_eta(name, theta_cols)))


def fit_family(name, data):
    """Fit a family by maximum likelihood on both branches (own rotation and, for signed families, the
    turned rotation) and return the fitted copula with the higher log-likelihood.

    A branch that fails to fit, or returns a non-finite log-likelihood, is skipped. Each distinct failure is logged
    once per process, and the reasons are included in the error raised when neither branch can be fitted.

    Parameters:
    name (str): Family key, alias or display name.
    data (np.ndarray): (n, 2) pseudo-observations.

    Returns:
    pv.Bicop: fitted copula (its ``rotation`` tells which branch won).
    """
    s = resolve(name)
    data = np.asarray(data, float)
    best = None
    failures = []
    for rot in _allowed_rotations(s):
        cop = pv.Bicop(family=s.pv_family, rotation=rot)
        try:
            cop.fit(data)
            ll = float(cop.loglik(data))
        except Exception as exc:
            failures.append(f"{s.display} rotation {rot}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if not np.isfinite(ll):
            failures.append(f"{s.display} rotation {rot}: non-finite log-likelihood")
        elif best is None or ll > best[0]:
            best = (ll, cop)
    for msg in failures:
        if msg not in _LOGGED_BRANCH_FAILURES:
            _LOGGED_BRANCH_FAILURES.add(msg)
            log.warning(f"fit_family: branch skipped ({msg})")
    if best is None:
        raise RuntimeError(f"fit_family: {s.display} could not be fitted on either branch ({'; '.join(failures)})")
    return best[1]


def key_of(cop):
    """Family key of a fitted ``pv.Bicop`` (reads family AND rotation).

    Parameters:
    cop (pv.Bicop): A fitted copula.

    Returns:
    str: One of the seven family keys.
    """
    rot = int(cop.rotation)
    for k, s in FAMILIES.items():
        if s.pv_family == cop.family and rot in _allowed_rotations(s):
            return k
    raise KeyError(f"Bicop family/rotation {cop.family}/{rot} is not one of the seven candidate families")


def sample_bicop(cop, n, seed=None):
    """Draw ``n`` pairs from a ``pv.Bicop`` (``sample`` in pyvinecopulib >= 0.7, else ``simulate``).

    Parameters:
    cop (pv.Bicop): The copula to sample from.
    n (int): Number of pairs.
    seed (int): Seed passed to pyvinecopulib. Default None (no seed).

    Returns:
    np.ndarray: (n, 2) sample.
    """
    seeds = [] if seed is None else [int(seed)]
    fn = getattr(cop, 'sample', None) or cop.simulate
    return fn(n, seeds=seeds)


def tail_deps_from_params(name, parameters, rotation=None):
    """Closed-form (lambda_L, lambda_U) of a family at given parameters.

    Rotation-aware: the 180° rotation swaps the two coefficients, so Gumbel (180°) has
    ``lambda_L = 2 - 2**(1/theta)`` and ``lambda_U = 0``.

    Sources: Nelsen, R. B. (2006). An introduction to copulas (2nd ed.). Springer (Clayton: lambda_L = 2**(-1/theta);
    Gumbel: lambda_U = 2 - 2**(1/theta); Gaussian and Frank: 0). Joe, H. (1997). Multivariate models and dependence
    concepts. Chapman and Hall (Joe: lambda_U = 2 - 2**(1/theta)). Demarta, S., & McNeil, A. J. (2005). The t copula
    and related copulas. International Statistical Review, 73(1), 111-129 (Student-t, eq. 13).

    Parameters:
    name (str): Family key, alias or display name.
    parameters (array-like): Parameter vector (``[rho, nu]`` for the Student-t, ``[theta]`` otherwise).
    rotation (int): Rotation of the copula. On the turned branch of a signed family (tau < 0)
        the dependence sits in the mixed corners, so both coefficients of the main diagonal are 0. Optional.

    Returns:
    tuple: (lower, upper)
    """
    s = resolve(name)
    if rotation is not None and s.neg_rotation is not None and int(rotation) == s.neg_rotation:
        return 0.0, 0.0
    p = np.asarray(parameters, float).ravel()
    if s.key in ('gaussian', 'frank'):
        lo = up = 0.0
    elif s.key == 't':
        rho = float(np.clip(p[0], -0.9999, 0.9999)); nu = float(p[1])
        lo = up = float(2 * _t_dist.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1))
    elif s.key == 'clayton':
        lo, up = float(2.0 ** (-1.0 / p[0])), 0.0
    elif s.key in ('gumbel', 'joe'):
        lo, up = 0.0, float(2.0 - 2.0 ** (1.0 / p[0]))
    elif s.key == 'gumbel180':
        lo, up = float(2.0 - 2.0 ** (1.0 / p[0])), 0.0      # mirror image of Gumbel
    else:                                                    # pragma: no cover - resolve() already rejects
        raise KeyError(s.key)
    return lo, up


# ──────────────────────────────────────────────────────────────────────────
# Quadrature, CVPE and bootstrap summaries of fitted copulas
# ──────────────────────────────────────────────────────────────────────────
QUAD_N = 100          # nodes per conditional expectation (Gauss-Legendre on [0, 1])
MAX_DEGENERATE_FRAC = 0.05   # largest tolerated share of points whose density integrates to ~0


@lru_cache(maxsize=None)
def nodes(n=QUAD_N):
    """Gauss-Legendre nodes and weights mapped to [0, 1] (float64).

    Parameters:
    n (int): Number of nodes. Default QUAD_N.

    Returns:
    tuple: (nodes, weights), two np.ndarray of length n.
    """
    x, w = np.polynomial.legendre.leggauss(int(n))
    return (x + 1.0) / 2.0, w / 2.0


def conditional_expectation(density, fixed, fixed_dim, w=None, n=QUAD_N, chunk=64):
    """E[U_target | U_fixed_dim = fixed_i (, w_i)] for every i, by Gauss-Legendre quadrature.

    density(pts, w_rep) -> (m,) values of the density at pts of shape (m, 2); w_rep repeats w_i once per
    node (None when the density does not depend on a covariate). fixed_dim is the coordinate that is held
    fixed (0 or 1); the other one is integrated. Returns an (len(fixed),) array; 0.5 where the density
    integrates to ~0.

    That 0.5 is an uninformative placeholder, not a prediction. Every such point is counted and
    reported with a warning, and the call raises if more than MAX_DEGENERATE_FRAC of the points are
    affected, so a collapsed density cannot pass as a model with a constant loss.

    Parameters:
    density (callable): density(pts, w_rep) -> (m,) values at pts of shape (m, 2).
    fixed (array-like): Values of the fixed coordinate, one per prediction.
    fixed_dim (int): The coordinate that is held fixed (0 or 1).
    w (array-like): Covariate value per prediction, or None if the density does not depend on one. Optional.
    n (int): Number of quadrature nodes. Default QUAD_N.
    chunk (int): Number of predictions evaluated per density call. Default 64.

    Returns:
    np.ndarray: Conditional expectation for each fixed value.
    """
    fixed = np.asarray(fixed, np.float32)
    w = None if w is None else np.asarray(w, np.float32)
    g, wt = nodes(n)
    g32 = g.astype(np.float32)
    out = np.empty(len(fixed))
    n_degenerate = 0
    for a in range(0, len(fixed), chunk):
        fa = fixed[a:a + chunk]
        m = len(fa)
        fixed_col = np.repeat(fa, n)
        grid_col = np.tile(g32, m)
        pts = (np.column_stack([fixed_col, grid_col]) if fixed_dim == 0
               else np.column_stack([grid_col, fixed_col]))
        wr = None if w is None else np.repeat(w[a:a + chunk], n)
        dens = np.maximum(np.asarray(density(pts, wr), float), 0.0).reshape(m, n)
        norm = dens @ wt
        num = (dens * g[None, :]) @ wt
        ok = norm > 1e-12
        n_degenerate += int((~ok).sum())
        out[a:a + m] = np.where(ok, num / np.where(ok, norm, 1.0), 0.5)
    if n_degenerate:
        msg = (f"conditional_expectation: the density integrates to ~0 at {n_degenerate} of {len(fixed)} points; "
               f"their prediction is the placeholder 0.5")
        if n_degenerate > MAX_DEGENERATE_FRAC * len(fixed):
            raise RuntimeError(msg + f" (more than {MAX_DEGENERATE_FRAC:.0%} of the points)")
        warnings.warn(msg)
    return out


def expectations_pair(density, u1, u2, w=None, n=QUAD_N, chunk=64):
    """(E[U1 | U2 = u2_i], E[U2 | U1 = u1_i]) for all i; the two CVPE predictions of one model.

    Parameters:
    density (callable): density(pts, w_rep) -> (m,) values at pts of shape (m, 2).
    u1 (array-like): First coordinate of each test point.
    u2 (array-like): Second coordinate of each test point.
    w (array-like): Covariate value per test point, or None. Optional.
    n (int): Number of quadrature nodes. Default QUAD_N.
    chunk (int): Number of predictions evaluated per density call. Default 64.

    Returns:
    tuple: (e1, e2), two np.ndarray with E[U1 | U2 = u2_i] and E[U2 | U1 = u1_i].
    """
    e1 = conditional_expectation(density, u2, 1, w, n, chunk)
    e2 = conditional_expectation(density, u1, 0, w, n, chunk)
    return e1, e2


def _spline_basis_w(w: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Cubic truncated-power spline basis expansion of a covariate.

    Shared by conditional_copula_flow.py (conditioning_mode='spline_basis'
    on the affine/spline coupling layers) and codine.py
    (conditioning_mode='spline_basis' on ConditionalCopulaNet); defined here
    so that neither module has to import it from the other.

    [1, w, w^2, w^3, (w-k_1)_+^3, ..., (w-k_K)_+^3], locally supported
    around each knot rather than globally-affecting like a trig feature,
    so a wiggle near one knot doesn't distort the fit far away in w.

    knots must be QUANTILES of the training w distribution, not evenly
    spaced: some +/-0.5 windows in w hold fewer than 30 real observations
    (notebook 04), so evenly spaced knots risk instability in sparse
    regions. Quantile knots put each spline segment on comparable amounts
    of real data instead.

    Source: Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning (2nd ed.).
    Springer (truncated power basis, Section 5.2).

    Parameters:
    w (torch.Tensor): Standardised covariate values, shape (...,).
    knots (torch.Tensor): Fixed interior knot positions, shape (K,).

    Returns:
    torch.Tensor: Shape (..., 4 + K).
    """
    feats = [torch.ones_like(w), w, w ** 2, w ** 3]
    for k in knots:
        feats.append(torch.clamp(w - k, min=0.0) ** 3)
    return torch.stack(feats, dim=-1)

def _safe_bicop(pv_fam, params, margin=1e-3, rotation=0):
    """Construct a pv.Bicop from a full parameter vector (pyvinecopulib's own
    row order for this family), nudging whichever individual row is just
    past its numerical bound. A locally-fit parameter can under/overflow
    past pv's floor/ceiling for one row while the rest of the vector is
    fine (e.g. Clayton theta ~1e-12 vs pv's 1e-10 floor) -- clamp only
    that row rather than assume the violating parameter is always first.

    Parameters:
    pv_fam (pv.BicopFamily): Copula family.
    params (list): Parameter values, in pyvinecopulib's own row order for
        this family.
    margin (float): Fractional nudge applied to a parameter just past its
        bound. Default 1e-3.
    rotation (int): Copula rotation (0 or 180). Must be passed for Gumbel (180°); prefer
        ``safe_make_bicop(key, params)``, which looks the rotation up from the family key.

    Returns:
    pv.Bicop: The constructed copula object.
    """
    params = list(params)
    def build(p):
        """Build the copula of the family, rotation and parameter list p.

        Parameters:
        p (sequence of float): Parameters in pyvinecopulib order.

        Returns:
        pv.Bicop: The copula.
        """
        return pv.Bicop(family=pv_fam, rotation=rotation, parameters=np.array([[v] for v in p]))
    try:
        return build(params)
    except RuntimeError as e:
        msg = str(e)
        bound_block = re.search(r'bound:\s*\n((?:[\-0-9.eE]+\s*\n?)+)actual:', msg)
        actual_block = re.search(r'actual:\s*\n((?:[\-0-9.eE]+\s*\n?)+)', msg)
        if not bound_block or not actual_block:
            raise
        bounds = [float(x) for x in bound_block.group(1).split()]
        actuals = [float(x) for x in actual_block.group(1).split()]
        if len(bounds) != len(params) or len(actuals) != len(params):
            raise
        fixed = list(params)
        for i, (b, a) in enumerate(zip(bounds, actuals)):
            if a < b:
                fixed[i] = b * (1 + margin) if b > 0 else b - abs(b) * margin - margin
            elif a > b:
                fixed[i] = b * (1 - margin) if b > 0 else b + abs(b) * margin + margin
        return build(fixed)


def safe_make_bicop(name, params, margin=1e-3):
    """``_safe_bicop`` for a family key / display name: family AND rotation come from the family specs of this module.

    Parameters:
    name (str): Family key, alias or display name.
    params (array-like): Parameter vector; for a signed family the sign of the first entry selects the turned rotation.
    margin (float): Fractional nudge applied to a parameter just past its bound. Default 1e-3.

    Returns:
    pv.Bicop: The constructed copula.
    """
    spec = resolve(name)
    p = list(np.asarray(params, float).ravel())
    rot = spec.rotation
    if spec.neg_rotation is not None:           # signed family: the sign of the first entry selects the branch
        if p[0] < 0:
            rot = spec.neg_rotation
        p[0] = abs(p[0])
    return _safe_bicop(spec.pv_family, p, margin=margin, rotation=rot)


def get_tail_deps(cop, name):
    """Read tail dependence coefficients analytically from copula parameters.

    Closed-form expressions for the elliptical and single-parameter Archimedean families follow
    Nelsen, R. B. (2006), An introduction to copulas, Springer. The 180-degree rotation swaps the
    two coefficients, so Gumbel (180 degrees) has lambda_L = 2 - 2**(1/theta) and lambda_U = 0.

    Parameters:
    cop (pv.Bicop): A fitted bivariate copula object.
    name (str): Family key, alias or display name accepted by ``resolve``
        ('Gaussian', 't', 'Clayton', 'Gumbel', 'Gumbel (180°)', 'Frank', 'Joe'). It must agree with
        the family AND rotation of ``cop``; a mismatch raises instead of returning the coefficients
        of the wrong tail.

    Returns:
    tuple: (lower_val, upper_val), the lower and upper tail dependence coefficients.

    Raises:
    KeyError: If `name` is not one of the seven families.
    ValueError: If `name` disagrees with ``cop.family`` / ``cop.rotation``.
    """
    spec = resolve(name)
    allowed = (spec.rotation,) if spec.neg_rotation is None else (spec.rotation, spec.neg_rotation)
    if cop.family != spec.pv_family or int(cop.rotation) not in allowed:
        raise ValueError(f"get_tail_deps: name {name!r} (family {spec.pv_family}, rotation {spec.rotation}) "
                         f"does not match the copula (family {cop.family}, rotation {int(cop.rotation)})")
    return tail_deps_from_params(spec.key, np.asarray(cop.parameters, float).ravel(), rotation=int(cop.rotation))

def copula_conditional_expectation(cop):
    """Scalar conditional-expectation functions of a copula (any object with a vectorised `.pdf`).

    Returns (E[U1 | U2 = u2], E[U2 | U1 = u1]) as closures taking one float, computed by the shared
    Gauss-Legendre rule of ``nodes``. For many points use copula_expectations, which batches them.

    Parameters:
    cop: Copula object with a vectorised ``pdf`` method.

    Returns:
    tuple: (E1, E2), two functions of one float.
    """
    dens = lambda pts, _w: cop.pdf(pts)
    E1 = lambda u2_val: float(conditional_expectation(dens, [u2_val], 1)[0])
    E2 = lambda u1_val: float(conditional_expectation(dens, [u1_val], 0)[0])
    return E1, E2


def copula_expectations(cop, u1, u2):
    """Vectorised (E[U1 | U2 = u2_i], E[U2 | U1 = u1_i]) of one copula for all i.

    Parameters:
    cop: Copula object with a vectorised ``pdf`` method.
    u1 (array-like): First coordinate of each point.
    u2 (array-like): Second coordinate of each point.

    Returns:
    tuple: (e1, e2), two np.ndarray.
    """
    return expectations_pair(lambda pts, _w: cop.pdf(pts), np.asarray(u1, float), np.asarray(u2, float))


def _cvpe_of_fitted(cop, u1_test, u2_test):
    """Mean CVPE (per-step squared error of predicting each coordinate from the other) of a fitted copula.

    Returns np.inf when the test set is empty or the expectation cannot be computed.

    Parameters:
    cop: Fitted copula object with a vectorised ``pdf`` method.
    u1_test (array-like): First coordinate of the test points.
    u2_test (array-like): Second coordinate of the test points.

    Returns:
    float: Mean CVPE over the test points.
    """
    u1_test = np.asarray(u1_test, float); u2_test = np.asarray(u2_test, float)
    if len(u1_test) == 0:
        return np.inf
    e1, e2 = copula_expectations(cop, u1_test, u2_test)
    return float(np.mean((u1_test - e1) ** 2 + (u2_test - e2) ** 2))


# Display names for every family this module knows about, keyed the same way ``resolve`` accepts them.
# 'student' is kept as an alias of 't' for the callers that use conditional_copula.FAMILY_INFO's vocabulary.
FAMILY_DISPLAY_NAMES = {**{k: d for k, d in FAMILY_DISPLAY.items()}, 'student': 't'}

# Default candidate set: all seven families (Gaussian, Student-t, Clayton, Gumbel, Gumbel (180°), Frank, Joe).
# BB1/BB6/BB7/BB8/Tawn were dropped from the real-data pipeline.
DEFAULT_FAMILIES = list(FAMILY_KEYS)


def fit_and_evaluate(u, v, train_mask=None, test_mask=None, test_frac=0.2,
                      purge=0, families=None, quiet=False, on_fail='raise'):
    """Fit a set of parametric copula families and evaluate each on AIC, BIC, and CVPE.

    Parameters:
    u (np.ndarray): First variable's pseudo-observations (full series).
    v (np.ndarray): Second variable's pseudo-observations (full series).
    train_mask (np.ndarray): If provided, used directly (e.g. loaded from
        the "preprocessing" artifact so every notebook shares the exact
        same split). Optional.
    test_mask (np.ndarray): As above. Optional. If either mask is not
        provided, both are computed internally from test_frac/purge --
        kept for backward compatibility with callers that don't have a
        shared split available. 
    test_frac (float): Used only if train_mask/test_mask are not provided.
        Default 0.2.
    purge (int): Used only if train_mask/test_mask are not provided.
        Default 0.
    families (list): Family keys / display names (``resolve``).
        Defaults to DEFAULT_FAMILIES, the seven families of the pipeline. Pass a restricted list for
        ad-hoc use only: the thesis pool needs all seven.
    quiet (bool): If True, no progress bar and no per-family printout
        (used by the fold loops, which call this hundreds of times).
        Default False.
    on_fail (str): 'raise' (default) raises RuntimeError if any requested family cannot be fitted,
        because a silently missing family would shrink the 74-candidate pool; 'warn' prints the
        failure and continues with the families that did fit.

    Returns:
    tuple: (best_aic, rows), where best_aic is the row dict for the
        lowest-AIC family and rows is the list of result dicts (keys: name, key, rotation, aic,
        bic, cvpe, cop, fam). ``fam`` is the pyvinecopulib enum (rotation is in ``rotation``).
    """
    data = np.column_stack([u, v])
    if families is None:
        families = DEFAULT_FAMILIES
    specs = [resolve(f) for f in families]
    n = len(u)
    if train_mask is None or test_mask is None:
        train_mask, test_mask = train_test_split_chrono(n, test_frac, purge=purge)
    rows, failed = [], []
    for spec in tqdm(specs, desc='fit_and_evaluate', leave=True, disable=quiet):
        try:
            cop = fit_family(spec.key, data[train_mask])          # both branches of a signed family; the better one is kept
            ll = cop.loglik(data[train_mask])
            aic  = -2 * ll + 2 * cop.npars
            bic  = -2 * ll + np.log(int(train_mask.sum())) * cop.npars
            cvpe = _cvpe_of_fitted(cop, u[test_mask], v[test_mask])
            rows.append({'name': spec.display, 'key': spec.key, 'rotation': int(cop.rotation),
                         'aic': aic, 'bic': bic, 'cvpe': cvpe, 'cop': cop, 'fam': spec.pv_family})
            if not quiet:
                tqdm.write(f"  {spec.display}: AIC={aic:.2f}  CVPE={cvpe:.5f}")
        except Exception as e:
            failed.append((spec.display, repr(e)))
            tqdm.write(f"  {spec.display} failed: {e}")
    if failed and on_fail == 'raise':
        raise RuntimeError("fit_and_evaluate: family fit failed for " + "; ".join(f"{n_}: {m}" for n_, m in failed))
    if not rows:
        raise RuntimeError("fit_and_evaluate: no family could be fitted")
    best_aic = min(rows, key=lambda x: x['aic'])
    return best_aic, rows


def _bootstrap_one_replicate(bi, idx, uu1, uu2, fixed_family, fixed_family_name, q_list):
    """Fit one bootstrap replicate and compute its tau/tail-dependence.

    Top-level (picklable) function for ProcessPoolExecutor -- must not be
    nested or a lambda. idx is precomputed by the caller, sequentially and
    before any dispatch, rather than drawn here: drawing indices is cheap,
    and doing it first keeps every replicate identical between a
    sequential and a parallel run.

    Parameters:
    bi (int): Replicate index -- used only for _tail_dep_at_q's seed.
    idx (np.ndarray): This replicate's precomputed resample indices.
    uu1, uu2 (np.ndarray): Full pseudo-observation series.
    fixed_family (str): Family key / display name (``resolve``); the copula is built
        with ``make_bicop`` so the rotation of Gumbel (180°) is kept in every replicate.
    fixed_family_name (str): Family name, passed to get_tail_deps.
    q_list (list or None): Tail-dependence thresholds to simulate at, or
        None to use get_tail_deps's closed-form values instead.

    Returns:
    tuple: (bi, tau, lo_by_q, up_by_q) on success, where lo_by_q/up_by_q
        are dicts keyed by q (or {None: value} if q_list is None).
        Returns (bi, None, None, None) on failure; the caller counts and skips it.
    """
    try:
        cop = fit_family(fixed_family, np.column_stack([uu1[idx], uu2[idx]]))
        tau = float(cop.parameters_to_tau(cop.parameters))
        lo_by_q, up_by_q = {}, {}
        if q_list is not None:
            for qi, (lo, up) in _tail_dep_at_qs(cop, q_list, n_sim=50_000, seed=bi).items():
                lo_by_q[qi] = lo
                up_by_q[qi] = up
        else:
            lo, up = get_tail_deps(cop, fixed_family_name)
            lo_by_q[None] = lo
            up_by_q[None] = up
        return bi, tau, lo_by_q, up_by_q
    except Exception:
        return bi, None, None, None


def bootstrap_copula_stats_fast(uu1, uu2, fixed_family, fixed_family_name,
                                n_boot=200, q=None, block=False, b=None,
                                seed=None, max_workers=None, regime_mask=None, index_sets=None):
    """Bootstrap Kendall's tau and tail dependence for a fixed copula family.

    q can be a float or list of floats. If list, returns
    (taus, {q_val: (lowers, uppers)}). If float or None, returns
    (taus, lowers, uppers).

    Every replicate's resample indices are generated sequentially first
    (so the RNG stream does not depend on the number of workers), then the expensive part -- fitting the copula and
    computing tail dependence for each replicate -- runs in parallel via
    ProcessPoolExecutor. Results are reassembled in original replicate
    order, so output matches a sequential run exactly aside from
    wall-clock time.

    Parameters:
    uu1 (np.ndarray): First variable's pseudo-observations. If
        regime_mask is given, this must be the FULL (unmasked)
        chronological series, not pre-sliced to the regime -- see
        regime_mask below.
    uu2 (np.ndarray): As above.
    fixed_family (str): Family key / display name to refit on each replicate (rotation included).
    fixed_family_name (str): Family name, passed to get_tail_deps and shown in the progress bar.
    n_boot (int): Number of bootstrap replicates. Default 200.
    q (float): Tail-dependence threshold(s) to evaluate. A float, a list
        of floats, or None. Optional.
    block (bool): If True, resample via the Politis and Romano (1994)
        stationary bootstrap (circular geometric-length blocks, mean
        length `b`) instead of i.i.d. resampling. Required (and implied)
        when regime_mask is given. Only valid otherwise when uu1/uu2 are
        a single chronologically contiguous series (e.g. the full series
        or a contiguous train split) -- for a non-contiguous regime mask
        with regime_mask not passed, block=False (i.i.d.) is the fallback,
        but pass regime_mask instead wherever the data actually is a
        regime subset; see regime_mask below. Default False.
    b (float): Mean block length for the stationary bootstrap. Required
        if block=True or regime_mask is given. Use the same b as the
        purge gap -- do not re-derive a separate
        value here. Optional.
    seed (int): Optional int/Generator seed for the per-replicate rng when
        block=True, for reproducibility across reruns.
    max_workers (int): Concurrent worker processes. Default None: chosen by run_support.resolve_max_workers from the CPUs
        available to the job.
    regime_mask (np.ndarray): Boolean, same length as uu1/uu2, True for
        regime members. When given, uu1/uu2 are treated as the FULL
        chronological series and each replicate is drawn via
        data_split.regime_stationary_bootstrap_indices instead of a
        plain block/i.i.d. resample of a pre-sliced regime array -- see
        that function's docstring for why this matters (a block drawn
        from the full timeline can span both regime and non-regime
        timesteps, capturing real within-regime temporal clustering,
        e.g. divergence's diurnal cycle, that a resample of an
        already-masked array cannot represent). Requires block=True and
        b set. The point estimate this bootstrap is building a CI around
        (whatever was already fit outside this function) must itself
        have been fit on uu1[regime_mask]/uu2[regime_mask], not uu1/uu2
        directly -- this function only handles the resampled replicates.
        Default None (uu1/uu2 are used exactly as given).

    index_sets (list of np.ndarray): Optional precomputed full-timeline resamples (see ``shared_timeline_resamples``).
        Replicate r then uses index_sets[r] (restricted to regime_mask when given) instead of drawing its own, so
        several folds can be bootstrapped on the SAME resampled timeline and keep their dependence. n_boot is
        taken as len(index_sets); block, b and seed are ignored.

    Returns:
    tuple: (taus, lowers, uppers) if q is a float or None; (taus, by_q) if
        q is a list, where by_q maps each q value to (lowers, uppers).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if index_sets is not None:
        n_boot = len(index_sets)
    if index_sets is None and block and b is None:
        raise ValueError("block=True requires a block length b (use the "
                          "same b as the purge gap / MCS bootstrap).")
    if regime_mask is not None and index_sets is None:
        if not block:
            raise ValueError("regime_mask requires block=True -- a regime-aware "
                              "resample only makes sense as a block bootstrap of "
                              "the full timeline, not an i.i.d. resample.")
        if b is None:
            raise ValueError("regime_mask requires a block length b.")

    q_list   = None if q is None else ([q] if isinstance(q, float) else list(q))
    multi_q  = isinstance(q, list)

    n = len(uu1)
    rng = np.random.default_rng(seed)

    # Draw every replicate's resample indices sequentially before dispatch: this is cheap, and the RNG
    # stream stays independent of the number of workers while the fits run in parallel.
    all_idx = []
    for bi in range(n_boot):
        if index_sets is not None:
            idx_full = np.asarray(index_sets[bi])
            idx = idx_full if regime_mask is None else idx_full[regime_mask[idx_full]]
        elif regime_mask is not None:
            idx = regime_stationary_bootstrap_indices(n, regime_mask, b, rng)
        elif block:
            idx = stationary_bootstrap_indices(n, b, rng)
        else:
            idx = rng.integers(0, n, size=n)
        all_idx.append(idx)

    n_failed = 0
    taus_by_bi = {}
    lowers_by_q_bi = {qi: {} for qi in q_list} if q_list else None
    uppers_by_q_bi = {qi: {} for qi in q_list} if q_list else None
    lowers_by_bi, uppers_by_bi = {}, {}

    # CPU/task limit and RAM headroom are checked for each family bootstrap pool.
    max_workers = resolve_pool_workers(n_tasks=n_boot, max_workers=max_workers,
                                       worker_gib=float(os.getenv('THESIS_OTHER_WORKER_GIB', '1.5')))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_bootstrap_one_replicate, bi, all_idx[bi], uu1, uu2,
                            fixed_family, fixed_family_name, q_list)
            for bi in range(n_boot)
        ]
        for future in tqdm(as_completed(futures), total=n_boot,
                            desc=f'bootstrap {fixed_family_name}', leave=False):
            bi, tau, lo_by_q, up_by_q = future.result()
            if tau is None:
                n_failed += 1
                continue
            taus_by_bi[bi] = tau
            if q_list is not None:
                for qi in q_list:
                    lowers_by_q_bi[qi][bi] = lo_by_q[qi]
                    uppers_by_q_bi[qi][bi] = up_by_q[qi]
            else:
                lowers_by_bi[bi] = lo_by_q[None]
                uppers_by_bi[bi] = up_by_q[None]

    # Reassemble in original replicate order; failed replicates are skipped and counted in the warning below.
    # With shared resamples the replicate index must keep its meaning across folds, so a failed replicate stays in
    # place as NaN (the plotting code averages with nanmean) instead of shifting the later ones up.
    successful_bi = list(range(n_boot)) if index_sets is not None else sorted(taus_by_bi.keys())
    if index_sets is not None:
        _nan = float('nan')
        for _bi in successful_bi:
            if _bi not in taus_by_bi:
                taus_by_bi[_bi] = _nan
                if q_list is not None:
                    for _qi in q_list:
                        lowers_by_q_bi[_qi][_bi] = _nan; uppers_by_q_bi[_qi][_bi] = _nan
                else:
                    lowers_by_bi[_bi] = _nan; uppers_by_bi[_bi] = _nan
    if n_failed:
        warnings.warn(f"bootstrap {fixed_family_name}: {n_failed}/{n_boot} replicates failed to fit and are "
                      f"excluded from the interval")
    taus = np.array([taus_by_bi[bi] for bi in successful_bi])
    if multi_q:
        return taus, {qi: (np.array([lowers_by_q_bi[qi][bi] for bi in successful_bi]),
                            np.array([uppers_by_q_bi[qi][bi] for bi in successful_bi]))
                      for qi in q_list}
    if q_list:  # single float passed as list internally
        qi = q_list[0]
        lowers = np.array([lowers_by_q_bi[qi][bi] for bi in successful_bi])
        uppers = np.array([uppers_by_q_bi[qi][bi] for bi in successful_bi])
        return taus, lowers, uppers
    return taus, np.array([lowers_by_bi[bi] for bi in successful_bi]), \
           np.array([uppers_by_bi[bi] for bi in successful_bi])


def _tail_dep_at_q(cop, q=0.95, n_sim=50_000, seed=42):
    """Empirical tail dependence at a finite threshold q, from simulated draws.

    Simulates n_sim pairs from cop and computes the empirical exceedance
    probability at threshold q for both tails, matching the finite-q chi
    estimator used throughout the thesis rather than the
    theoretical q -> 1 limit. Symmetric families (Gaussian, Student-t,
    Frank) are averaged across both tails to reduce simulation noise.

    Source: Coles, S., Heffernan, J., & Tawn, J. (1999). Dependence measures for extreme value analyses.
    Extremes, 2(4), 339-365. The conditional probability P(V > u | U > u) is the quantity whose limit for u -> 1 defines their chi
    (Section 3.3.1). Their finite-u function chi(u) = 2 - log C(u, u) / log u is asymptotically equivalent but not identical,
    and is not what is computed here.

    Parameters:
    cop (pv.Bicop): A fitted bivariate copula object.
    q (float): Exceedance threshold for the upper tail; 1 - q is used for
        the lower tail. Default 0.95.
    n_sim (int): Number of simulated pairs. Default 50,000.
    seed (int): Simulation seed. Default 42.

    Returns:
    tuple: (lower, upper) empirical tail dependence coefficients.
    """
    samp = sample_bicop(cop, n_sim, seed=seed)
    u, v = samp[:, 0], samp[:, 1]

    denom_u = (v > q).sum()
    upper = float(((u > q) & (v > q)).sum() / denom_u) if denom_u > 0 else 0.0

    denom_l = (v < (1 - q)).sum()
    lower = float(((u < (1-q)) & (v < (1-q))).sum() / denom_l) if denom_l > 0 else 0.0

    symmetric = {pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.frank}
    if cop.family in symmetric:
        avg = (upper + lower) / 2
        return avg, avg

    return lower, upper


def _tail_dep_at_qs(cop, qs, n_sim=50_000, seed=42):
    """_tail_dep_at_q for several thresholds from one simulation.

    _tail_dep_at_q draws the same sample for the same seed whatever q is, so simulating once and reading off every
    threshold gives identical values.

    Parameters:
    cop (pv.Bicop): A fitted bivariate copula object.
    qs (sequence of float): Exceedance thresholds.
    n_sim (int): Number of simulated pairs. Default 50,000.
    seed (int): Simulation seed. Default 42.

    Returns:
    dict: {q: (lower, upper)}, as _tail_dep_at_q returns for each q.
    """
    samp = sample_bicop(cop, n_sim, seed=seed)
    u, v = samp[:, 0], samp[:, 1]
    symmetric = cop.family in {pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.frank}
    out = {}
    for q in qs:
        denom_u = (v > q).sum()
        upper = float(((u > q) & (v > q)).sum() / denom_u) if denom_u > 0 else 0.0
        denom_l = (v < (1 - q)).sum()
        lower = float(((u < (1 - q)) & (v < (1 - q))).sum() / denom_l) if denom_l > 0 else 0.0
        if symmetric:
            lower = upper = (upper + lower) / 2
        out[q] = (lower, upper)
    return out


def shared_timeline_resamples(n, b, n_boot, seed=0):
    """n_boot stationary-bootstrap resamples of the full timeline 0..n-1 (mean block length b).

    Passing the same list as ``index_sets`` to every fold's bootstrap makes all folds see the same resampled
    blocks, each fold keeping only the draws inside its own training mask. Fold estimates computed from one
    replicate are then dependent in the same way the real fold estimates are (overlapping training data).

    Parameters:
    n (int): Length of the full timeline.
    b (float): Mean block length of the stationary bootstrap.
    n_boot (int): Number of resamples.
    seed (int): Seed of the random number generator. Default 0.

    Returns:
    list: n_boot integer index arrays of length n.
    """
    rng = np.random.default_rng(seed)
    return [stationary_bootstrap_indices(n, b, rng) for _ in range(n_boot)]


def bootstrap_all_families(u1, u2, rows, n_boot=200, q=0.95,
                            block=False, b=None, seed=None, regime_mask=None, index_sets=None):
    """Bootstrap Kendall's tau and tail dependence for every fitted family in rows.

    q can be a float or list of floats. Per-q keys lowers_qXX / uppers_qXX
    are always populated. Legacy keys lowers / uppers use the first
    (lowest) q for backward compatibility.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations. If regime_mask
        is given, this must be the FULL (unmasked) chronological series
        -- see regime_mask below.
    u2 (np.ndarray): As above.
    rows (list): List of fitted-family result dicts, as returned by
        fit_and_evaluate (each needs 'name', 'key', 'cop' keys). Each
        row's 'cop' must already have been fit on the regime-restricted
        data if regime_mask is given -- this function only resamples,
        it does not refit the point estimate.
    n_boot (int): Number of bootstrap replicates per family. Default 200.
    q (float): Tail-dependence threshold(s). A float or list of floats.
        Default 0.95.
    block (bool): Forwarded to bootstrap_copula_stats_fast. Required
        (and implied) when regime_mask is given. Set True otherwise only
        when u1/u2 form a single chronologically contiguous series (e.g.
        the full/unconditional series or a contiguous train split).
        Default False.
    b (float): Forwarded to bootstrap_copula_stats_fast. Required if
        block=True or regime_mask is given. Optional.
    seed (int): Forwarded to bootstrap_copula_stats_fast. Optional.
    regime_mask (np.ndarray): Boolean, same length as u1/u2, True for
        regime members. When given, resamples via a block bootstrap of
        the full chronological timeline, keeping only in-regime draws
        (data_split.regime_stationary_bootstrap_indices), instead of
        treating the regime-restricted subsequence as already i.i.d. --
        see bootstrap_copula_stats_fast's docstring for the full
        rationale. Default None.

    index_sets (list of np.ndarray): Optional shared full-timeline resamples, forwarded to
        bootstrap_copula_stats_fast (see ``shared_timeline_resamples``).

    Returns:
    list: One dict per family in rows, with tau/lower/upper point
        estimates and their bootstrap distributions (legacy and per-q
        keyed).
    """
    q_list  = [q] if isinstance(q, float) else sorted(q)
    multi_q = len(q_list) > 1

    all_stats = []
    for r in tqdm(rows, desc='bootstrap_all_families', leave=True):
        if multi_q:
            taus, by_q = bootstrap_copula_stats_fast(
                u1, u2, r['key'], r['name'], n_boot=n_boot, q=q_list,
                block=block, b=b, seed=seed, regime_mask=regime_mask, index_sets=index_sets)
        else:
            taus, lows, ups = bootstrap_copula_stats_fast(
                u1, u2, r['key'], r['name'], n_boot=n_boot, q=q_list[0],
                block=block, b=b, seed=seed, regime_mask=regime_mask, index_sets=index_sets)
            by_q = {q_list[0]: (lows, ups)}

        tau_pt = float(r['cop'].parameters_to_tau(r['cop'].parameters))

        pt_by_q = _tail_dep_at_qs(r['cop'], q_list)
        entry = {
            'name': r['name'],
            'tau':  tau_pt,
            'taus': taus,
            'branch': branch_of(r['cop']),
            # share of bootstrap replicates that ended on the turned branch (tau < 0); NaN for Gaussian, t, Frank
            'turned_share': float(np.mean(np.asarray(taus)[np.isfinite(taus)] < 0)) if is_signed(r['key']) and np.isfinite(taus).any() else np.nan,
            # legacy keys — first q
            'lowers': by_q[q_list[0]][0],
            'uppers': by_q[q_list[0]][1],
            'lower':  float(pt_by_q[q_list[0]][0]),
            'upper':  float(pt_by_q[q_list[0]][1]),
        }
        for qi in q_list:
            qk = int(round(qi * 100))
            lo_pt, up_pt = pt_by_q[qi]
            entry[f'lowers_q{qk}'] = by_q[qi][0]
            entry[f'uppers_q{qk}'] = by_q[qi][1]
            entry[f'lower_q{qk}']  = float(lo_pt)
            entry[f'upper_q{qk}']  = float(up_pt)

        all_stats.append(entry)
    return all_stats
