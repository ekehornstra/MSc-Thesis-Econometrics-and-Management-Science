"""fold_models.py

Per-fold model objects: the parametric fold fits with their per-step CVPE losses, the stored state of trained
networks and the predictor wrappers used in the comparison.

The CVPE loss of a step is the sum of the squared errors of E[U1 | U2] and E[U2 | U1] under the fitted copula (the
cross-validated prediction error criterion of Acar et al., 2011), computed by the shared quadrature of
copula_families (QUAD_N Gauss-Legendre nodes). fit_parametric_fold fits every parametric candidate on the training rows of one fold: the
unconditional families and, per divergence regime, the same families, from which the binary-regime pairs are
formed. The selected pair is chosen by training AIC only.

save_state writes a fitted CopulaNet, ConditionalCopulaNet or ConditionalCopulaFlow atomically, together with its
configuration, its normalisation attributes and a SHA-256 sidecar. load_state rebuilds the network and refuses a
missing or modified file (NotReconstructable). load_fitted loads the network of a job by its key, and inventory
reports the status of a set of jobs. State files are unpickled by torch.load, so they must come from a trusted
source (the checksum guards against corruption, not against tampering).

make_codine_predictor and make_acar_predictor return callables predictor(u1, u2) -> (e1, e2) that give the
conditional expectations used by the CVPE.

References:
Acar, E. F., Craiu, R. V., & Yao, F. (2011). Dependence calibration in conditional copulas: A nonparametric
    approach. Biometrics, 67(2), 445-453.
"""
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from copula_families import (DEFAULT_FAMILIES, QUAD_N, copula_conditional_expectation, copula_expectations,
                          expectations_pair, fit_and_evaluate, safe_make_bicop)


def copula_step_loss(cop, u1, u2):
    """Per-step CVPE loss of a fitted bivariate copula.

    The loss is the sum of the squared errors of E[U1 | U2] and E[U2 | U1], each computed at the observed values by
    the shared quadrature (copula_families).

    Parameters:
    cop (pyvinecopulib.Bicop): The fitted copula.
    u1 (np.ndarray): First variable's pseudo-observations, shape (n,).
    u2 (np.ndarray): Second variable's pseudo-observations, shape (n,).

    Returns:
    np.ndarray: Loss per observation, shape (n,).
    """
    u1 = np.asarray(u1, float); u2 = np.asarray(u2, float)
    e1, e2 = copula_expectations(cop, u1, u2)
    return (u1 - e1) ** 2 + (u2 - e2) ** 2


def fit_parametric_fold(u1, u2, train, test, high_mask, families=None, verbose=False):
    """Fit all parametric candidates for one fold and return the per-step losses.

    Parameters:
    u1 (np.ndarray): First variable's pseudo-observations over the full usable series (the kernel-conditional PIT).
    u2 (np.ndarray): Second variable's pseudo-observations over the full usable series.
    train (np.ndarray): Boolean mask of the training rows.
    test (np.ndarray): Boolean mask of the test rows.
    high_mask (np.ndarray): True for the high-divergence regime (fold-specific).
    families (list): Family keys. Default DEFAULT_FAMILIES.
    verbose (bool): Print the fitting progress. Default False.

    Returns:
    dict: With the keys
        'loss': {model name: (n,) array, NaN outside the test rows}, with the names 'Uncond:<Fam>' and
            'Binary:<FamLow>|<FamHigh>' (low regime first).
        'aic_uncond': {Fam: training AIC}.
        'aic_regime': {'low': {Fam: AIC}, 'high': {Fam: AIC}}.
        'rows_regime': {'low': rows, 'high': rows} (the rows of fit_and_evaluate, with the fitted copula).
        'rows_uncond': the rows of the unconditional fit.
        'aic_pair': {(FamLow, FamHigh): AIC of low plus AIC of high}.
        'selected_pair': the pair with the lowest training AIC (never chosen by test loss).
    """
    families = families or DEFAULT_FAMILIES
    n = len(u1)
    train = np.asarray(train, bool); test = np.asarray(test, bool)
    low = ~np.asarray(high_mask, bool); high = np.asarray(high_mask, bool)
    out = {'loss': {}, 'aic_uncond': {}, 'aic_regime': {'low': {}, 'high': {}},
           'rows_regime': {}, 'rows_uncond': None, 'aic_pair': {}}

    _, rows_u = fit_and_evaluate(u1, u2, train_mask=train, test_mask=test, families=families, quiet=not verbose)
    out['rows_uncond'] = rows_u
    for r in rows_u:
        loss = np.full(n, np.nan)
        loss[test] = copula_step_loss(r['cop'], u1[test], u2[test])
        out['loss'][f"Uncond:{r['name']}"] = loss
        out['aic_uncond'][r['name']] = r['aic']

    reg_loss = {'low': {}, 'high': {}}
    for rname, rmask in (('low', low), ('high', high)):
        _, rows = fit_and_evaluate(u1, u2, train_mask=train & rmask, test_mask=test & rmask,
                                   families=families, quiet=not verbose)
        out['rows_regime'][rname] = rows
        for r in rows:
            loss = np.full(n, np.nan)
            sel = test & rmask
            loss[sel] = copula_step_loss(r['cop'], u1[sel], u2[sel])
            reg_loss[rname][r['name']] = loss
            out['aic_regime'][rname][r['name']] = r['aic']

    for fl, ll in reg_loss['low'].items():
        for fh, lh in reg_loss['high'].items():
            col = np.full(n, np.nan)
            col[test & low] = ll[test & low]
            col[test & high] = lh[test & high]
            out['loss'][f"Binary:{fl}|{fh}"] = col
            out['aic_pair'][(fl, fh)] = out['aic_regime']['low'][fl] + out['aic_regime']['high'][fh]
    if not out['aic_pair']:
        raise RuntimeError("no regime pair could be fitted")
    out['selected_pair'] = min(out['aic_pair'], key=out['aic_pair'].get)
    return out


# ── stored state of trained networks ───────────────────────────────────────
FORMAT = "thesis-net-state"
VERSION = 1
STATE_SUBDIR = "states"


class NotReconstructable(RuntimeError):
    """The fitted network of a job cannot be rebuilt: it has no state file, or the state file is invalid."""


# ── configuration introspection and construction ───────────────────────────
def _first_linear(module):
    """First nn.Linear layer of a module.

    Parameters:
    module (torch.nn.Module): The module to search.

    Returns:
    torch.nn.Linear: The first linear layer.
    """
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return m
    raise ValueError("no Linear layer found")


def _tensor_or_none(x):
    """Detached CPU copy of a tensor, or None.

    Parameters:
    x (torch.Tensor or None): The tensor.

    Returns:
    torch.Tensor or None: The copy.
    """
    return None if x is None else x.detach().cpu().clone()


def extract_config(obj):
    """Constructor arguments of a fitted network, read from the object itself.

    Parameters:
    obj: A CopulaNet, ConditionalCopulaNet or ConditionalCopulaFlow.

    Returns:
    tuple: (class name, config dict) such that build_from_config(class name, config) builds an untrained network of
        the same architecture.
    """
    import torch.nn as nn
    name = type(obj).__name__
    if name == "CopulaNet":
        lin = obj.net[0]
        drop = [m for m in obj.net if isinstance(m, nn.Dropout)][0]
        return name, dict(d=int(lin.in_features), hidden=int(lin.out_features), dropout=float(drop.p))
    if name == "ConditionalCopulaNet":
        mode = obj.conditioning_mode
        if mode == "film":
            hidden, drop = int(obj.linear1.out_features), float(obj.dropout1.p)
        else:
            hidden = int(obj.net[0].out_features)
            drop = float([m for m in obj.net if isinstance(m, nn.Dropout)][0].p)
        cfg = dict(d=int(obj.d), hidden=hidden, conditioning_mode=mode, dropout=drop)
        cfg["spline_knots"] = _tensor_or_none(getattr(obj, "spline_knots", None))
        return name, cfg
    if name == "ConditionalCopulaFlow":
        layer0 = obj.layers[0]
        knots = getattr(layer0, "spline_knots", None)
        cfg = dict(n_layers=int(obj.n_layers), hidden=int(_first_linear(layer0).out_features),
                   conditioning_mode=obj.conditioning_mode, coupling_type=obj.coupling_type,
                   spline_knots=_tensor_or_none(knots))
        if obj.coupling_type == "affine":
            cfg["scale_bound"] = float(layer0.scale_bound)
        else:
            cfg.update(num_bins=int(layer0.num_bins), tail_bound=float(layer0.tail_bound),
                       min_derivative=float(layer0.min_derivative), max_derivative=float(layer0.max_derivative))
        return name, cfg
    raise TypeError(f"unsupported network class {name!r}")


def build_from_config(cls, cfg):
    """Build an untrained network from the output of extract_config.

    Parameters:
    cls (str): "CopulaNet", "ConditionalCopulaNet" or "ConditionalCopulaFlow".
    cfg (dict): Constructor arguments.

    Returns:
    torch.nn.Module: The network.
    """
    cfg = dict(cfg)
    if cls == "CopulaNet":
        from codine import CopulaNet
        return CopulaNet(**cfg)
    if cls == "ConditionalCopulaNet":
        from codine import ConditionalCopulaNet
        return ConditionalCopulaNet(**cfg)
    if cls == "ConditionalCopulaFlow":
        from conditional_copula_flow import ConditionalCopulaFlow
        return ConditionalCopulaFlow(**cfg)
    raise ValueError(f"unknown class {cls!r}")


_EXTRA_ATTRS = ("_w_mean", "_w_std", "_w_grid", "_norm_grid", "_norm_const")


def _extras(obj):
    """Non-parameter attributes of a network that its state_dict does not hold (standardisation and normalisation).

    Parameters:
    obj: A fitted network.

    Returns:
    dict: {attribute: value} for those of _EXTRA_ATTRS that the network has.
    """
    out = {}
    for a in _EXTRA_ATTRS:
        if hasattr(obj, a):
            v = getattr(obj, a)
            out[a] = np.asarray(v) if isinstance(v, (list, tuple, np.ndarray)) else v
    return out


# ── save and load ───────────────────────────────────────────────────────────
def state_path(batch_dir, key):
    """Path of the state file of a job.

    Parameters:
    batch_dir (str or Path): Batch directory.
    key (str): Job key.

    Returns:
    pathlib.Path: <batch_dir>/states/<key>.pt.
    """
    return Path(batch_dir) / STATE_SUBDIR / f"{key}.pt"


def _sha256(path):
    """SHA-256 checksum of a file.

    Parameters:
    path (str or Path): The file.

    Returns:
    str: Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_state(obj, path, meta=None):
    """Write a fitted network atomically (temporary file, fsync, rename), with a checksum sidecar.

    Parameters:
    obj: A fitted network.
    path (str or Path): Target state file. The sidecar is the same path with the suffix .state.json.
    meta (dict): Metadata stored in the file and the sidecar. Optional. fold_pipeline passes the fingerprint of the
        code that trained the network as meta['code_version'], which is how the file is tied to a code version.

    Returns:
    dict: The sidecar contents (format, version, class, sha256, size, library versions, creation time, meta).
    """
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cls, cfg = extract_config(obj)
    payload = dict(format=FORMAT, version=VERSION, cls=cls, config=cfg,
                   state_dict={k: v.detach().cpu() for k, v in obj.state_dict().items()},
                   extras=_extras(obj), meta=dict(meta or {}))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    side = dict(format=FORMAT, version=VERSION, cls=cls, sha256=_sha256(path), bytes=path.stat().st_size,
                torch=torch.__version__, numpy=np.__version__, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                meta=_jsonable(meta or {}))
    stmp = path.with_suffix(".state.json.tmp")
    stmp.write_text(json.dumps(side, indent=1))
    os.replace(stmp, path.with_suffix(".state.json"))
    return side


def _jsonable(o):
    """Convert an object to JSON-serialisable types (anything else becomes its repr).

    Parameters:
    o: The object.

    Returns:
    The converted object.
    """
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def verify_state(path):
    """Check a state file against the checksum in its sidecar.

    Parameters:
    path (str or Path): The state file.

    Returns:
    tuple: (ok, message).
    """
    path = Path(path)
    side = path.with_suffix(".state.json")
    if not path.exists():
        return False, "state file absent"
    if not side.exists():
        return False, "sidecar (checksum) absent"
    try:
        want = json.loads(side.read_text())["sha256"]
    except Exception as exc:
        return False, f"sidecar unreadable: {exc}"
    got = _sha256(path)
    return (got == want), ("ok" if got == want else "checksum mismatch")


def load_state(path, verify=True):
    """Rebuild a fitted network in eval mode from its state file.

    The payload is read with torch.load(weights_only=False), because it holds numpy arrays and other non-tensor
    attributes next to the weights. This unpickles arbitrary objects, so only load state files that you wrote yourself.

    Parameters:
    path (str or Path): The state file.
    verify (bool): Check the checksum first. Default True.

    Returns:
    tuple: (network, meta).

    Raises:
    NotReconstructable: If the file is missing or invalid, or has an unsupported format or version.
    """
    import torch
    path = Path(path)
    if verify:
        ok, msg = verify_state(path)
        if not ok:
            raise NotReconstructable(f"{path.name}: {msg}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT or payload.get("version", 0) > VERSION:
        raise NotReconstructable(f"{path.name}: unsupported format/version {payload.get('format')} v{payload.get('version')}")
    net = build_from_config(payload["cls"], payload["config"])
    net.load_state_dict(payload["state_dict"], strict=True)
    for k, v in payload["extras"].items():
        setattr(net, k, v)
    net.eval()
    return net, payload["meta"]


def load_fitted(batch_dir, key):
    """Fitted network of a job, loaded from its state file.

    A job without a valid state file is reported as not reconstructable, and a network is never re-trained silently.

    Parameters:
    batch_dir (str or Path): Batch directory.
    key (str): Job key.

    Returns:
    tuple: (network, meta).

    Raises:
    NotReconstructable: If the job has no state file, or the file is invalid.
    """
    p = state_path(batch_dir, key)
    if not p.exists():
        raise NotReconstructable(f"{key}: no state file for this job (fitted weights were not stored)")
    return load_state(p)


def inventory(batch_dir, keys):
    """Status of the stored network of each job.

    Parameters:
    batch_dir (str or Path): Batch directory.
    keys (list): Job keys.

    Returns:
    pd.DataFrame: One row per key, with the status 'state file', 'state file INVALID (...)' or
        'no state file'.
    """
    import pandas as pd
    rows = []
    for k in keys:
        p = state_path(batch_dir, k)
        if p.exists():
            ok, msg = verify_state(p)
            status = "state file" if ok else f"state file INVALID ({msg})"
        else:
            status = "no state file"
        rows.append(dict(key=k, status=status))
    return pd.DataFrame(rows)


# ── predictors ──────────────────────────────────────────────────────────────
def make_codine_predictor(net, grid_size=None):
    """Predictor factory for a fitted unconditional CODINE network.

    The conditional expectations are computed by Gauss-Legendre quadrature (copula_families) of the network's own
    density, as for the parametric families.

    Parameters:
    net (CopulaNet): A trained and normalisation-calibrated network.
    grid_size (int): Number of Gauss-Legendre nodes. Default copula_families.QUAD_N.

    Returns:
    callable: predictor(u1_arr, u2_arr) -> (e1, e2).
    """
    from codine import eval_density
    grid_size = QUAD_N if grid_size is None else int(grid_size)
    dens = lambda pts, _w: eval_density(net, pts)
    def predictor(u1_arr, u2_arr):
        """Conditional means (e1, e2) of the network's copula at the observed pairs.

        Parameters:
        u1_arr (array-like): First coordinate of the observed pairs.
        u2_arr (array-like): Second coordinate of the observed pairs.

        Returns:
        tuple: (e1, e2), the expectation of each coordinate given the other.
        """
        return expectations_pair(dens, np.asarray(u1_arr, float), np.asarray(u2_arr, float), n=grid_size)
    return predictor


def make_acar_predictor(fam_name, df_curve_train_by_family, fixed_params_train_by_family, w_test,
                        max_fallback_frac=0.01):
    """Predictor factory for one continuous-conditioning Acar family.

    The fitted calibration curve theta(w) is interpolated at the w of each test point, the copula (with the family's
    rotation) is built there and its conditional expectation is evaluated. For the Student-t the degrees of freedom
    are the fixed training value, clipped to [2.001, 49]. A point whose copula cannot be built or evaluated gets the
    uninformative prediction (0.5, 0.5). Every such point is counted and reported with a warning, and if more than
    max_fallback_frac of the points fall back the predictor raises, so a broken fit cannot enter the MCS with a loss
    that is partly a constant.

    Parameters:
    fam_name (str): Acar family key, one of the keys of conditional_copula.FAMILY_INFO ('gaussian', 'student',
        'clayton', 'gumbel', 'gumbel180', 'frank', 'joe').
    df_curve_train_by_family (dict): {family: DataFrame with columns 'x' and 'theta'} of fitted calibration curves.
    fixed_params_train_by_family (dict): {family: fixed (not interpolated) parameters}, for example the Student-t
        degrees of freedom.
    w_test (np.ndarray): Covariate values of the test batch.
    max_fallback_frac (float): Largest tolerated share of fallback points before raising. Default 0.01.

    Returns:
    callable: predictor(u1_arr, u2_arr) -> (e1, e2). The number of fallback points of the last call is stored in
        predictor.n_fallback.
    """
    from copula_families import interp_theta_col

    df_curve = df_curve_train_by_family[fam_name]
    fixed_params_train = fixed_params_train_by_family[fam_name]

    def predictor(u1_arr, u2_arr):
        """Conditional means (e1, e2) at the observed pairs, with the (0.5, 0.5) fallback described above.

        Parameters:
        u1_arr (array-like): First coordinate of the observed pairs.
        u2_arr (array-like): Second coordinate of the observed pairs.

        Returns:
        tuple: (e1, e2), the expectation of each coordinate given the other.
        """
        n = len(u1_arr)
        e1 = np.empty(n); e2 = np.empty(n)
        construction_fallback = 0
        eval_fallback_types = {}              # exception class name -> count

        for i in range(n):
            theta_i = float(interp_theta_col(fam_name, w_test[i], df_curve['x'].values, df_curve['theta'].values))   # signed; interpolated in the continuous excess scale
            try:
                if fam_name == 'student':
                    cop = safe_make_bicop(fam_name, [theta_i, float(np.clip(fixed_params_train[0], 2.001, 49.0))])
                else:
                    cop = safe_make_bicop(fam_name, [theta_i])
            except RuntimeError:
                construction_fallback += 1
                e1[i] = 0.5; e2[i] = 0.5
                continue
            try:
                E1_fn, E2_fn = copula_conditional_expectation(cop)
                e1[i] = E1_fn(float(u2_arr[i])); e2[i] = E2_fn(float(u1_arr[i]))
            except Exception as e:
                eval_fallback_types[type(e).__name__] = eval_fallback_types.get(type(e).__name__, 0) + 1
                e1[i] = 0.5; e2[i] = 0.5

        n_fb = construction_fallback + sum(eval_fallback_types.values())
        predictor.n_fallback = n_fb
        if n_fb:
            msg = (f"[Acar (continuous): {fam_name}] {n_fb}/{n} points fell back to (0.5, 0.5): "
                   f"construction {construction_fallback}, evaluation by type {eval_fallback_types}")
            if n_fb > max_fallback_frac * n:
                raise RuntimeError(msg + f" (more than {max_fallback_frac:.0%} of the points)")
            warnings.warn(msg)
        return e1, e2
    predictor.n_fallback = 0
    return predictor
