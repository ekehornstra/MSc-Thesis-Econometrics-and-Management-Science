"""The batched scoring paths give the same numbers as the one-model-at-a-time paths (to floating-point summation order)."""
import os
import numpy as np

os.environ.setdefault("THESIS_ALLOW_UNCALIBRATED_IMPORT", "1")
import ces_score as CES
import gijbels_copula as GJ

K = 16


def _masses(rng):
    """Random cell masses of a K x K discrete copula.

    Parameters:
    rng (np.random.Generator): Random number generator.

    Returns:
    np.ndarray: K * K masses summing to one.
    """
    m = rng.dirichlet(np.ones(K * K))
    return m


def test_es_from_masses_many_matches_single():
    """The batched energy score equals the score of each model computed alone."""
    rng = np.random.default_rng(0)
    obs = rng.random((500, 2))
    ms = [_masses(rng) for _ in range(5)]
    many = CES.es_from_masses_many(ms, obs, K)
    for m, x in zip(ms, many):
        assert np.allclose(CES.es_from_masses(m, obs, K), x, rtol=0, atol=1e-12)


def test_ces_conditional_both_matches_two_calls():
    """The raw and calibrated scores of ces_conditional_both equal two separate calls, with half the mass evaluations."""
    rng = np.random.default_rng(1)
    w = rng.normal(size=400); obs = rng.random((400, 2))
    base = [_masses(rng) for _ in range(9)]
    grid = np.linspace(w.min(), w.max(), 9)
    calls = []

    def mf(wv):
        """Masses of the nearest grid point, with effective sample size 1.

        Parameters:
        wv (float): Covariate value.

        Returns:
        tuple: (masses, 1.0).
        """
        calls.append(wv)
        return base[int(np.argmin(np.abs(grid - wv)))], 1.0

    d_a = {}; d_b = {}
    raw = CES.ces_conditional(mf, w, obs, grid, K, calibrated=False, diagnostics=d_a)
    cal = CES.ces_conditional(mf, w, obs, grid, K, calibrated=True)
    n_two = len(calls); calls.clear()
    raw2, cal2 = CES.ces_conditional_both(mf, w, obs, grid, K, diagnostics=d_b)
    assert len(calls) * 2 == n_two
    assert np.allclose(raw, raw2, rtol=0, atol=1e-12) and np.allclose(cal, cal2, rtol=0, atol=1e-12)
    assert d_a.keys() == d_b.keys()
    for k in d_a:
        assert np.allclose(d_a[k], d_b[k])


def test_pointwise_on_grid_both_matches_two_calls():
    """pointwise_on_grid_both equals the raw and the calibrated call of pointwise_on_grid."""
    rng = np.random.default_rng(2)
    w = rng.normal(size=300); o1, o2 = rng.random(300), rng.random(300)
    grid = np.linspace(w.min(), w.max(), 7)
    base = [_masses(rng) for _ in grid]
    mf = lambda wv: base[int(np.argmin(np.abs(grid - wv)))]
    r0 = GJ.pointwise_on_grid(mf, w, o1, o2, grid, K, calibrated=False)
    c0 = GJ.pointwise_on_grid(mf, w, o1, o2, grid, K, calibrated=True)
    r1, c1 = GJ.pointwise_on_grid_both(mf, w, o1, o2, grid, K)
    for a, b in ((r0, r1), (c0, c1)):
        for k in ("loss", "es"):
            assert np.allclose(a[k], b[k], rtol=0, atol=1e-12, equal_nan=True)


def test_tail_dep_at_qs_matches_single_calls():
    """The tail dependence at several thresholds equals the single-threshold calls for the same seed."""
    import pyvinecopulib as pv
    import copula_families as CF
    for fam, par in ((pv.BicopFamily.clayton, [2.0]), (pv.BicopFamily.gaussian, [0.5]), (pv.BicopFamily.gumbel, [1.8])):
        cop = pv.Bicop(fam, 0, np.array([par]).reshape(-1, 1))
        qs = [0.9, 0.95, 0.99]
        many = CF._tail_dep_at_qs(cop, qs, n_sim=5000, seed=7)
        for q in qs:
            assert many[q] == CF._tail_dep_at_q(cop, q=q, n_sim=5000, seed=7)


def test_glrt_null_does_not_depend_on_worker_count():
    """The parametric null of the GLRT is the same whether the replicates run in one call or are split over workers."""
    import acar_glrt as AG
    rng = np.random.default_rng(5)
    n = 120
    x = rng.uniform(size=n); u1 = rng.uniform(size=n); u2 = np.clip(0.5 * u1 + 0.5 * rng.uniform(size=n), 1e-3, 1 - 1e-3)
    a = AG.null_parametric(u1, u2, x, "gaussian", seed=3, reps=range(6))
    b = np.concatenate([AG.null_parametric(u1, u2, x, "gaussian", seed=3, reps=r) for r in ([0, 1, 2], [3], [4, 5])])
    assert np.array_equal(a, b)


def test_t_design_noise_is_independent_of_covariate_stream():
    """The noise of the Student-t design is independent of the stream that draws the covariate."""
    import synthetic_study as SS
    spec = next(s for s in (SS.DGPS.values() if isinstance(SS.DGPS, dict) else SS.DGPS) if s["family"] == "t")
    u1, u2, w, _ = SS.draw_dgp(4000, spec, 11)
    rng = np.random.default_rng(11); w0 = rng.uniform(*SS.W_RANGE, size=4000)
    assert np.array_equal(w, w0)
    from scipy.stats import norm
    # a noise stream seeded like the covariate (default_rng(11)) would share its raw words with w: its first normal draws are a function of the same raw words as w
    z = np.random.default_rng(11).standard_normal((4000, 2))[:, 0]
    assert abs(np.corrcoef(norm.ppf(np.clip(u1, 1e-6, 1 - 1e-6)), z)[0, 1]) < 0.1


def test_lsm_path_accepts_both_file_names(tmp_path):
    """The land-sea mask is found as lsm.nc or as lsm_sfc_<date>.nc, and lsm.nc takes precedence.

    Parameters:
    tmp_path (pathlib.Path): Temporary folder provided by pytest.
    """
    import run_support as RS
    assert RS.lsm_path(tmp_path).name == "lsm.nc"
    (tmp_path / "lsm_sfc_20200901.nc").write_bytes(b"")
    assert RS.lsm_path(tmp_path).name == "lsm_sfc_20200901.nc"
    (tmp_path / "lsm.nc").write_bytes(b"")
    assert RS.lsm_path(tmp_path).name == "lsm.nc"
