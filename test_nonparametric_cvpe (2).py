"""Regression checks for the common nonparametric CVPE scorer (Gijbels and ECBC)."""
import unittest

import numpy as np

import gijbels_copula as GJ


class TestNonparametricCVPE(unittest.TestCase):
    """Regression checks for the common nonparametric CVPE scorer (Gijbels and ECBC)."""
    def test_prediction_interpolated_before_squared_loss(self):
        """The conditional predictions are interpolated between grid points before the loss is squared."""
        K = 16
        c = (np.arange(K) + 0.5) / K
        m0 = np.outer(np.ones(K), np.ones(K)) / K ** 2
        m1 = np.eye(K) / K
        masses = {0.0: m0.ravel(), 1.0: m1.ravel()}
        w = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        o1 = np.array([0.3] * len(w)); o2 = np.array([c[12]] * len(w))
        res = GJ.pointwise_on_grid(lambda x: masses[x], w, o1, o2, np.array([0.0, 1.0]), K)
        p10, p20 = GJ.cvpe_predictions_from_masses(m0, o1, o2, K)
        p11, p21 = GJ.cvpe_predictions_from_masses(m1, o1, o2, K)
        want = (o1 - ((1 - w) * p10 + w * p11)) ** 2 + (o2 - ((1 - w) * p20 + w * p21)) ** 2
        np.testing.assert_allclose(res["loss"], want, atol=1e-14)
        np.testing.assert_allclose(res["loss"][[0, -1]], [GJ.cvpe_from_masses(m0, o1[:1], o2[:1], K)[0],
                                                        GJ.cvpe_from_masses(m1, o1[-1:], o2[-1:], K)[0]])
        old = (1 - w) * GJ.cvpe_from_masses(m0, o1, o2, K) + w * GJ.cvpe_from_masses(m1, o1, o2, K)
        self.assertGreater(old[2] - res["loss"][2], 0.01)

    def test_coarse_grid_and_calibration(self):
        """With K = 64 the loss and the energy score are finite, with and without calibration, and the uncalibrated loss equals cvpe_from_masses."""
        K = 64
        rng = np.random.default_rng(8)
        m = rng.uniform(size=(K, K)); m /= m.sum()
        w = np.array([0.1, 0.4, 0.9])
        u1 = np.array([0.02, 0.5, 0.98]); u2 = np.array([0.9, 0.3, 0.05])
        for calibrated in (False, True):
            res = GJ.pointwise_on_grid(lambda _: m.ravel(), w, u1, u2, np.array([0.0, 1.0]), K,
                                       calibrated=calibrated)
            self.assertTrue(np.isfinite(res["loss"]).all())
            self.assertTrue(np.isfinite(res["es"]).all())
            if not calibrated:
                np.testing.assert_allclose(res["loss"], GJ.cvpe_from_masses(m, u1, u2, K))

    def test_energy_score_still_interpolated_as_before(self):
        """The energy score is still interpolated linearly between the two bracketing grid points."""
        K = 16
        a = np.eye(K) / K; b = np.fliplr(a)
        w = np.array([0.2, 0.8]); o1 = np.array([0.1, 0.7]); o2 = np.array([0.9, 0.2])
        masses = {0.0: a.ravel(), 1.0: b.ravel()}
        res = GJ.pointwise_on_grid(lambda x: masses[x], w, o1, o2, np.array([0.0, 1.0]), K)
        import ces_score as CES
        obs = np.column_stack([o1, o2])
        want = (1 - w) * CES.es_from_masses(masses[0.0], obs, K) + w * CES.es_from_masses(masses[1.0], obs, K)
        np.testing.assert_allclose(res["es"], want)


if __name__ == "__main__":
    unittest.main()
