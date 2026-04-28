from __future__ import annotations

import unittest

import numpy as np

from baseline_v2.reference_math import get_typical_value, get_typical_value_dominant_region


class RobustStep1CTests(unittest.TestCase):
    def test_dominant_region_rejects_ambiguous_balanced_bimodal_window(self):
        vals = np.concatenate(
            [
                np.full(20, -10.0),
                np.full(18, 10.0),
                np.array([-9.0, -10.0, 9.0, 10.0]),
            ]
        )
        mu_ref, sigma_ref = get_typical_value(vals, return_diagnostics=False)
        mu_robust, sigma_robust = get_typical_value_dominant_region(
            vals,
            return_diagnostics=False,
        )

        self.assertTrue(np.isfinite(mu_ref))
        self.assertTrue(np.isfinite(sigma_ref))
        self.assertTrue(np.isfinite(mu_robust))
        self.assertTrue(np.isnan(sigma_robust))

    def test_dominant_region_accepts_mode_with_long_tail(self):
        core = np.array(
            [-2.0, -1.8, -1.5, -1.3, -1.1, -1.0, -0.8, -0.7, -0.5, -0.3, 0.0, 0.2, 0.4, 0.5, 0.8, 1.0]
        )
        tail = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
        vals = np.concatenate([core, tail])

        _, sigma_ref = get_typical_value(vals, return_diagnostics=False)
        mu_robust, sigma_robust = get_typical_value_dominant_region(
            vals,
            return_diagnostics=False,
        )

        self.assertTrue(np.isfinite(mu_robust))
        self.assertTrue(np.isfinite(sigma_robust))
        self.assertTrue(np.isfinite(sigma_ref))
        self.assertLess(sigma_robust, sigma_ref)


if __name__ == "__main__":
    unittest.main()
