from __future__ import annotations

import unittest

import numpy as np

from baseline import CoordinateRotator


class CoordinateRotatorTests(unittest.TestCase):
    def test_recovers_declination_from_large_absolute_components(self):
        samples_per_day = 48
        n_samples = 19 * samples_per_day
        t = np.arange(
            np.datetime64("2026-01-01T00:00"),
            np.datetime64("2026-01-01T00:00") + n_samples * np.timedelta64(30, "m"),
            np.timedelta64(30, "m"),
        )

        expected_q = np.arctan2(3000.0, 7500.0)
        horizontal_field = np.hypot(7500.0, 3000.0)
        rng = np.random.default_rng(42)
        n = horizontal_field + rng.normal(0.0, 50.0, n_samples)
        e = rng.normal(0.0, 20.0, n_samples)
        x = n * np.cos(expected_q) - e * np.sin(expected_q)
        y = n * np.sin(expected_q) + e * np.cos(expected_q)
        z = np.zeros(n_samples)

        rotator = CoordinateRotator(t, x, y, z)
        rotator.rotate()

        q = rotator.df["q"].to_numpy(dtype=float)
        rotated_e = rotator.df["E"].to_numpy(dtype=float)

        self.assertAlmostEqual(
            np.degrees(np.nanmedian(q)),
            np.degrees(expected_q),
            delta=0.1,
        )
        self.assertLess(abs(np.nanmean(rotated_e)), 20.0)
        self.assertGreater(abs(np.nanmean(y)), 2500.0)

    def test_rejects_nonpositive_declination_bin_width(self):
        t = np.array([np.datetime64("2026-01-01T00:00")])
        values = np.ones(1)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            CoordinateRotator(
                t,
                values,
                values,
                values,
                declination_bin_width_degrees=0.0,
            )

        with self.assertRaisesRegex(ValueError, "must be positive"):
            CoordinateRotator(
                t,
                values,
                values,
                values,
                declination_bin_width_degrees=np.nan,
            )


if __name__ == "__main__":
    unittest.main()
