from __future__ import annotations

import unittest

import numpy as np

from baseline_v2.types import VarianceInputs
from baseline_v2.variance import ModernVarianceEngine, causal_cosine_memory_smooth

try:
    from baseline import VarianceEstimator
except Exception:  # pragma: no cover - optional compatibility check only
    VarianceEstimator = None


def brute_force_rolling_sum_of_squares(x: np.ndarray, window_size: int):
    x = np.asarray(x, dtype=float)
    ss = np.zeros(x.shape, dtype=float)
    count = np.zeros(x.shape, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window_size)
        window = x[start : i + 1]
        valid = np.isfinite(window)
        count[i] = float(np.sum(valid))
        if count[i] > 0:
            values = window[valid]
            ss[i] = float(np.sum((values - np.mean(values)) ** 2))
    return ss, count


def brute_force_causal_memory_smooth(x: np.ndarray, window_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lag = np.arange(1, window_size + 1, dtype=float)
    k = 1.0 / window_size
    kernel = k * (1.0 + np.cos(lag * np.pi * k))
    y = np.full(x.shape, np.nan, dtype=float)
    for i in range(len(x)):
        total = 0.0
        support = 0.0
        for lag_idx, weight in enumerate(kernel, start=1):
            src = i - lag_idx
            if src < 0:
                break
            if np.isfinite(x[src]):
                total += x[src] * weight
                support += 1.0
        if support > 0:
            y[i] = total
    return y


class ModernVarianceEngineTests(unittest.TestCase):
    def test_zero_variance_stays_zero(self):
        n = 20
        t = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + n * np.timedelta64(12, "h"),
            np.timedelta64(12, "h"),
        )
        inputs = VarianceInputs(
            t=t,
            n=np.full(n, 10.0),
            e=np.full(n, -3.0),
            z=np.full(n, 2.0),
            mlat=55.0,
            cadence_seconds=12 * 3600,
        )
        result = ModernVarianceEngine().fit(inputs)
        np.testing.assert_allclose(result.v, 0.0)
        self.assertTrue(np.isnan(result.u_n[0]))
        self.assertTrue(np.isnan(result.u_e[0]))
        self.assertTrue(np.isnan(result.u_z[0]))
        np.testing.assert_allclose(result.u_n[1:], 0.0)
        np.testing.assert_allclose(result.u_e[1:], 0.0)
        np.testing.assert_allclose(result.u_z[1:], 0.0)

    def test_cosine_memory_matches_bruteforce(self):
        x = np.array([1.0, np.nan, 3.0, 5.0, np.nan, 2.0], dtype=float)
        actual = causal_cosine_memory_smooth(x, window_size=3)
        expected = brute_force_causal_memory_smooth(x, window_size=3)
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_v_matches_bruteforce_definition(self):
        n = np.array([1.0, 3.0, np.nan, 5.0, 7.0], dtype=float)
        e = np.array([2.0, 4.0, 6.0, np.nan, 10.0], dtype=float)
        z = np.array([1.0, 1.0, 2.0, 2.0, np.nan], dtype=float)

        t = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + len(n) * np.timedelta64(12, "h"),
            np.timedelta64(12, "h"),
        )
        inputs = VarianceInputs(
            t=t,
            n=n,
            e=e,
            z=z,
            mlat=45.0,
            cadence_seconds=12 * 3600,
        )
        result = ModernVarianceEngine().fit(inputs)

        ss_n, count_n = brute_force_rolling_sum_of_squares(n, window_size=2)
        ss_e, count_e = brute_force_rolling_sum_of_squares(e, window_size=2)
        ss_z, count_z = brute_force_rolling_sum_of_squares(z, window_size=2)
        total_ss = ss_n + ss_e + ss_z
        total_count = count_n + count_e + count_z
        expected_v = np.full(total_ss.shape, np.nan, dtype=float)
        valid = total_count > 0
        expected_v[valid] = total_ss[valid] / total_count[valid]

        np.testing.assert_allclose(result.v, expected_v, equal_nan=True)

    @unittest.skipIf(
        VarianceEstimator is None,
        "legacy reference variance estimator is unavailable in this environment",
    )
    def test_matches_legacy_reference(self):
        t = np.arange(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01") + 40 * np.timedelta64(12, "h"),
            np.timedelta64(12, "h"),
        )
        n = np.sin(np.arange(len(t), dtype=float) / 3.0)
        e = np.cos(np.arange(len(t), dtype=float) / 5.0)
        z = np.linspace(-2.0, 2.0, len(t))
        n[7] = np.nan
        e[11] = np.nan
        z[17] = np.nan

        inputs = VarianceInputs(
            t=t,
            n=n,
            e=e,
            z=z,
            mlat=52.0,
            cadence_seconds=12 * 3600,
        )
        actual = ModernVarianceEngine().fit(inputs)

        reference = VarianceEstimator(t=t, N=n, E=e, Z=z, mlat=52.0)
        reference.estimate()
        np.testing.assert_allclose(
            actual.v,
            reference.df["v"].to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual.u_n,
            reference.df["uN"].to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual.u_e,
            reference.df["uE"].to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual.u_z,
            reference.df["uZ"].to_numpy(dtype=float),
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
