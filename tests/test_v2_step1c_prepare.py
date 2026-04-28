from __future__ import annotations

import unittest

import numpy as np

from baseline_v2.step1c_prepare import (
    build_step1c_target_index,
    collect_step1c_window_chunks,
    compute_step1c_bin30,
    expand_step1c_window,
    get_max_odd_window_size,
    prepare_step1c_day_bin_cache,
    validate_odd_window_days,
)


class Step1CPrepareTests(unittest.TestCase):
    def test_compute_bin30(self):
        t = np.array(
            [
                np.datetime64("2024-03-06T00:00"),
                np.datetime64("2024-03-06T00:29"),
                np.datetime64("2024-03-06T00:30"),
                np.datetime64("2024-03-06T23:59"),
            ]
        )
        actual = compute_step1c_bin30(t)
        expected = np.array([0, 0, 1, 47])
        np.testing.assert_array_equal(actual, expected)

    def test_build_target_index(self):
        days = np.array(
            [np.datetime64("2024-03-06"), np.datetime64("2024-03-07")],
            dtype="datetime64[D]",
        )
        target_index = build_step1c_target_index(days)
        self.assertEqual(target_index.shape, (96,))
        self.assertEqual(target_index[0], np.datetime64("2024-03-06T00:15"))
        self.assertEqual(target_index[1], np.datetime64("2024-03-06T00:45"))
        self.assertEqual(target_index[48], np.datetime64("2024-03-07T00:15"))

    def test_prepare_day_bin_cache(self):
        t = np.array(
            [
                np.datetime64("2024-03-06T00:15"),
                np.datetime64("2024-03-06T00:15"),
                np.datetime64("2024-03-06T00:45"),
                np.datetime64("2024-03-07T00:15"),
                np.datetime64("2024-03-07T12:15"),
            ]
        )
        x = np.array([1.0, np.nan, 2.0, 3.0, 4.0], dtype=float)
        residual = np.array([10.0, np.nan, 20.0, 30.0, np.nan], dtype=float)
        fwhm = np.array([5.0, np.nan, 7.0, 11.0, 13.0], dtype=float)

        cache = prepare_step1c_day_bin_cache(t, x, residual, fwhm)

        expected_days = np.array(
            [np.datetime64("2024-03-06"), np.datetime64("2024-03-07")],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(cache.days, expected_days)
        self.assertEqual(cache.target_counts[0, 0], 1)
        self.assertEqual(cache.target_counts[0, 1], 1)
        self.assertEqual(cache.target_counts[1, 0], 1)
        self.assertEqual(cache.target_counts[1, 24], 1)
        np.testing.assert_array_equal(cache.residuals_by_bin[0][0], np.array([10.0]))
        np.testing.assert_array_equal(cache.residuals_by_bin[1][0], np.array([20.0]))
        np.testing.assert_array_equal(cache.residuals_by_bin[0][1], np.array([30.0]))
        self.assertEqual(cache.residuals_by_bin[24][1].size, 0)
        self.assertEqual(cache.fwhm_sums[0, 0], 5.0)
        self.assertEqual(cache.fwhm_sums[0, 1], 7.0)
        self.assertEqual(cache.fwhm_sums[1, 0], 11.0)
        self.assertEqual(cache.fwhm_sums[1, 24], 13.0)
        self.assertEqual(cache.fwhm_counts[0, 0], 1)
        self.assertEqual(cache.fwhm_counts[1, 24], 1)

    def test_collect_and_expand_window(self):
        empty = np.empty(0, dtype=float)
        day_arrays = [
            np.array([1.0, 2.0]),
            empty,
            np.array([3.0]),
            np.array([4.0, 5.0, 6.0]),
            empty,
        ]
        fwhm_sums = np.array([10.0, 0.0, 20.0, 30.0, 0.0])
        fwhm_counts = np.array([1, 0, 2, 3, 0])

        chunks, n_samples = collect_step1c_window_chunks(day_arrays, 2, 2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(n_samples, 1)

        expanded = expand_step1c_window(
            day_idx=2,
            num_days=5,
            current_window_days=1,
            max_window_days=5,
            current_lo=2,
            current_hi=2,
            current_chunks=chunks,
            current_n_samples=n_samples,
            residual_day_arrays=day_arrays,
            fwhm_sums_bin=fwhm_sums,
            fwhm_counts_bin=fwhm_counts,
            current_fwhm_sum=20.0,
            current_fwhm_count=2,
        )
        next_window_days, next_lo, next_hi, next_chunks, next_n, next_sum, next_count = expanded
        self.assertEqual(next_window_days, 3)
        self.assertEqual((next_lo, next_hi), (1, 3))
        self.assertEqual(next_n, 4)
        self.assertEqual(next_sum, 50.0)
        self.assertEqual(next_count, 5)
        self.assertEqual(len(next_chunks), 2)

    def test_window_validation_helpers(self):
        self.assertEqual(validate_odd_window_days(5, "window"), 5)
        self.assertEqual(get_max_odd_window_size(5), 5)
        self.assertEqual(get_max_odd_window_size(6), 5)
        with self.assertRaises(ValueError):
            validate_odd_window_days(4, "window")
        with self.assertRaises(ValueError):
            get_max_odd_window_size(0)


if __name__ == "__main__":
    unittest.main()
