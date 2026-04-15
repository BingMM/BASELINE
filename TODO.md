## TODO

- Decide whether to keep the current IRLS-based `get_typical_value()` or replace it with a more explicit density-mode estimator such as a histogram- or KDE-based approach.
- Revisit the weighting formulas in baseline step 1c and step 2a. They are still informed guesses, not direct instructions from the paper.
- Step 1c / 1d detail: the paper says the semi-hourly weights should reflect both the spread of each local fit and the width of the window needed to obtain an acceptable solution. The current code uses `1 / (sigma**2 * window_days / 3)`, which is reasonable as a heuristic, but it should be checked against alternatives such as separating the spread term and the window-width penalty or using a milder penalty for wider windows.
- Step 2a / 2b detail: the paper says the yearly-trend smoothing should be weighted by uncertainty defined from two parts, the instantaneous variance and the delayed variance history. The current code compresses that into a daily scalar using `1 / (u * window_days / 17)`. This should be revisited to decide whether the daily weight should use the mean, median, minimum, or another summary of `u` over the window, and how strongly the expanded window length should reduce the weight.
- Validation task for the weighting: compare candidate weighting formulas on synthetic storms and quiet intervals, then inspect whether the yearly trend starts to follow storm recovery too closely or becomes too stiff to track seasonal structure.
- Speed up the mode-based typical-value path. Likely options: avoid repeated full `curve_fit` calls when the histogram is clearly well behaved, cache or incrementally update histograms across neighboring windows, fit only near the modal region instead of the full histogram, or replace the fitted sigma with a cheaper local histogram-width estimate when full fitting is not needed.
- Implement step 3 (`QO`) or stop calling it from `BaselineEstimator.get_baseline()` until a concrete approach is chosen.
- Add regression tests against the synthetic example data for `VarianceEstimator` and `BaselineEstimator`.
- Update `scripts/example_with_synthetic_data.py` to match the current class interfaces and imports.
- Validate edge cases explicitly: short records, large gaps, irregular cadence, and all-NaN windows.
