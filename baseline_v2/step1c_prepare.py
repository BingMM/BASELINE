from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Step1CDayBinCache:
    days: np.ndarray
    target_index: np.ndarray
    residuals_by_bin: list[list[np.ndarray]]
    target_counts: np.ndarray
    fwhm_sums: np.ndarray
    fwhm_counts: np.ndarray


def prepare_step1c_day_bin_cache(
    t: np.ndarray,
    x: np.ndarray,
    residual_step_1: np.ndarray,
    fwhm_stat: np.ndarray,
) -> Step1CDayBinCache:
    """
    Precompute per-day/per-bin arrays for the Step 1c inner loop.

    This is the V2 array-first equivalent of the legacy dataframe/groupby
    preparation step. It builds:

    - sorted unique days
    - one semi-hourly target timestamp grid
    - target-day finite sample counts
    - per-day/per-bin residual arrays
    - per-day/per-bin FWHM sums and counts
    """
    t = np.asarray(t)
    x = np.asarray(x, dtype=float)
    residual_step_1 = np.asarray(residual_step_1, dtype=float)
    fwhm_stat = np.asarray(fwhm_stat, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if x.shape != t.shape or residual_step_1.shape != t.shape or fwhm_stat.shape != t.shape:
        raise ValueError("t, x, residual_step_1, and fwhm_stat must have matching shapes")

    days = t.astype("datetime64[D]")
    unique_days, day_idx = np.unique(days, return_inverse=True)
    bin30 = _compute_bin30(t)

    num_days = unique_days.size
    num_bins = 48

    empty = np.empty(0, dtype=float)
    residuals_by_bin = [[empty for _ in range(num_days)] for _ in range(num_bins)]
    target_counts = np.zeros((num_days, num_bins), dtype=int)
    fwhm_sums = np.zeros((num_days, num_bins), dtype=float)
    fwhm_counts = np.zeros((num_days, num_bins), dtype=int)

    finite_x = np.isfinite(x)
    np.add.at(target_counts, (day_idx[finite_x], bin30[finite_x]), 1)

    finite_fwhm = np.isfinite(fwhm_stat)
    np.add.at(fwhm_sums, (day_idx[finite_fwhm], bin30[finite_fwhm]), fwhm_stat[finite_fwhm])
    np.add.at(fwhm_counts, (day_idx[finite_fwhm], bin30[finite_fwhm]), 1)

    finite_residual = np.isfinite(residual_step_1)
    if np.any(finite_residual):
        residual_day_idx = day_idx[finite_residual]
        residual_bin30 = bin30[finite_residual]
        residual_values = residual_step_1[finite_residual]
        group_key = residual_day_idx * num_bins + residual_bin30
        order = np.argsort(group_key, kind="stable")
        sorted_key = group_key[order]
        sorted_day_idx = residual_day_idx[order]
        sorted_bin30 = residual_bin30[order]
        sorted_values = residual_values[order]

        split_points = np.flatnonzero(np.diff(sorted_key)) + 1
        key_groups = np.split(sorted_key, split_points)
        day_groups = np.split(sorted_day_idx, split_points)
        bin_groups = np.split(sorted_bin30, split_points)
        value_groups = np.split(sorted_values, split_points)

        for key_group, day_group, bin_group, value_group in zip(
            key_groups,
            day_groups,
            bin_groups,
            value_groups,
        ):
            if key_group.size == 0:
                continue
            day_value = int(day_group[0])
            bin_value = int(bin_group[0])
            residuals_by_bin[bin_value][day_value] = value_group.astype(float, copy=True)

    target_index = build_step1c_target_index(unique_days)
    return Step1CDayBinCache(
        days=unique_days,
        target_index=target_index,
        residuals_by_bin=residuals_by_bin,
        target_counts=target_counts,
        fwhm_sums=fwhm_sums,
        fwhm_counts=fwhm_counts,
    )


def build_step1c_target_index(days: np.ndarray) -> np.ndarray:
    """Return one 15-minute-offset semi-hourly timestamp per day/bin."""
    days = np.asarray(days).astype("datetime64[m]")
    if days.ndim != 1:
        raise ValueError("days must be one-dimensional")
    num_days = days.size
    if num_days == 0:
        return np.empty(0, dtype="datetime64[m]")
    bin_offsets = (np.arange(48, dtype=int) * 30 + 15).astype("timedelta64[m]")
    return (days[:, None] + bin_offsets[None, :]).reshape(num_days * 48)


def compute_step1c_bin30(t: np.ndarray) -> np.ndarray:
    """Return the half-hour bin index [0, 47] for each timestamp."""
    return _compute_bin30(np.asarray(t))


def collect_step1c_window_chunks(
    day_arrays: list[np.ndarray],
    lo: int,
    hi: int,
) -> tuple[list[np.ndarray], int]:
    """Return the non-empty residual arrays and sample count for one day window."""
    chunks: list[np.ndarray] = []
    n_samples = 0
    for arr in day_arrays[lo : hi + 1]:
        if arr.size == 0:
            continue
        chunks.append(arr)
        n_samples += int(arr.size)
    return chunks, n_samples


def expand_step1c_window(
    day_idx: int,
    num_days: int,
    current_window_days: int,
    max_window_days: int,
    current_lo: int,
    current_hi: int,
    current_chunks: list[np.ndarray],
    current_n_samples: int,
    residual_day_arrays: list[np.ndarray],
    fwhm_sums_bin: np.ndarray,
    fwhm_counts_bin: np.ndarray,
    current_fwhm_sum: float,
    current_fwhm_count: int,
) -> tuple[int, int, int, list[np.ndarray], int, float, int]:
    """
    Expand one Step 1c day window by two days and update cached aggregates.

    The returned chunks, sample count, and FWHM accumulators stay in sync with
    the returned window-days/lo/hi state.
    """
    next_window_days = current_window_days + 2
    if next_window_days > max_window_days:
        return (
            next_window_days,
            current_lo,
            current_hi,
            current_chunks,
            current_n_samples,
            current_fwhm_sum,
            current_fwhm_count,
        )

    next_half = next_window_days // 2
    next_lo = max(0, day_idx - next_half)
    next_hi = min(num_days - 1, day_idx + next_half)
    chunks = list(current_chunks)
    n_samples = int(current_n_samples)
    fwhm_sum = float(current_fwhm_sum)
    fwhm_count = int(current_fwhm_count)

    if next_lo < current_lo:
        left_arr = residual_day_arrays[next_lo]
        if left_arr.size > 0:
            chunks.append(left_arr)
            n_samples += int(left_arr.size)
        fwhm_sum += float(fwhm_sums_bin[next_lo])
        fwhm_count += int(fwhm_counts_bin[next_lo])

    if next_hi > current_hi:
        right_arr = residual_day_arrays[next_hi]
        if right_arr.size > 0:
            chunks.append(right_arr)
            n_samples += int(right_arr.size)
        fwhm_sum += float(fwhm_sums_bin[next_hi])
        fwhm_count += int(fwhm_counts_bin[next_hi])

    return (
        next_window_days,
        next_lo,
        next_hi,
        chunks,
        n_samples,
        fwhm_sum,
        fwhm_count,
    )


def get_max_odd_window_size(num_days: int) -> int:
    """Return the largest odd window size that fits within `num_days`."""
    if num_days <= 0:
        raise ValueError("num_days must be positive")
    if num_days % 2 == 1:
        return int(num_days)
    return int(num_days - 1)


def validate_odd_window_days(window_days: int, name: str) -> int:
    """Validate that a configured day window is a positive odd integer."""
    try:
        window_days = int(window_days)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive odd integer") from exc
    if window_days <= 0 or window_days % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return window_days


def _compute_bin30(t: np.ndarray) -> np.ndarray:
    t_minutes = np.asarray(t).astype("datetime64[m]")
    day_floor = t_minutes.astype("datetime64[D]")
    minute_of_day = (t_minutes - day_floor).astype("timedelta64[m]").astype(int)
    bin30 = minute_of_day // 30
    if np.any((bin30 < 0) | (bin30 >= 48)):
        raise ValueError("timestamps must map to half-hour bins in the range [0, 47]")
    return bin30.astype(int, copy=False)
