from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .types import VarianceInputs, VarianceResult


class ModernVarianceEngine:
    """
    Array-first implementation of the modified-variance estimator.

    This is the first V2 component that no longer delegates to the legacy
    pandas-based implementation.
    """

    def fit(self, inputs: VarianceInputs) -> VarianceResult:
        v = _compute_v(
            n=inputs.n,
            e=inputs.e,
            z=inputs.z,
            cadence_seconds=inputs.cadence_seconds,
        )
        f_n, f_e, f_z = _compute_f(inputs.mlat)
        d_n, an = _compute_d(v, f_n, inputs.cadence_seconds, inputs.mlat)
        d_e, ae = _compute_d(v, f_e, inputs.cadence_seconds, inputs.mlat)
        d_z, az = _compute_d(v, f_z, inputs.cadence_seconds, inputs.mlat)
        return VarianceResult(
            t=np.asarray(inputs.t),
            v=v,
            u_n=v + d_n,
            u_e=v + d_e,
            u_z=v + d_z,
            diagnostics={
                "fN": np.full(v.shape, f_n, dtype=float),
                "fE": np.full(v.shape, f_e, dtype=float),
                "fZ": np.full(v.shape, f_z, dtype=float),
                "AN": an,
                "AE": ae,
                "AZ": az,
                "dN": d_n,
                "dE": d_e,
                "dZ": d_z,
            },
        )


def _compute_v(
    *,
    n: np.ndarray,
    e: np.ndarray,
    z: np.ndarray,
    cadence_seconds: int,
) -> np.ndarray:
    window_size = _seconds_to_window_size(24 * 3600, cadence_seconds)
    ss_n, count_n = rolling_sum_of_squares(n, window_size)
    ss_e, count_e = rolling_sum_of_squares(e, window_size)
    ss_z, count_z = rolling_sum_of_squares(z, window_size)

    total_ss = ss_n + ss_e + ss_z
    total_count = count_n + count_e + count_z
    v = np.full(total_ss.shape, np.nan, dtype=float)
    valid = total_count > 0
    v[valid] = total_ss[valid] / total_count[valid]
    return v


def _compute_f(mlat: float) -> tuple[float, float, float]:
    mlat_rad = float(mlat) / 180.0 * np.pi
    f_n = float(np.abs(np.cos(mlat_rad)))
    f_e = 0.0
    f_z = float(np.abs(np.sin(mlat_rad)))
    return f_n, f_e, f_z


def _compute_d(
    v: np.ndarray,
    f_component: float,
    cadence_seconds: int,
    mlat: float,
) -> tuple[np.ndarray, np.ndarray]:
    window_size = _seconds_to_window_size(8 * 24 * 3600, cadence_seconds)
    scaled = np.asarray(v, dtype=float) * float(f_component)
    if float(mlat) > 60.0:
        scaled = np.zeros_like(scaled, dtype=float)
    d_component = causal_cosine_memory_smooth(scaled, window_size)
    return d_component, scaled


def _seconds_to_window_size(duration_seconds: int, cadence_seconds: int) -> int:
    window_size = int(duration_seconds / int(cadence_seconds))
    if window_size <= 0:
        raise ValueError(
            "cadence_seconds is too large for the configured smoothing window"
        )
    return window_size


def rolling_sum_of_squares(
    x: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rolling sum of squared deviations and valid-sample count."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    x_filled = np.where(valid, x, 0.0)

    sum_x = rolling_window_sum(x_filled, window_size)
    sum_x2 = rolling_window_sum(x_filled**2, window_size)
    count = rolling_window_sum(valid.astype(float), window_size)

    ss = np.zeros_like(sum_x, dtype=float)
    nonzero = count > 0
    ss[nonzero] = sum_x2[nonzero] - (sum_x[nonzero] ** 2) / count[nonzero]
    return ss, count


def rolling_window_sum(x: np.ndarray, window_size: int) -> np.ndarray:
    """Compute the trailing window sum for a one-dimensional array."""
    x = np.asarray(x, dtype=float)
    csum = np.cumsum(x)
    csum = np.concatenate(([0.0], csum))
    i = np.arange(len(x))
    i0 = np.maximum(0, i - window_size)
    return csum[i + 1] - csum[i0]


def causal_cosine_memory_smooth(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply the causal 8-day cosine memory kernel used in equation 12.

    This keeps the same behavior as the reference code:
    the kernel starts one sample in the past, and missing support yields NaN.
    """
    x = np.asarray(x, dtype=float)
    lag = np.arange(1, window_size + 1, dtype=float)
    k = 1.0 / window_size
    kernel = k * (1.0 + np.cos(lag * np.pi * k))

    valid = np.isfinite(x).astype(float)
    x_filled = np.where(np.isfinite(x), x, 0.0)
    full = fftconvolve(x_filled, kernel, mode="full")
    support = fftconvolve(valid, np.ones_like(kernel), mode="full")

    y = np.concatenate(([0.0], full[: len(x) - 1]))
    support = np.concatenate(([0.0], support[: len(x) - 1]))
    y[support == 0] = np.nan
    return y
