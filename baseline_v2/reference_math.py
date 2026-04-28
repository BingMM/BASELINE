from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from tqdm import tqdm


def get_fwhm_stat(mlat: float, component: str, size: int) -> np.ndarray:
    """Evaluate the latitude-dependent FWHM threshold from equation 10."""
    mlat_array = np.full(int(size), float(mlat), dtype=float)
    if component == "N":
        return 30 - 0.20 * mlat_array + 17 * np.exp(-((mlat_array - 76) ** 2) / 100)
    if component == "E":
        return 7 + 0.12 * mlat_array + 14 * np.exp(-((mlat_array - 78) ** 2) / 150)
    if component == "Z":
        return 5 + 0.13 * mlat_array + 19 * np.exp(-((mlat_array - 78) ** 2) / 150)
    raise ValueError("component must be one of 'N', 'E', or 'Z'")


def get_typical_value(
    vals,
    return_diagnostics: bool = False,
):
    """Estimate the paper-style typical value and spread."""
    mu, sigma, diagnostics = _get_typical_value_paper_mode(
        vals,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        return mu, sigma, diagnostics
    return mu, sigma


def get_typical_value_dominant_region(
    vals,
    *,
    bin_width: float = 1.0,
    min_samples: int = 15,
    shoulder_fraction: float = 0.35,
    ambiguity_mode_dominance: float = 1.25,
    ambiguity_region_fraction: float = 0.60,
    return_diagnostics: bool = False,
):
    """
    Estimate a robust quiet-time center from the dominant contiguous histogram region.

    This is an experimental V2 estimator for Step 1c. It differs from the
    paper-style fit in two ways:

    - it identifies a dominant contiguous region around the modal peak instead
      of fitting the full histogram distribution;
    - it can reject windows as ambiguous when the leading and competing peaks
      are too similar and the dominant region does not contain enough of the
      histogram mass.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, None

    counts, edges = _fixed_width_histogram(vals, bin_width=bin_width)
    counts = counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.size == 0 or np.max(counts) <= 0:
        return np.nan, np.nan, None

    diagnostics = None
    if return_diagnostics:
        diagnostics = {
            "method": "dominant_region",
            "bin_width": float(bin_width),
            "min_samples": int(min_samples),
            "shoulder_fraction": float(shoulder_fraction),
        }

    if vals.size < min_samples:
        if diagnostics is not None:
            diagnostics["failure_reason"] = "too_few_samples_for_histogram"
        return _return_typical_result(np.nan, np.nan, diagnostics, return_diagnostics)

    if np.all(vals == vals[0]):
        if diagnostics is not None:
            diagnostics["failure_reason"] = "constant_values"
        return _return_typical_result(np.nan, np.nan, diagnostics, return_diagnostics)

    modal_bins = np.flatnonzero(counts == np.max(counts))
    mode_count = float(np.max(counts))
    second_peak_count = _get_second_peak_count(counts, modal_bins)
    mode_dominance = mode_count / second_peak_count if second_peak_count > 0 else np.inf

    left = int(modal_bins[0])
    right = int(modal_bins[-1])
    threshold = shoulder_fraction * mode_count
    while left > 0 and counts[left - 1] >= threshold:
        left -= 1
    while right < counts.size - 1 and counts[right + 1] >= threshold:
        right += 1

    region_edges = (edges[left], edges[right + 1])
    if right == counts.size - 1:
        in_region = (vals >= region_edges[0]) & (vals <= region_edges[1])
    else:
        in_region = (vals >= region_edges[0]) & (vals < region_edges[1])
    region_vals = vals[in_region]
    region_counts = counts[left : right + 1]
    region_centers = centers[left : right + 1]
    region_mass_fraction = float(np.sum(region_counts) / np.sum(counts))

    if diagnostics is not None:
        diagnostics.update(
            {
                "centers": centers,
                "counts": counts,
                "widths": np.diff(edges),
                "mode_count": mode_count,
                "second_peak_count": second_peak_count,
                "mode_dominance": mode_dominance,
                "dominant_region_left_bin": left,
                "dominant_region_right_bin": right,
                "dominant_region_fraction": region_mass_fraction,
                "dominant_region_left_edge": float(region_edges[0]),
                "dominant_region_right_edge": float(region_edges[1]),
            }
        )

    if region_vals.size < 5:
        if diagnostics is not None:
            diagnostics["failure_reason"] = "dominant_region_too_small"
        return _return_typical_result(np.nan, np.nan, diagnostics, return_diagnostics)

    typical_value = float(np.average(region_centers, weights=region_counts))
    sigma_region = float(np.sqrt(np.mean((region_vals - typical_value) ** 2)))
    if not np.isfinite(sigma_region) or sigma_region <= 0:
        if diagnostics is not None:
            diagnostics["failure_reason"] = "invalid_region_sigma"
        return _return_typical_result(typical_value, np.nan, diagnostics, return_diagnostics)

    if mode_dominance < ambiguity_mode_dominance and region_mass_fraction < ambiguity_region_fraction:
        if diagnostics is not None:
            diagnostics["failure_reason"] = "ambiguous_competing_modes"
            diagnostics["typical_value"] = typical_value
            diagnostics["sigma_fit"] = sigma_region
        return _return_typical_result(typical_value, np.nan, diagnostics, return_diagnostics)

    if diagnostics is not None:
        diagnostics["typical_value"] = typical_value
        diagnostics["sigma_fit"] = sigma_region
    return _return_typical_result(
        typical_value,
        sigma_region,
        diagnostics,
        return_diagnostics,
    )


def get_weight_sigma(
    vals,
    typical_value: float,
    sigma_fit: float,
    central_fraction: float = 68.0,
) -> float:
    """
    Estimate the uncertainty used for Step 1c weighting.

    The weighting should reflect both the Gaussian fit width and the central
    sample spread around the selected typical value.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    sigma_candidates = []
    if np.isfinite(sigma_fit) and sigma_fit > 0:
        sigma_candidates.append(float(sigma_fit))

    if vals.size >= 5 and np.isfinite(typical_value):
        distances = np.abs(vals - typical_value)
        radius = np.nanpercentile(distances, central_fraction)
        core = vals[distances <= radius]
        if core.size >= 5:
            sigma_core = np.sqrt(np.mean((core - typical_value) ** 2))
            if np.isfinite(sigma_core) and sigma_core > 0:
                sigma_candidates.append(float(sigma_core))

    if not sigma_candidates:
        return np.nan
    return max(sigma_candidates)


def cubic_convolution_interpolate(
    t_nodes,
    y_nodes,
    t_full,
    a: float = -0.5,
    progress_desc: str | None = None,
) -> np.ndarray:
    """Interpolate node values onto a target grid with cubic convolution."""
    t_nodes = np.asarray(t_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    t_full = np.asarray(t_full, dtype=float)

    y_interp = np.full_like(t_full, np.nan, dtype=float)
    finite_t = np.isfinite(t_nodes)
    t_nodes = t_nodes[finite_t]
    y_nodes = y_nodes[finite_t]

    if t_nodes.size == 0:
        return y_interp
    if t_nodes.size == 1:
        if np.isfinite(y_nodes[0]):
            y_interp[:] = y_nodes[0]
        return y_interp

    dt_nodes = np.median(np.diff(t_nodes))
    if not np.isfinite(dt_nodes) or dt_nodes <= 0:
        return y_interp

    iterator = enumerate(t_full)
    if progress_desc is not None:
        iterator = tqdm(iterator, total=t_full.size, desc=progress_desc)

    for i, t in iterator:
        u = (t - t_nodes[0]) / dt_nodes
        k = int(np.floor(u))
        val = 0.0
        wsum = 0.0

        for j in range(k - 1, k + 3):
            if 0 <= j < len(y_nodes) and np.isfinite(y_nodes[j]):
                wj = cubic_convolution_weight(u - j, a=a)
                val += y_nodes[j] * wj
                wsum += wj

        if wsum > 0:
            y_interp[i] = val / wsum

    return y_interp


def cubic_convolution_weight(x, a: float = -0.5):
    """Evaluate the cubic-convolution kernel at one or more offsets."""
    ax = np.abs(np.asarray(x, dtype=float))
    w = np.zeros_like(ax)

    m1 = ax < 1
    m2 = (ax >= 1) & (ax < 2)
    w[m1] = (a + 2) * ax[m1] ** 3 - (a + 3) * ax[m1] ** 2 + 1
    w[m2] = a * ax[m2] ** 3 - 5 * a * ax[m2] ** 2 + 8 * a * ax[m2] - 4 * a
    return float(w) if np.ndim(w) == 0 else w


def weighted_gaussian_smooth(
    t_nodes,
    y_nodes,
    w_nodes,
    sigma_days: float,
    progress_desc: str | None = None,
) -> np.ndarray:
    """Apply Gaussian temporal smoothing with user-supplied point weights."""
    t_nodes = np.asarray(t_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    w_nodes = np.asarray(w_nodes, dtype=float)

    y_smooth = np.zeros_like(y_nodes)
    base_sigma = float(sigma_days) * 86400.0
    iterator = enumerate(t_nodes)
    if progress_desc is not None:
        iterator = tqdm(iterator, total=t_nodes.size, desc=progress_desc)

    for i, ti in iterator:
        dt = t_nodes - ti
        temporal_weights = np.exp(-0.5 * (dt / base_sigma) ** 2)
        weights = w_nodes * temporal_weights
        mask = np.isfinite(y_nodes) & np.isfinite(weights) & (weights > 0)
        if np.sum(mask) == 0:
            y_smooth[i] = np.nan
            continue
        weight_sum = np.sum(weights[mask])
        if weight_sum <= 0:
            y_smooth[i] = np.nan
            continue
        y_smooth[i] = np.sum(y_nodes[mask] * weights[mask]) / weight_sum
    return y_smooth


def _get_typical_value_paper_mode(
    vals,
    bin_width: float = 1.0,
    min_samples: int = 15,
    return_diagnostics: bool = True,
):
    """
    Estimate the paper-style typical value using fixed 1 nT histogram bins.

    The typical value is the histogram mode from equation (5). The acceptance
    spread comes from a Gaussian fit to the full histogram distribution.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, None

    counts, edges = _fixed_width_histogram(vals, bin_width=bin_width)
    counts = counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.size == 0 or np.max(counts) <= 0:
        return np.nan, np.nan, None

    modal_bins = np.flatnonzero(counts == np.max(counts))
    mode_value = float(np.mean(centers[modal_bins]))

    diagnostics = None
    if return_diagnostics:
        diagnostics = {
            "centers": centers,
            "counts": counts,
            "widths": np.diff(edges),
            "mode_value": mode_value,
            "typical_value": mode_value,
            "fit_success": False,
            "amplitude": np.nan,
            "mu_fit": np.nan,
            "sigma_fit": np.nan,
            "method": "paper_mode",
            "bin_width": float(bin_width),
            "min_samples": int(min_samples),
            "spike_replaced": False,
            "failure_reason": None,
        }

    if vals.size < min_samples:
        if diagnostics is not None:
            diagnostics["failure_reason"] = "too_few_samples_for_histogram"
        return np.nan, np.nan, diagnostics

    if np.all(vals == vals[0]):
        if diagnostics is not None:
            diagnostics["failure_reason"] = "constant_values"
        return np.nan, np.nan, diagnostics

    populated = counts > 0
    if np.sum(populated) < 3:
        if diagnostics is not None:
            diagnostics["failure_reason"] = "too_few_populated_fit_bins"
        return mode_value, np.nan, diagnostics

    second_peak_count = _get_second_peak_count(counts, modal_bins)
    mode_count = float(np.max(counts))
    if diagnostics is not None:
        diagnostics["mode_count"] = mode_count
        diagnostics["second_peak_count"] = second_peak_count
        diagnostics["mode_dominance"] = (
            mode_count / second_peak_count if second_peak_count > 0 else np.inf
        )

    fit_weight_sum = np.sum(counts)
    mu0 = np.sum(centers * counts) / fit_weight_sum
    sigma0 = np.sqrt(np.sum(counts * (centers - mu0) ** 2) / fit_weight_sum)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = bin_width

    amp0 = mode_count
    mu_lower = edges[0]
    mu_upper = edges[-1]
    sigma_upper = max(bin_width, (edges[-1] - edges[0]) / 2)
    sigma0 = float(np.clip(sigma0, bin_width / 2, sigma_upper))
    mu0 = float(np.clip(mu0, mu_lower, mu_upper))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(
                _gaussian_pdf_shape,
                centers,
                counts,
                p0=(amp0, mu0, sigma0),
                bounds=([0.0, mu_lower, bin_width / 2], [np.inf, mu_upper, sigma_upper]),
                maxfev=10000,
            )
            amplitude, mu_fit, sigma_fit = params
            sigma_fit = abs(float(sigma_fit))
            fit_success = True
    except (RuntimeError, ValueError):
        amplitude = np.nan
        mu_fit = np.nan
        sigma_fit = np.nan
        fit_success = False

    if diagnostics is not None:
        diagnostics["fit_success"] = fit_success
        diagnostics["amplitude"] = float(amplitude) if np.isfinite(amplitude) else np.nan
        diagnostics["mu_fit"] = float(mu_fit) if np.isfinite(mu_fit) else np.nan
        diagnostics["sigma_fit"] = float(sigma_fit) if np.isfinite(sigma_fit) else np.nan
        diagnostics["fit_mu_minus_mode"] = (
            float(mu_fit - mode_value)
            if np.isfinite(mu_fit) and np.isfinite(mode_value)
            else np.nan
        )

    if not fit_success or not np.isfinite(sigma_fit):
        if diagnostics is not None:
            diagnostics["failure_reason"] = "gaussian_fit_failed"
        return mode_value, np.nan, diagnostics

    left = max(int(modal_bins[0]) - 1, 0)
    right = min(int(modal_bins[-1]) + 1, counts.size - 1)
    eq4_local_mean = float(np.mean(counts[left : right + 1]))
    eq4_condition_met = bool(amplitude > eq4_local_mean) if np.isfinite(amplitude) else False
    if diagnostics is not None:
        diagnostics["eq4_local_mean"] = eq4_local_mean
        diagnostics["eq4_condition_met"] = eq4_condition_met

    typical_value = mode_value
    if not eq4_condition_met and np.isfinite(mu_fit):
        typical_value = float(mu_fit)
        if diagnostics is not None:
            diagnostics["spike_replaced"] = True

    if diagnostics is not None:
        diagnostics["typical_value"] = float(typical_value)
        diagnostics["edges"] = edges

    return typical_value, sigma_fit, diagnostics


def _fixed_width_histogram(vals, bin_width: float = 1.0):
    vals = np.asarray(vals, dtype=float)
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    min_edge = np.floor(np.min(vals) / bin_width) * bin_width - bin_width / 2
    max_edge = np.ceil(np.max(vals) / bin_width) * bin_width + bin_width / 2
    if np.isclose(min_edge, max_edge):
        min_edge -= bin_width
        max_edge += bin_width

    edges = np.arange(min_edge, max_edge + bin_width, bin_width)
    counts, edges = np.histogram(vals, bins=edges)
    return counts, edges


def _get_second_peak_count(counts, modal_bins) -> float:
    counts = np.asarray(counts, dtype=float)
    non_modal = np.ones(counts.size, dtype=bool)
    non_modal[np.asarray(modal_bins, dtype=int)] = False
    if not np.any(non_modal):
        return 0.0
    return float(np.max(counts[non_modal]))


def _gaussian_pdf_shape(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _return_typical_result(mu, sigma, diagnostics, return_diagnostics: bool):
    if return_diagnostics:
        return mu, sigma, diagnostics
    return mu, sigma
