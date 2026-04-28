from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .diagnostics import sigma_ratio_confidence
from .reference_math import (
    cubic_convolution_interpolate,
    get_fwhm_stat,
    get_typical_value,
    get_weight_sigma,
    weighted_gaussian_smooth,
)
from .step1c_prepare import (
    collect_step1c_window_chunks,
    expand_step1c_window,
    get_max_odd_window_size,
    prepare_step1c_day_bin_cache,
)
from .types import (
    BaselineInputs,
    BaselineResult,
    ModernBaselineConfig,
    Step1CResult,
    VarianceResult,
)


def run_reference_component(
    inputs: BaselineInputs,
    variance: VarianceResult,
    config: ModernBaselineConfig,
) -> BaselineResult:
    return run_component_with_local_estimator(
        inputs=inputs,
        variance=variance,
        config=config,
        local_estimator=get_typical_value,
    )


def run_component_with_local_estimator(
    *,
    inputs: BaselineInputs,
    variance: VarianceResult,
    config: ModernBaselineConfig,
    local_estimator,
) -> BaselineResult:
    progress_label = config.progress_label or inputs.component
    u = variance.u_for_component(inputs.component)
    fwhm_stat = get_fwhm_stat(inputs.mlat, inputs.component, inputs.t.size)

    step1a_t, step1a_value = _step_1a(
        inputs.t,
        inputs.x,
        progress_desc=f"{progress_label} Step 1a" if config.verbose else None,
    )
    step1b = _step_1b(
        step1a_t,
        step1a_value,
        inputs.t,
        progress_desc=f"{progress_label} Step 1b" if config.verbose else None,
    )
    residual_step_1 = inputs.x - step1b

    expected_target_index = prepare_step1c_day_bin_cache(
        inputs.t,
        inputs.x,
        residual_step_1,
        fwhm_stat,
    ).target_index
    checkpoint_path = _resolve_checkpoint_path(
        config.step_1c_checkpoint_path,
        component=inputs.component,
    )
    if checkpoint_path is not None and config.reuse_step_1c_checkpoint and checkpoint_path.exists():
        step1c = _load_step1c_checkpoint(
            checkpoint_path,
            inputs.component,
            expected_target_index=expected_target_index,
        )
    else:
        step1c = _step_1c(
            t=inputs.t,
            x=inputs.x,
            residual_step_1=residual_step_1,
            fwhm_stat=fwhm_stat,
            min_window_days=config.step_1c_min_window_days,
            local_estimator=local_estimator,
            progress_label=progress_label,
            verbose=config.verbose,
            progress_every_days=config.progress_every_days,
        )
        if checkpoint_path is not None and config.write_step_1c_checkpoint:
            _save_step1c_checkpoint(checkpoint_path, inputs.component, step1c)

    qd, step1d_node_value = _step_1d(
        target_index=step1c.t,
        step1c=step1c,
        t_full=inputs.t,
        x=inputs.x,
        a=config.step_1d_a,
        sigma_days=config.step_1d_sigma_days,
        progress_label=progress_label if config.verbose else None,
    )
    x_qd = inputs.x - qd

    step2a_t, step2a_value, step2a_weight = _step_2a(
        t=inputs.t,
        x_qd=x_qd,
        u=u,
        fwhm_stat=fwhm_stat,
        progress_desc=f"{progress_label} Step 2a" if config.verbose else None,
    )
    qy = _step_2b(
        step2a_t=step2a_t,
        step2a_value=step2a_value,
        step2a_weight=step2a_weight,
        t_full=inputs.t,
        x_qd=x_qd,
        a=config.step_2b_a,
        sigma_days=config.step_2b_sigma_days,
        progress_label=progress_label if config.verbose else None,
    )
    residual = x_qd - qy

    diagnostics = _build_reference_diagnostics(
        step1a_t=step1a_t,
        step1a_value=step1a_value,
        step1b=step1b,
        residual_step_1=residual_step_1,
        step1d_node_value=step1d_node_value,
        step2a_t=step2a_t,
        step2a_value=step2a_value,
        step2a_weight=step2a_weight,
        step1c=step1c,
    )
    return BaselineResult(
        t=np.asarray(inputs.t),
        component=inputs.component,
        x=np.asarray(inputs.x, dtype=float),
        u=np.asarray(u, dtype=float),
        qd=qd,
        qy=qy,
        residual=residual,
        step1c=step1c,
        diagnostics=diagnostics,
    )


def _step_1a(
    t: np.ndarray,
    x: np.ndarray,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    days = t.astype("datetime64[D]")
    unique_days, inverse = np.unique(days, return_inverse=True)
    values = np.full(unique_days.shape, np.nan, dtype=float)
    iterator = range(unique_days.size)
    if progress_desc is not None:
        iterator = tqdm(iterator, total=unique_days.size, desc=progress_desc)

    for day_idx in iterator:
        mu, _ = get_typical_value(x[inverse == day_idx], return_diagnostics=False)
        values[day_idx] = mu
    timestamps = unique_days.astype("datetime64[m]") + np.timedelta64(12 * 60, "m")
    return timestamps, values


def _step_1b(
    step1a_t: np.ndarray,
    step1a_value: np.ndarray,
    t_full: np.ndarray,
    progress_desc: str | None = None,
) -> np.ndarray:
    t_nodes = _datetime64_to_seconds(step1a_t)
    t_full_seconds = _datetime64_to_seconds(t_full)
    return cubic_convolution_interpolate(
        t_nodes,
        step1a_value,
        t_full_seconds,
        a=-0.5,
        progress_desc=progress_desc,
    )


def _step_1c(
    *,
    t: np.ndarray,
    x: np.ndarray,
    residual_step_1: np.ndarray,
    fwhm_stat: np.ndarray,
    min_window_days: int,
    local_estimator,
    progress_label: str,
    verbose: bool,
    progress_every_days: int,
) -> Step1CResult:
    cache = prepare_step1c_day_bin_cache(t, x, residual_step_1, fwhm_stat)
    days = cache.days
    max_window_days = get_max_odd_window_size(days.size)

    num_targets = cache.target_index.size
    value = np.full(num_targets, np.nan, dtype=float)
    weight = np.zeros(num_targets, dtype=float)
    sigma = np.full(num_targets, np.nan, dtype=float)
    status = np.empty(num_targets, dtype=object)
    window_days_out = np.full(num_targets, np.nan, dtype=float)

    diag_target_n_samples = np.zeros(num_targets, dtype=int)
    diag_max_n_samples = np.zeros(num_targets, dtype=int)
    diag_last_n_samples = np.zeros(num_targets, dtype=int)
    diag_last_window_days = np.full(num_targets, np.nan, dtype=float)
    diag_last_mu = np.full(num_targets, np.nan, dtype=float)
    diag_last_sigma = np.full(num_targets, np.nan, dtype=float)
    diag_last_sigma_weight = np.full(num_targets, np.nan, dtype=float)
    diag_last_fwhm = np.full(num_targets, np.nan, dtype=float)
    diag_last_fwhm_stat = np.full(num_targets, np.nan, dtype=float)

    day_iter = range(days.size)
    if verbose:
        day_iter = tqdm(
            day_iter,
            total=days.size,
            desc=f"{progress_label} Step 1c",
            unit="day",
            leave=True,
            miniters=max(1, progress_every_days),
        )

    target_idx = 0
    for day_idx in day_iter:
        for b in range(48):
            target_n_samples = int(cache.target_counts[day_idx, b])
            diag_target_n_samples[target_idx] = target_n_samples
            status_value = "too_few_samples"
            max_n_samples = 0
            last_n_samples = 0
            last_window_days = np.nan
            last_mu = np.nan
            last_sigma_value = np.nan
            last_sigma_weight = np.nan
            last_fwhm = np.nan
            last_fwhm_stat = np.nan

            if target_n_samples == 0:
                status[target_idx] = "missing_input"
                target_idx += 1
                continue

            window_days = min(min_window_days, max_window_days)
            current_half = window_days // 2
            current_lo = max(0, day_idx - current_half)
            current_hi = min(days.size - 1, day_idx + current_half)
            current_chunks, n_samples = collect_step1c_window_chunks(
                cache.residuals_by_bin[b],
                current_lo,
                current_hi,
            )
            fwhm_sum_window = float(np.sum(cache.fwhm_sums[current_lo : current_hi + 1, b]))
            fwhm_count_window = int(np.sum(cache.fwhm_counts[current_lo : current_hi + 1, b]))

            accepted_value = np.nan
            accepted_sigma = np.nan
            accepted_window_days = np.nan

            while window_days <= max_window_days:
                max_n_samples = max(max_n_samples, n_samples)
                if n_samples < 5:
                    (
                        window_days,
                        current_lo,
                        current_hi,
                        current_chunks,
                        n_samples,
                        fwhm_sum_window,
                        fwhm_count_window,
                    ) = expand_step1c_window(
                        day_idx=day_idx,
                        num_days=days.size,
                        current_window_days=window_days,
                        max_window_days=max_window_days,
                        current_lo=current_lo,
                        current_hi=current_hi,
                        current_chunks=current_chunks,
                        current_n_samples=n_samples,
                        residual_day_arrays=cache.residuals_by_bin[b],
                        fwhm_sums_bin=cache.fwhm_sums[:, b],
                        fwhm_counts_bin=cache.fwhm_counts[:, b],
                        current_fwhm_sum=fwhm_sum_window,
                        current_fwhm_count=fwhm_count_window,
                    )
                    continue

                vals = current_chunks[0] if len(current_chunks) == 1 else np.concatenate(current_chunks)
                fwhm_window = (
                    fwhm_sum_window / fwhm_count_window if fwhm_count_window > 0 else np.nan
                )
                mu, sigma_value = local_estimator(vals, return_diagnostics=False)
                last_n_samples = n_samples
                last_window_days = window_days
                last_mu = mu
                last_sigma_value = sigma_value
                last_fwhm_stat = fwhm_window

                if not np.isfinite(mu) or not np.isfinite(sigma_value):
                    status_value = "typical_value_failed"
                    (
                        window_days,
                        current_lo,
                        current_hi,
                        current_chunks,
                        n_samples,
                        fwhm_sum_window,
                        fwhm_count_window,
                    ) = expand_step1c_window(
                        day_idx=day_idx,
                        num_days=days.size,
                        current_window_days=window_days,
                        max_window_days=max_window_days,
                        current_lo=current_lo,
                        current_hi=current_hi,
                        current_chunks=current_chunks,
                        current_n_samples=n_samples,
                        residual_day_arrays=cache.residuals_by_bin[b],
                        fwhm_sums_bin=cache.fwhm_sums[:, b],
                        fwhm_counts_bin=cache.fwhm_counts[:, b],
                        current_fwhm_sum=fwhm_sum_window,
                        current_fwhm_count=fwhm_count_window,
                    )
                    continue

                last_sigma_weight = get_weight_sigma(vals, mu, sigma_value)
                last_fwhm = 2.355 * sigma_value
                if np.isfinite(sigma_value) and sigma_value <= fwhm_window:
                    accepted_value = mu
                    accepted_sigma = sigma_value
                    accepted_window_days = float(window_days)
                    status_value = "ok"
                    break

                status_value = "fwhm_rejected"
                (
                    window_days,
                    current_lo,
                    current_hi,
                    current_chunks,
                    n_samples,
                    fwhm_sum_window,
                    fwhm_count_window,
                ) = expand_step1c_window(
                    day_idx=day_idx,
                    num_days=days.size,
                    current_window_days=window_days,
                    max_window_days=max_window_days,
                    current_lo=current_lo,
                    current_hi=current_hi,
                    current_chunks=current_chunks,
                    current_n_samples=n_samples,
                    residual_day_arrays=cache.residuals_by_bin[b],
                    fwhm_sums_bin=cache.fwhm_sums[:, b],
                    fwhm_counts_bin=cache.fwhm_counts[:, b],
                    current_fwhm_sum=fwhm_sum_window,
                    current_fwhm_count=fwhm_count_window,
                )

            if not np.isfinite(accepted_value):
                if max_n_samples == 0:
                    status_value = "no_residual_samples"
                elif max_n_samples < 5:
                    status_value = "too_few_samples"
            else:
                value[target_idx] = accepted_value
                sigma[target_idx] = accepted_sigma
                window_days_out[target_idx] = accepted_window_days

            denom = last_sigma_weight**2 * window_days_out[target_idx] / min_window_days
            if np.isfinite(denom) and denom > 0:
                weight[target_idx] = 1.0 / denom
            status[target_idx] = status_value

            diag_max_n_samples[target_idx] = max_n_samples
            diag_last_n_samples[target_idx] = last_n_samples
            diag_last_window_days[target_idx] = last_window_days
            diag_last_mu[target_idx] = last_mu
            diag_last_sigma[target_idx] = last_sigma_value
            diag_last_sigma_weight[target_idx] = last_sigma_weight
            diag_last_fwhm[target_idx] = last_fwhm
            diag_last_fwhm_stat[target_idx] = last_fwhm_stat
            target_idx += 1

    diagnostics = {
        "target_n_samples": diag_target_n_samples,
        "max_n_samples": diag_max_n_samples,
        "last_n_samples": diag_last_n_samples,
        "last_window_days": diag_last_window_days,
        "last_mu": diag_last_mu,
        "last_sigma": diag_last_sigma,
        "last_sigma_weight": diag_last_sigma_weight,
        "last_fwhm": diag_last_fwhm,
        "last_fwhm_stat": diag_last_fwhm_stat,
    }
    confidence = sigma_ratio_confidence(
        sigma=diag_last_sigma,
        threshold=diag_last_fwhm_stat,
        status=status,
    )
    result = Step1CResult(
        t=np.asarray(cache.target_index),
        value=value,
        weight=weight,
        sigma=diag_last_sigma,
        confidence=confidence,
        status=status,
        window_days=diag_last_window_days,
        diagnostics=diagnostics,
    )
    return result


def _step_1d(
    *,
    target_index: np.ndarray,
    step1c: Step1CResult,
    t_full: np.ndarray,
    x: np.ndarray,
    a: float,
    sigma_days: float,
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    t_nodes = _datetime64_to_seconds(target_index)
    t_full_seconds = _datetime64_to_seconds(t_full)
    y_smooth = weighted_gaussian_smooth(
        t_nodes,
        step1c.value,
        step1c.weight,
        sigma_days=sigma_days,
        progress_desc=(
            f"{progress_label} Step 1d smooth" if progress_label is not None else None
        ),
    )
    missing_input_mask = step1c.status == "missing_input"
    y_smooth[missing_input_mask] = np.nan
    qd = cubic_convolution_interpolate(
        t_nodes,
        y_smooth,
        t_full_seconds,
        a=a,
        progress_desc=(
            f"{progress_label} Step 1d interp" if progress_label is not None else None
        ),
    )
    qd[~np.isfinite(x)] = np.nan
    return qd, y_smooth


def _step_2a(
    *,
    t: np.ndarray,
    x_qd: np.ndarray,
    u: np.ndarray,
    fwhm_stat: np.ndarray,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = t.astype("datetime64[D]")
    unique_days = np.unique(days)
    max_window_days = get_max_odd_window_size(unique_days.size)
    values = np.full(unique_days.shape, np.nan, dtype=float)
    weights = np.zeros(unique_days.shape, dtype=float)

    iterator = enumerate(unique_days)
    if progress_desc is not None:
        iterator = tqdm(iterator, total=unique_days.size, desc=progress_desc)

    for day_idx, day in iterator:
        target_mask = days == day
        target_n_samples = int(np.isfinite(x_qd[target_mask]).sum())
        if target_n_samples == 0:
            continue

        window_days = min(17, max_window_days)
        value = np.nan
        u_mean = np.nan
        accepted_window_days = np.nan
        while window_days <= max_window_days:
            half = window_days // 2
            lo = max(0, day_idx - half)
            hi = min(unique_days.size - 1, day_idx + half)
            window_mask = (days >= unique_days[lo]) & (days <= unique_days[hi])
            vals = x_qd[window_mask]
            vals = vals[np.isfinite(vals)]
            if vals.size < 10:
                window_days += 2
                continue

            fwhm_window = np.nanmean(fwhm_stat[window_mask])
            u_mean = np.nanmean(u[window_mask])
            mu, sigma_value = get_typical_value(vals, return_diagnostics=False)
            if np.isfinite(sigma_value) and sigma_value <= fwhm_window:
                value = mu
                accepted_window_days = float(window_days)
                break
            window_days += 2

        values[day_idx] = value
        denom = u_mean * accepted_window_days / 17 if np.isfinite(accepted_window_days) else np.nan
        if np.isfinite(denom) and denom > 0:
            weights[day_idx] = 1.0 / denom

    timestamps = unique_days.astype("datetime64[m]") + np.timedelta64(12 * 60, "m")
    return timestamps, values, weights


def _step_2b(
    *,
    step2a_t: np.ndarray,
    step2a_value: np.ndarray,
    step2a_weight: np.ndarray,
    t_full: np.ndarray,
    x_qd: np.ndarray,
    a: float,
    sigma_days: float,
    progress_label: str | None = None,
) -> np.ndarray:
    t_nodes = _datetime64_to_seconds(step2a_t)
    t_full_seconds = _datetime64_to_seconds(t_full)
    y_smooth = weighted_gaussian_smooth(
        t_nodes,
        step2a_value,
        step2a_weight,
        sigma_days=sigma_days,
        progress_desc=(
            f"{progress_label} Step 2b smooth" if progress_label is not None else None
        ),
    )
    qy = cubic_convolution_interpolate(
        t_nodes,
        y_smooth,
        t_full_seconds,
        a=a,
        progress_desc=(
            f"{progress_label} Step 2b interp" if progress_label is not None else None
        ),
    )
    qy[~np.isfinite(x_qd)] = np.nan
    return qy


def _build_reference_diagnostics(
    *,
    step1a_t: np.ndarray,
    step1a_value: np.ndarray,
    step1b: np.ndarray,
    residual_step_1: np.ndarray,
    step1d_node_value: np.ndarray,
    step2a_t: np.ndarray,
    step2a_value: np.ndarray,
    step2a_weight: np.ndarray,
    step1c: Step1CResult,
) -> dict[str, Any]:
    diagnostics = {
        "step_1a_t": np.asarray(step1a_t),
        "step_1a_value": np.asarray(step1a_value, dtype=float),
        "step_1b_value": np.asarray(step1b, dtype=float),
        "residual_step_1": np.asarray(residual_step_1, dtype=float),
        "step_1d_node_value": np.asarray(step1d_node_value, dtype=float),
        "step_2a_t": np.asarray(step2a_t),
        "step_2a_value": np.asarray(step2a_value, dtype=float),
        "step_2a_weight": np.asarray(step2a_weight, dtype=float),
    }
    if hasattr(step1c, "diagnostics"):
        diagnostics["step_1c_diagnostics"] = getattr(step1c, "diagnostics")
    return diagnostics


def _resolve_checkpoint_path(checkpoint_path: Path | None, *, component: str) -> Path | None:
    if checkpoint_path is None:
        return None
    if checkpoint_path.suffix:
        return checkpoint_path
    return checkpoint_path / f"step1c_{component.lower()}.pkl"


def _save_step1c_checkpoint(path: Path, component: str, step1c: Step1CResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "component": component,
        "step1c": step1c,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_step1c_checkpoint(
    path: Path,
    component: str,
    *,
    expected_target_index: np.ndarray,
) -> Step1CResult:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("component") != component:
        raise ValueError(
            "Step 1c checkpoint component mismatch: "
            f"expected {component!r}, got {payload.get('component')!r}"
        )
    step1c = payload.get("step1c")
    if not isinstance(step1c, Step1CResult):
        raise TypeError("Step 1c checkpoint does not contain a Step1CResult")
    if not np.array_equal(np.asarray(step1c.t), np.asarray(expected_target_index)):
        raise ValueError("Step 1c checkpoint does not match the current time grid/day layout")
    return step1c


def _datetime64_to_seconds(t: np.ndarray) -> np.ndarray:
    return np.asarray(t).astype("datetime64[s]").astype(float)
