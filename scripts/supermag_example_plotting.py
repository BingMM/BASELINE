from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MINUTES_PER_DAY = 24 * 60
STEP_1C_STATUS_STYLES = {
    "missing_input": {
        "label": "No input in Step 1c bin",
        "marker": "v",
        "color": "0.25",
        "y": 0.08,
    },
    "no_residual_samples": {
        "label": "No residual samples",
        "marker": "s",
        "color": "tab:purple",
        "y": 0.14,
    },
    "too_few_samples": {
        "label": "Too few samples",
        "marker": "1",
        "color": "tab:brown",
        "y": 0.20,
    },
    "typical_value_failed": {
        "label": "Typical-value fit failed",
        "marker": "x",
        "color": "tab:red",
        "y": 0.26,
    },
    "fwhm_rejected": {
        "label": "FWHM rejected",
        "marker": "x",
        "color": "tab:olive",
        "y": 0.32,
    },
}


def save_example_figure(fig, example_figure_dir, subdir_name, filename):
    """Save and close a figure in a named SM_example subdirectory."""
    save_path = Path(example_figure_dir) / subdir_name / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def day_slice(start_day, num_days, num_points):
    """Return a sample slice defined in units of whole days."""
    start = start_day * MINUTES_PER_DAY
    stop = min(num_points, (start_day + num_days) * MINUTES_PER_DAY)
    return slice(start, stop)


def date_slice(t, start_time, num_days):
    """Return a sample slice from a timestamp and duration in days."""
    times = pd.to_datetime(t)
    start_time = pd.Timestamp(start_time)
    stop_time = start_time + pd.Timedelta(days=num_days)
    start = int(np.searchsorted(times.values, start_time.to_datetime64(), side="left"))
    stop = int(np.searchsorted(times.values, stop_time.to_datetime64(), side="left"))
    return slice(start, min(stop, times.size))


def iter_time_chunks(t, chunk_days):
    """Yield contiguous calendar-aligned time chunks covering the full record."""
    times = pd.to_datetime(t)
    if times.size == 0:
        return

    start_time = pd.Timestamp(times[0]).floor("D")
    stop_limit = pd.Timestamp(times[-1]).ceil("D")
    cursor = start_time

    while cursor < stop_limit:
        chunk_stop = min(cursor + pd.Timedelta(days=chunk_days), stop_limit)
        chunk_slice = date_slice(times, cursor, (chunk_stop - cursor) / pd.Timedelta(days=1))
        if chunk_slice.stop > chunk_slice.start:
            yield cursor, chunk_stop, chunk_slice
        cursor = chunk_stop


def chunk_filename(prefix, start_time, stop_time):
    """Build a stable filename for a time chunk."""
    inclusive_stop = stop_time - pd.Timedelta(minutes=1)
    return f"{prefix}_{start_time:%Y%m%d}_{inclusive_stop:%Y%m%d}.png"


def slice_has_finite(values, view_slice):
    """Return whether a slice contains any finite samples."""
    return bool(np.isfinite(np.asarray(values)[view_slice]).any())


def any_series_has_finite(series_list, view_slice):
    """Return whether any provided series contains finite samples in the slice."""
    return any(slice_has_finite(values, view_slice) for values in series_list)


def add_missing_data_spans(ax, t, values, view_slice):
    """Shade spans where the input component itself is missing."""
    times = pd.to_datetime(t[view_slice])
    vals = np.asarray(values[view_slice], dtype=float)
    missing = ~np.isfinite(vals)

    if times.size == 0 or not np.any(missing):
        return

    if times.size > 1:
        time_ns = times.values.astype("datetime64[ns]").astype(np.int64)
        diffs = np.diff(time_ns)
        diffs = diffs[diffs > 0]
        cadence_ns = int(np.median(diffs)) if diffs.size > 0 else 60_000_000_000
    else:
        cadence_ns = 60_000_000_000
    cadence = pd.to_timedelta(cadence_ns, unit="ns")

    edges = np.flatnonzero(np.diff(np.r_[False, missing, False]))
    label = "Input data missing"
    for start, stop in zip(edges[0::2], edges[1::2]):
        ax.axvspan(
            times[start],
            times[stop - 1] + cadence,
            color="0.85",
            alpha=0.55,
            label=label,
            zorder=0,
        )
        label = "_nolegend_"


def add_step_1c_status_markers(ax, estimator, start_time, stop_time):
    """Mark semi-hourly QD nodes that were absent or rejected."""
    status = estimator.QD_step_1c_status
    in_view = (status.index >= start_time) & (status.index <= stop_time)
    view_status = status.loc[in_view]

    for status_value, style in STEP_1C_STATUS_STYLES.items():
        marker_times = view_status.index[view_status == status_value]
        if marker_times.size == 0:
            continue
        ax.scatter(
            marker_times,
            np.full(marker_times.size, style["y"]),
            marker=style["marker"],
            color=style["color"],
            s=34,
            label=f"{style['label']} ({marker_times.size})",
            transform=ax.get_xaxis_transform(),
            zorder=7,
        )


def plot_qd_component(
    ax,
    t,
    input_values,
    supermag_qd,
    estimator,
    view_slice,
    title,
    error_plot=True,
):
    """Plot SuperMAG QD, local Step 1c nodes, and the smoothed local QD."""
    view_t = t[view_slice]
    start_time = pd.Timestamp(view_t[0])
    stop_time = pd.Timestamp(view_t[-1])

    add_missing_data_spans(ax, t, input_values, view_slice)
    ax.plot(view_t, supermag_qd[view_slice], color="tab:blue", label="SuperMAG QD")

    node_status = estimator.QD_step_1c_status
    ok = (
        (node_status.index >= start_time) &
        (node_status.index <= stop_time) &
        (node_status.values == "ok")
    )
    ok_times = node_status.index[ok]
    weights = estimator.QD_step_1c_w.loc[ok_times]
    if error_plot:
        yerr = np.full(weights.size, np.nan)
        finite_weights = np.isfinite(weights.values) & (weights.values > 0)
        yerr[finite_weights] = 1 / np.sqrt(weights.values[finite_weights])
        ax.errorbar(
            ok_times,
            estimator.QD_step_1c.loc[ok_times],
            yerr=yerr,
            color="tab:orange",
            ecolor="tab:orange",
            ls="none",
            marker=".",
            label="Typical values",
            alpha=0.9,
            zorder=4,
        )
    else:
        ax.plot(
            ok_times,
            estimator.QD_step_1c.loc[ok_times],
            ".",
            color="tab:orange",
            label="Typical values",
            alpha=0.9,
            zorder=4,
        )

    ax.plot(
        view_t,
        estimator.df["QD"][view_slice].values,
        color="tab:green",
        label="Local QD",
        linewidth=1.8,
        zorder=3,
    )
    add_step_1c_status_markers(ax, estimator, start_time, stop_time)
    ax.set_title(title)


def plot_component_triplet(axs, t, components, titles, view_slice):
    """Plot three component series on aligned axes."""
    for ax, values, title in zip(axs, components, titles):
        ax.plot(t[view_slice], values[view_slice])
        ax.set_title(title)


def plot_component_comparison_triplet(axs, t, component_pairs, titles, view_slice):
    """Plot three reference/estimate component comparisons on aligned axes."""
    for ax, (reference, estimate), title in zip(axs, component_pairs, titles):
        ax.plot(t[view_slice], reference[view_slice])
        if isinstance(estimate, pd.Series):
            estimate_view = estimate.iloc[view_slice].values
        else:
            estimate_view = np.asarray(estimate)[view_slice]
        ax.plot(t[view_slice], estimate_view)
        ax.set_title(title)


def save_chunked_component_triplet(
    example_figure_dir,
    t,
    components,
    titles,
    subdir_name,
    prefix,
    chunk_days,
    progress_label=None,
):
    """Write one three-panel component plot per time chunk."""
    chunks = list(iter_time_chunks(t, chunk_days))
    total = len(chunks)
    for chunk_idx, (start_time, stop_time, view_slice) in enumerate(chunks, start=1):
        if progress_label and (chunk_idx == 1 or chunk_idx % 25 == 0 or chunk_idx == total):
            print(f"{progress_label}: chunk {chunk_idx}/{total}")
        if not any_series_has_finite(components, view_slice):
            continue
        fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        plot_component_triplet(axs, t, components, titles, view_slice)
        save_example_figure(fig, example_figure_dir, subdir_name, chunk_filename(prefix, start_time, stop_time))


def save_chunked_component_comparison(
    example_figure_dir,
    t,
    component_pairs,
    titles,
    subdir_name,
    prefix,
    chunk_days,
    progress_label=None,
):
    """Write one three-panel comparison plot per time chunk."""
    chunks = list(iter_time_chunks(t, chunk_days))
    total = len(chunks)
    for chunk_idx, (start_time, stop_time, view_slice) in enumerate(chunks, start=1):
        if progress_label and (chunk_idx == 1 or chunk_idx % 25 == 0 or chunk_idx == total):
            print(f"{progress_label}: chunk {chunk_idx}/{total}")
        reference_series = [reference for reference, _ in component_pairs]
        if not any_series_has_finite(reference_series, view_slice):
            continue
        fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        plot_component_comparison_triplet(axs, t, component_pairs, titles, view_slice)
        save_example_figure(fig, example_figure_dir, subdir_name, chunk_filename(prefix, start_time, stop_time))


def save_chunked_qd_comp_triplet(
    example_figure_dir,
    t,
    component_specs,
    chunk_days,
    error_plot,
    progress_label=None,
):
    """Write the three-component QD comparison plot per chunk."""
    chunks = list(iter_time_chunks(t, chunk_days))
    total = len(chunks)
    for chunk_idx, (start_time, stop_time, view_slice) in enumerate(chunks, start=1):
        if progress_label and (chunk_idx == 1 or chunk_idx % 25 == 0 or chunk_idx == total):
            print(f"{progress_label}: chunk {chunk_idx}/{total}")
        input_series = [input_values for input_values, _, _, _ in component_specs]
        if not any_series_has_finite(input_series, view_slice):
            continue
        fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        for ax, (input_values, supermag_qd, estimator, title) in zip(axs, component_specs):
            plot_qd_component(ax, t, input_values, supermag_qd, estimator, view_slice, title, error_plot=error_plot)
        axs[0].legend(loc="best")
        save_example_figure(fig, example_figure_dir, "SM_QD_comp", chunk_filename("SM_QD_comp", start_time, stop_time))


def save_chunked_estimator_plot(
    example_figure_dir,
    estimator,
    t,
    chunk_days,
    finite_values,
    subdir_name,
    prefix,
    figure_builder,
    progress_label=None,
):
    """Write one estimator plot per time chunk using a supplied figure builder."""
    chunks = list(iter_time_chunks(t, chunk_days))
    total = len(chunks)
    for chunk_idx, (start_time, stop_time, view_slice) in enumerate(chunks, start=1):
        if progress_label and (chunk_idx == 1 or chunk_idx % 25 == 0 or chunk_idx == total):
            print(f"{progress_label}: chunk {chunk_idx}/{total}")
        if not slice_has_finite(finite_values, view_slice):
            continue
        fig = figure_builder(estimator, view_slice, start_time, stop_time)
        save_example_figure(fig, example_figure_dir, subdir_name, chunk_filename(prefix, start_time, stop_time))


def build_step_1a_figure(estimator, view_slice, start_time, stop_time):
    """Build the Step 1a figure for one time window."""
    fig = plt.figure(figsize=(15, 9))
    plt.plot(
        estimator.df["datetime"][view_slice],
        estimator.df["x"][view_slice],
        label="Observed magnetic field",
    )
    nodes = estimator.QD_step_1a.loc[
        (estimator.QD_step_1a.index >= start_time) &
        (estimator.QD_step_1a.index < stop_time)
    ]
    plt.plot(nodes.index, nodes.values, ".", label="Daily typical value")
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    plt.legend()
    return fig


def build_step_1b_figure(estimator, view_slice, _start_time, _stop_time):
    """Build the Step 1b figure for one time window."""
    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(estimator.df["datetime"][view_slice], estimator.df["x"][view_slice], label="Observed magnetic field")
    axs[0].plot(estimator.df["datetime"][view_slice], estimator.df["step_1b"][view_slice], label="Weighted fit to daily typical value")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(estimator.df["datetime"][view_slice], estimator.df["residual_step_1"][view_slice], label="Difference")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    return fig


def build_step_1c_figure(estimator, view_slice, start_time, stop_time):
    """Build the Step 1c figure for one time window."""
    half_hour_mask = (
        (estimator.QD_step_1c.index >= start_time) &
        (estimator.QD_step_1c.index < stop_time)
    )
    fig = plt.figure(figsize=(15, 9))
    plt.plot(estimator.df["datetime"][view_slice], estimator.df["residual_step_1"][view_slice], label="Field minus daily fit")
    plt.plot(estimator.QD_step_1c.loc[half_hour_mask], ".", label="Semi-hourly typical values")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    return fig


def build_step_1d_figure(estimator, view_slice, start_time, stop_time):
    """Build the Step 1d figure for one time window."""
    half_hour_mask = (
        (estimator.QD_step_1c.index >= start_time) &
        (estimator.QD_step_1c.index < stop_time)
    )
    fig = plt.figure(figsize=(15, 9))
    plt.plot(estimator.df["datetime"][view_slice], estimator.df["residual_step_1"][view_slice], label="Field minus daily fit")
    plt.plot(estimator.QD_step_1c.loc[half_hour_mask], ".", label="Semi-hourly typical values")
    plt.plot(estimator.df["datetime"][view_slice], estimator.df["QD"][view_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    return fig


def build_step_1e_figure(estimator, view_slice, _start_time, _stop_time):
    """Build the Step 1e figure for one time window."""
    fig = plt.figure(figsize=(15, 9))
    plt.plot(estimator.df["datetime"][view_slice], estimator.df["x"][view_slice], label="Observed signal")
    plt.plot(estimator.df["datetime"][view_slice], estimator.df["x_QD"][view_slice], label="Without daily variation")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    return fig


def build_step_2a_figure(estimator, view_slice, start_time, stop_time):
    """Build the Step 2a figure for one time window."""
    fig = plt.figure(figsize=(15, 9))
    plt.plot(estimator.df["datetime"][view_slice], estimator.df["x_QD"][view_slice], label="Observed signal without daily variation")
    nodes = estimator.QD_step_2a.loc[
        (estimator.QD_step_2a.index >= start_time) &
        (estimator.QD_step_2a.index < stop_time)
    ]
    plt.plot(nodes.index, nodes.values, ".", label="Daily typical value")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    return fig


def build_step_2b_figure(estimator, view_slice, start_time, stop_time):
    """Build the Step 2b figure for one time window."""
    fig = plt.figure(figsize=(15, 9))
    plt.plot(estimator.df["datetime"][view_slice], estimator.df["x_QD"][view_slice], label="Observed signal without daily variation")
    nodes = estimator.QD_step_2a.loc[
        (estimator.QD_step_2a.index >= start_time) &
        (estimator.QD_step_2a.index < stop_time)
    ]
    plt.plot(nodes.index, nodes.values, ".", label="Daily typical value")
    plt.plot(estimator.df["datetime"][view_slice], estimator.df["QY"][view_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    return fig


def build_step_2c_figure(estimator, view_slice, _start_time, _stop_time):
    """Build the Step 2c figure for one time window."""
    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(estimator.df["datetime"][view_slice], estimator.df["x_QD"][view_slice], label="Without daily variation")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(estimator.df["datetime"][view_slice], estimator.df["x_QD_QY"][view_slice], label="Without daily and yearly variation")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    return fig


def save_step_chunks(
    example_figure_dir,
    estimator,
    t,
    chunk_days,
    finite_values,
    subdir_name,
    figure_builder,
    progress_label=None,
):
    """Write the full-year chunk set for one estimator plot family."""
    save_chunked_estimator_plot(
        example_figure_dir,
        estimator,
        t,
        chunk_days,
        finite_values,
        subdir_name,
        subdir_name,
        figure_builder,
        progress_label=progress_label,
    )
