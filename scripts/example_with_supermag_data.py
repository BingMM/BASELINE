from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from apexpy import Apex
from baseline import BaselineEstimator, VarianceEstimator
from tqdm import tqdm
import netCDF4 as nc
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
EXAMPLE_FIGURE_DIR = FIGURE_DIR / "SM_example"
FIGURE_DIR.mkdir(exist_ok=True)
EXAMPLE_FIGURE_DIR.mkdir(exist_ok=True)

SM_PATH_no_BS = DATA_DIR / "DMH_SM_1min_2024_no_BS.netcdf"
SM_PATH_no_QD = DATA_DIR / "DMH_SM_1min_2024_no_QD.netcdf"
SM_PATH = DATA_DIR / "DMH_SM_1min_2024.netcdf"
GLAT = 76.77
GLON = 341.37
MINUTES_PER_DAY = 24 * 60
HALF_HOURS_PER_DAY = 48
STEP_1_CONTEXT_CHUNK_DAYS = 7
STEP_1_DETAIL_CHUNK_DAYS = 2
STEP_2_CHUNK_DAYS = 60
STEP_2B_SIGMA_DAYS = 15.0
STEP_1C_CHECKPOINT_DIR = DATA_DIR / "cache" / "step_1c"
REUSE_STEP_1C_CHECKPOINT = True
WRITE_STEP_1C_CHECKPOINT = True
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


def save_figure(fig, filename):
    """Save and close a matplotlib figure in the repository figures folder."""
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_example_figure(fig, subdir_name, filename):
    """Save and close a figure in a named SM_example subdirectory."""
    save_path = EXAMPLE_FIGURE_DIR / subdir_name / filename
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


def step_1c_checkpoint_path(component, min_window_days):
    """Return a stable checkpoint filename for one component configuration."""
    filename = (
        f"DMH_2024_{component}_"
        "paper_mode_"
        f"w{int(min_window_days)}_step1c.pkl"
    )
    return STEP_1C_CHECKPOINT_DIR / filename


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


def save_chunked_component_triplet(t, components, titles, subdir_name, prefix, chunk_days):
    """Write one three-panel component plot per time chunk."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not any_series_has_finite(components, view_slice):
            continue
        fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        for ax, values, title in zip(axs, components, titles):
            ax.plot(t[view_slice], values[view_slice])
            ax.set_title(title)
        save_example_figure(fig, subdir_name, chunk_filename(prefix, start_time, stop_time))


def save_chunked_component_comparison(
    t,
    component_pairs,
    titles,
    subdir_name,
    prefix,
    chunk_days,
):
    """Write one three-panel comparison plot per time chunk."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        reference_series = [reference for reference, _ in component_pairs]
        if not any_series_has_finite(reference_series, view_slice):
            continue
        fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        for ax, (reference, estimate), title in zip(axs, component_pairs, titles):
            ax.plot(t[view_slice], reference[view_slice])
            if isinstance(estimate, pd.Series):
                estimate_view = estimate.iloc[view_slice].values
            else:
                estimate_view = np.asarray(estimate)[view_slice]
            ax.plot(t[view_slice], estimate_view)
            ax.set_title(title)
        save_example_figure(fig, subdir_name, chunk_filename(prefix, start_time, stop_time))


def save_chunked_step_1a(estimator, t, chunk_days):
    """Write the Step 1a overview in chunks covering the full record."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not slice_has_finite(estimator.df["x"].values, view_slice):
            continue
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
        save_example_figure(fig, "SM_step_1a", chunk_filename("SM_step_1a", start_time, stop_time))


def save_chunked_step_1b(estimator, t, chunk_days):
    """Write the Step 1b plot in chunks covering the full record."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not slice_has_finite(estimator.df["x"].values, view_slice):
            continue
        fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        axs[0].plot(estimator.df["datetime"][view_slice], estimator.df["x"][view_slice], label="Observed magnetic field")
        axs[0].plot(estimator.df["datetime"][view_slice], estimator.df["step_1b"][view_slice], label="Weighted fit to daily typical value")
        axs[0].legend()
        axs[0].set_ylabel("Magnetic field [nT]")

        axs[1].plot(estimator.df["datetime"][view_slice], estimator.df["residual_step_1"][view_slice], label="Difference")
        axs[1].legend()
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Magnetic field [nT]")
        save_example_figure(fig, "SM_step_1b", chunk_filename("SM_step_1b", start_time, stop_time))


def save_chunked_step_1c(estimator, t, chunk_days):
    """Write the Step 1c plot in chunks covering the full record."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not slice_has_finite(estimator.df["x"].values, view_slice):
            continue
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
        save_example_figure(fig, "SM_step_1c", chunk_filename("SM_step_1c", start_time, stop_time))


def save_chunked_step_1d(estimator, t, chunk_days):
    """Write the Step 1d plot in chunks covering the full record."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not slice_has_finite(estimator.df["x"].values, view_slice):
            continue
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
        save_example_figure(fig, "SM_step_1d", chunk_filename("SM_step_1d", start_time, stop_time))


def save_chunked_step_1e(estimator, t, chunk_days):
    """Write the Step 1e plot in chunks covering the full record."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not slice_has_finite(estimator.df["x"].values, view_slice):
            continue
        fig = plt.figure(figsize=(15, 9))
        plt.plot(estimator.df["datetime"][view_slice], estimator.df["x"][view_slice], label="Observed signal")
        plt.plot(estimator.df["datetime"][view_slice], estimator.df["x_QD"][view_slice], label="Without daily variation")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Magnetic field [nT]")
        save_example_figure(fig, "SM_step_1e", chunk_filename("SM_step_1e", start_time, stop_time))


def save_chunked_step_2a(estimator, t, chunk_days):
    """Write the Step 2a plot in chunks covering the full record."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not slice_has_finite(estimator.df["x_QD"].values, view_slice):
            continue
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
        save_example_figure(fig, "SM_step_2a", chunk_filename("SM_step_2a", start_time, stop_time))


def save_chunked_step_2b(estimator, t, chunk_days):
    """Write the Step 2b plot in chunks covering the full record."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not slice_has_finite(estimator.df["x_QD"].values, view_slice):
            continue
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
        save_example_figure(fig, "SM_step_2b", chunk_filename("SM_step_2b", start_time, stop_time))


def save_chunked_step_2c(estimator, t, chunk_days):
    """Write the Step 2c plot in chunks covering the full record."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not slice_has_finite(estimator.df["x_QD"].values, view_slice):
            continue
        fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        axs[0].plot(estimator.df["datetime"][view_slice], estimator.df["x_QD"][view_slice], label="Without daily variation")
        axs[0].legend()
        axs[0].set_ylabel("Magnetic field [nT]")

        axs[1].plot(estimator.df["datetime"][view_slice], estimator.df["x_QD_QY"][view_slice], label="Without daily and yearly variation")
        axs[1].legend()
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Magnetic field [nT]")
        save_example_figure(fig, "SM_step_2c", chunk_filename("SM_step_2c", start_time, stop_time))


def save_chunked_qd_component_comparison(t, input_values, supermag_qd, estimator, subdir_name, prefix, chunk_days, error_plot):
    """Write one component-comparison plot per chunk using the existing QD helper."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        if not slice_has_finite(input_values, view_slice):
            continue
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        plot_qd_component(ax, t, input_values, supermag_qd, estimator, view_slice, estimator.component, error_plot=error_plot)
        ax.legend(loc="best")
        save_example_figure(fig, subdir_name, chunk_filename(prefix, start_time, stop_time))


def save_chunked_qd_comp_triplet(t, component_specs, chunk_days, error_plot):
    """Write the three-component QD comparison plot per chunk."""
    for start_time, stop_time, view_slice in iter_time_chunks(t, chunk_days):
        input_series = [input_values for input_values, _, _, _ in component_specs]
        if not any_series_has_finite(input_series, view_slice):
            continue
        fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        for ax, (input_values, supermag_qd, estimator, title) in zip(axs, component_specs):
            plot_qd_component(ax, t, input_values, supermag_qd, estimator, view_slice, title, error_plot=error_plot)
        axs[0].legend(loc="best")
        save_example_figure(fig, "SM_QD_comp", chunk_filename("SM_QD_comp", start_time, stop_time))


def load_real_data(sm_path, sm_path_no_QD, sm_path_no_BS):
    """Load the DMH CSV file and return timestamps and XYZ components."""
    dataset_no_BS = nc.Dataset(sm_path_no_BS, 'r')
    dataset_no_QD = nc.Dataset(sm_path_no_QD, 'r')
    dataset = nc.Dataset(sm_path, 'r')
    
    t = np.array([datetime(int(yy), int(mm), int(dd), int(HH), int(MM), int(SS)) for yy, mm, dd, HH, MM, SS in zip(dataset.variables['time_yr'][:].filled(np.nan), dataset.variables['time_mo'][:].filled(np.nan), dataset.variables['time_dy'][:].filled(np.nan), dataset.variables['time_hr'][:].filled(np.nan), dataset.variables['time_mt'][:].filled(np.nan), dataset.variables['time_sc'][:].filled(np.nan))])
    
    be_no_BS =   dataset_no_BS.variables['dbe_geo'][:].filled(np.nan).flatten()
    bn_no_BS =   dataset_no_BS.variables['dbn_geo'][:].filled(np.nan).flatten()
    bu_no_BS = - dataset_no_BS.variables['dbz_geo'][:].filled(np.nan).flatten()
    
    be_no_QD =   dataset_no_QD.variables['dbe_geo'][:].filled(np.nan).flatten()
    bn_no_QD =   dataset_no_QD.variables['dbn_geo'][:].filled(np.nan).flatten()
    bu_no_QD = - dataset_no_QD.variables['dbz_geo'][:].filled(np.nan).flatten()
    
    be =   dataset.variables['dbe_geo'][:].filled(np.nan).flatten()
    bn =   dataset.variables['dbn_geo'][:].filled(np.nan).flatten()
    bu = - dataset.variables['dbz_geo'][:].filled(np.nan).flatten()
    
    dataset.close()
    dataset_no_QD.close()
    dataset_no_BS.close()
    
    be_QD = be_no_QD - be
    bn_QD = bn_no_QD - bn
    bu_QD = bu_no_QD - bu
    
    be_QY = be_no_BS - be_no_QD
    bn_QY = bn_no_BS - bn_no_QD
    bu_QY = bu_no_BS - bu_no_QD
    
    
    return t, be, bn, bu, be_QD, bn_QD, bu_QD, be_QY, bn_QY, bu_QY, be_no_BS, bn_no_BS, bu_no_BS


def compute_mlat(t, glat, glon):
    """Compute magnetic latitude for each timestamp."""
    mlat = np.zeros(t.size, dtype=float)
    apex_by_year = {}

    for i, ti in tqdm(enumerate(t), total=t.size, desc="Compute mlat"):
        year = pd.Timestamp(ti).year
        apex = apex_by_year.setdefault(year, Apex(year, refh=0))
        mlat[i] = apex.convert(glat, glon, "geo", "apex", height=0)[0]

    return mlat

if __name__ == "__main__":
    """Run the real-data example and write diagnostic plots."""
    t, dbe, dbn, dbu, be_QD, bn_QD, bu_QD, be_QY, bn_QY, bu_QY, be, bn, bu = load_real_data(SM_PATH, SM_PATH_no_QD, SM_PATH_no_BS)
    
    d_start = pd.Timestamp("2024-03-06")
    error_plot = True
    n_points = len(t)
    short_slice = day_slice(0, min(7, max(1, n_points // MINUTES_PER_DAY)), n_points)
    detail_days = 2 if n_points >= 2 * MINUTES_PER_DAY else max(1, n_points // MINUTES_PER_DAY)
    detail_slice = date_slice(t, d_start, detail_days)
    detail_stop = d_start + pd.Timedelta(days=detail_days)
    long_days = min(180, max(1, n_points // MINUTES_PER_DAY))
    long_slice = day_slice(0, long_days, n_points)
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(t[detail_slice], be[detail_slice])
    axs[1].plot(t[detail_slice], bn[detail_slice])
    axs[2].plot(t[detail_slice], bu[detail_slice])
    axs[0].set_title('Be')
    axs[1].set_title('Bn')
    axs[2].set_title('Bu')
    save_figure(fig, "SM.png")
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(t[detail_slice], dbe[detail_slice])
    axs[1].plot(t[detail_slice], dbn[detail_slice])
    axs[2].plot(t[detail_slice], dbu[detail_slice])
    axs[0].set_title('Be')
    axs[1].set_title('Bn')
    axs[2].set_title('Bu')
    save_figure(fig, "SM_db.png")

    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(t[detail_slice], be_QD[detail_slice])
    axs[1].plot(t[detail_slice], bn_QD[detail_slice])
    axs[2].plot(t[detail_slice], bu_QD[detail_slice])
    axs[0].set_title('Be')
    axs[1].set_title('Bn')
    axs[2].set_title('Bu')
    save_figure(fig, "SM_QD.png")
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(t[detail_slice], be_QY[detail_slice])
    axs[1].plot(t[detail_slice], bn_QY[detail_slice])
    axs[2].plot(t[detail_slice], bu_QY[detail_slice])
    axs[0].set_title('Be')
    axs[1].set_title('Bn')
    axs[2].set_title('Bu')
    save_figure(fig, "SM_QY.png")

    save_chunked_component_triplet(
        t,
        (be, bn, bu),
        ("Be", "Bn", "Bu"),
        "SM",
        "SM",
        STEP_1_CONTEXT_CHUNK_DAYS,
    )
    save_chunked_component_triplet(
        t,
        (dbe, dbn, dbu),
        ("Be", "Bn", "Bu"),
        "SM_db",
        "SM_db",
        STEP_1_CONTEXT_CHUNK_DAYS,
    )
    save_chunked_component_triplet(
        t,
        (be_QD, bn_QD, bu_QD),
        ("Be", "Bn", "Bu"),
        "SM_QD",
        "SM_QD",
        STEP_1_DETAIL_CHUNK_DAYS,
    )
    save_chunked_component_triplet(
        t,
        (be_QY, bn_QY, bu_QY),
        ("Be", "Bn", "Bu"),
        "SM_QY",
        "SM_QY",
        STEP_2_CHUNK_DAYS,
    )
    
    mlat = compute_mlat(t, GLAT, GLON)

    ve = VarianceEstimator(t, bn, be, bu, mlat)
    ve.estimate()

    be_e = BaselineEstimator(t, be, ve.df["uE"].values, mlat, component="E", 
                             step_1c_min_window_days=5,
                             step_1c_plot_diagnostics=True,
                             step_1c_diagnostic_time_range=(d_start, detail_stop),
                             step_1c_plot_dir="figures/QD_diag")
    be_e_step_1c_checkpoint = step_1c_checkpoint_path(
        component="E",
        min_window_days=be_e.step_1c_min_window_days,
    )
    be_e.get_baseline(step_1d_sigma_days=1/12,
                      step_1d_adaptive_sigma=False,
                      step_1d_max_sigma_multiplier=6,
                      step_2b_sigma_days=STEP_2B_SIGMA_DAYS,
                      step_1c_checkpoint_path=be_e_step_1c_checkpoint,
                      reuse_step_1c_checkpoint=REUSE_STEP_1C_CHECKPOINT,
                      write_step_1c_checkpoint=WRITE_STEP_1C_CHECKPOINT)

    be_n = BaselineEstimator(t, bn, ve.df["uN"].values, mlat, component="N", 
                             step_1c_min_window_days=5,
                             step_1c_plot_diagnostics=True,
                             step_1c_diagnostic_time_range=(d_start, detail_stop),
                             step_1c_plot_dir="figures/QD_diag")
    be_n_step_1c_checkpoint = step_1c_checkpoint_path(
        component="N",
        min_window_days=be_n.step_1c_min_window_days,
    )
    be_n.get_baseline(step_1d_sigma_days=1/12,
                      step_1d_adaptive_sigma=False,
                      step_1d_max_sigma_multiplier=6,
                      step_2b_sigma_days=STEP_2B_SIGMA_DAYS,
                      step_1c_checkpoint_path=be_n_step_1c_checkpoint,
                      reuse_step_1c_checkpoint=REUSE_STEP_1C_CHECKPOINT,
                      write_step_1c_checkpoint=WRITE_STEP_1C_CHECKPOINT)
    
    be_u = BaselineEstimator(t, bu, ve.df["uZ"].values, mlat, component="Z", 
                             step_1c_min_window_days=5,
                             step_1c_plot_diagnostics=True,
                             step_1c_diagnostic_time_range=(d_start, detail_stop),
                             step_1c_plot_dir="figures/QD_diag")
    be_u_step_1c_checkpoint = step_1c_checkpoint_path(
        component="Z",
        min_window_days=be_u.step_1c_min_window_days,
    )
    be_u.get_baseline(step_1d_sigma_days=1/12,
                      step_1d_adaptive_sigma=False,
                      step_1d_max_sigma_multiplier=6,
                      step_2b_sigma_days=STEP_2B_SIGMA_DAYS,
                      step_1c_checkpoint_path=be_u_step_1c_checkpoint,
                      reuse_step_1c_checkpoint=REUSE_STEP_1C_CHECKPOINT,
                      write_step_1c_checkpoint=WRITE_STEP_1C_CHECKPOINT)

    detail_half_hour_mask = (
        (be_n.QD_step_1c.index >= d_start) &
        (be_n.QD_step_1c.index < detail_stop)
    )

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][short_slice], be_n.df["x"][short_slice], label="Observed magnetic field")
    plt.plot(be_n.QD_step_1a.iloc[:min(7, len(be_n.QD_step_1a))], ".", label="Daily typical value")
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    plt.legend()
    save_figure(fig, "SM_step_1a.png")
    save_chunked_step_1a(be_n, t, STEP_1_CONTEXT_CHUNK_DAYS)

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][short_slice], be_n.df["x"][short_slice], label="Observed magnetic field")
    axs[0].plot(be_n.df["datetime"][short_slice], be_n.df["step_1b"][short_slice], label="Weighted fit to daily typical value")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][short_slice], be_n.df["residual_step_1"][short_slice], label="Difference")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, "SM_step_1b.png")
    save_chunked_step_1b(be_n, t, STEP_1_CONTEXT_CHUNK_DAYS)

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["residual_step_1"][detail_slice], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c.loc[detail_half_hour_mask], ".", label="Semi-hourly typical values")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "SM_step_1c.png")
    save_chunked_step_1c(be_n, t, STEP_1_DETAIL_CHUNK_DAYS)

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["residual_step_1"][detail_slice], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c.loc[detail_half_hour_mask], ".", label="Semi-hourly typical values")
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["QD"][detail_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "SM_step_1d.png")
    save_chunked_step_1d(be_n, t, STEP_1_DETAIL_CHUNK_DAYS)

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["x"][detail_slice], label="Observed signal")
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["x_QD"][detail_slice], label="Without daily variation")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    save_figure(fig, "SM_step_1e.png")
    save_chunked_step_1e(be_n, t, STEP_1_CONTEXT_CHUNK_DAYS)

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a.iloc[:long_days], ".", label="Daily typical value")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "SM_step_2a.png")
    save_chunked_step_2a(be_n, t, STEP_2_CHUNK_DAYS)

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a.iloc[:long_days], ".", label="Daily typical value")
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["QY"][long_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "SM_step_2b.png")
    save_chunked_step_2b(be_n, t, STEP_2_CHUNK_DAYS)

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Without daily variation")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][long_slice], be_n.df["x_QD_QY"][long_slice], label="Without daily and yearly variation")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, "SM_step_2c.png")
    save_chunked_step_2c(be_n, t, STEP_2_CHUNK_DAYS)
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(t[detail_slice], dbe[detail_slice])
    axs[0].plot(t[detail_slice], be_e.df["x_QD_QY"][detail_slice].values)
    axs[1].plot(t[detail_slice], dbn[detail_slice])
    axs[1].plot(t[detail_slice], be_n.df["x_QD_QY"][detail_slice].values)
    axs[2].plot(t[detail_slice], dbu[detail_slice])
    axs[2].plot(t[detail_slice], be_u.df["x_QD_QY"][detail_slice].values)
    axs[0].set_title('Be')
    axs[1].set_title('Bn')
    axs[2].set_title('Bu')
    save_figure(fig, "SM_db_comp.png")
    save_chunked_component_comparison(
        t,
        (
            (dbe, be_e.df["x_QD_QY"]),
            (dbn, be_n.df["x_QD_QY"]),
            (dbu, be_u.df["x_QD_QY"]),
        ),
        ("Be", "Bn", "Bu"),
        "SM_db_comp",
        "SM_db_comp",
        STEP_1_DETAIL_CHUNK_DAYS,
    )

    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    plot_qd_component(axs[0], t, be, be_QD, be_e, detail_slice, "Be", error_plot=error_plot)
    plot_qd_component(axs[1], t, bn, bn_QD, be_n, detail_slice, "Bn", error_plot=error_plot)
    plot_qd_component(axs[2], t, bu, bu_QD, be_u, detail_slice, "Bu", error_plot=error_plot)
    axs[0].legend(loc="best")
    save_figure(fig, "SM_QD_comp.png")
    save_chunked_qd_comp_triplet(
        t,
        (
            (be, be_QD, be_e, "Be"),
            (bn, bn_QD, be_n, "Bn"),
            (bu, bu_QD, be_u, "Bu"),
        ),
        STEP_1_DETAIL_CHUNK_DAYS,
        error_plot,
    )
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(t[detail_slice], be_QY[detail_slice])
    axs[0].plot(t[detail_slice], be_e.df["QY"][detail_slice].values)
    axs[1].plot(t[detail_slice], bn_QY[detail_slice])
    axs[1].plot(t[detail_slice], be_n.df["QY"][detail_slice].values)
    axs[2].plot(t[detail_slice], bu_QY[detail_slice])
    axs[2].plot(t[detail_slice], be_u.df["QY"][detail_slice].values)
    axs[0].set_title('Be')
    axs[1].set_title('Bn')
    axs[2].set_title('Bu')
    save_figure(fig, "SM_QY_comp.png")
    save_chunked_component_comparison(
        t,
        (
            (be_QY, be_e.df["QY"]),
            (bn_QY, be_n.df["QY"]),
            (bu_QY, be_u.df["QY"]),
        ),
        ("Be", "Bn", "Bu"),
        "SM_QY_comp",
        "SM_QY_comp",
        STEP_2_CHUNK_DAYS,
    )
