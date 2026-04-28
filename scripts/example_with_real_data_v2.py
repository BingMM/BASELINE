from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from apexpy import Apex

from baseline import CoordinateRotator, InverseCoordinateRotator
from baseline_v2 import ModernBaselineConfig, ModernBaselineEngine, ModernVarianceEngine
from baseline_v2.types import BaselineInputs, BaselineResult, VarianceInputs

BASE_DIR = REPO_ROOT
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures" / "real_data_v2"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "DMH_1min_2025.csv"
STEP_1C_CHECKPOINT_DIR = DATA_DIR / "cache" / "step_1c_v2"
GLAT = 76.77
GLON = 341.37
CADENCE_SECONDS = 60
MINUTES_PER_DAY = 24 * 60
HALF_HOURS_PER_DAY = 48
STEP_2B_SIGMA_DAYS = 15.0
STEP_1D_SIGMA_DAYS = 1 / 24
STEP_1C_MIN_WINDOW_DAYS = 5
REUSE_STEP_1C_CHECKPOINT = False
WRITE_STEP_1C_CHECKPOINT = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local CSV example with the baseline_v2 engine."
    )
    parser.add_argument(
        "--mode",
        choices=("reference", "robust"),
        default="robust",
        help="Which Step 1c mode to run.",
    )
    return parser.parse_args()


def save_figure(fig, filename):
    """Save and close a matplotlib figure in the repository figures folder."""
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def day_slice(start_day, num_days, num_points):
    """Return a sample slice defined in units of whole days."""
    start = start_day * MINUTES_PER_DAY
    stop = min(num_points, (start_day + num_days) * MINUTES_PER_DAY)
    return slice(start, stop)


def step_1c_checkpoint_path(csv_path: Path, component: str, mode: str, min_window_days: int) -> Path:
    filename = f"{csv_path.stem}_{mode}_{component}_w{int(min_window_days)}_step1c.pkl"
    return STEP_1C_CHECKPOINT_DIR / filename


def load_real_data(csv_path):
    """Load one local CSV file and return timestamps and raw `X/Y/Z` components."""
    data = pd.read_csv(csv_path)
    data["date_UTC"] = pd.to_datetime(data["date_UTC"])
    t = data["date_UTC"].to_numpy()
    x = data["X"].to_numpy(dtype=float)
    y = data["Y"].to_numpy(dtype=float)
    z = data["Z"].to_numpy(dtype=float)
    return t, x, y, z


def compute_station_mlat(t, glat, glon):
    """Compute one scalar magnetic latitude for a single-year record."""
    years = np.unique(pd.to_datetime(t).year)
    if years.size != 1:
        raise ValueError("example_with_real_data_v2.py expects a single-year record")
    apex = Apex(int(years[0]), refh=0)
    return float(apex.convert(glat, glon, "geo", "apex", height=0)[0])


def build_plot_estimator(result: BaselineResult) -> SimpleNamespace:
    """Adapt a V2 result to the plotting interface used by the legacy example."""
    diagnostics = result.diagnostics
    t = pd.to_datetime(result.t)
    x_qd = result.x - result.qd
    df = pd.DataFrame(
        {
            "datetime": t,
            "x": result.x,
            "u": result.u,
            "step_1b": diagnostics["step_1b_value"],
            "residual_step_1": diagnostics["residual_step_1"],
            "QD": result.qd,
            "QY": result.qy,
            "x_QD": x_qd,
            "x_QD_QY": result.residual,
        }
    )
    return SimpleNamespace(
        component=result.component,
        df=df,
        QD_step_1a=pd.Series(
            diagnostics["step_1a_value"],
            index=pd.to_datetime(diagnostics["step_1a_t"]),
        ).sort_index(),
        QD_step_1c=pd.Series(
            result.step1c.value,
            index=pd.to_datetime(result.step1c.t),
        ).sort_index(),
        QD_step_1c_w=pd.Series(
            result.step1c.weight,
            index=pd.to_datetime(result.step1c.t),
        ).sort_index(),
        QD_step_1c_status=pd.Series(
            result.step1c.status,
            index=pd.to_datetime(result.step1c.t),
        ).sort_index(),
        QD_step_2a=pd.Series(
            diagnostics["step_2a_value"],
            index=pd.to_datetime(diagnostics["step_2a_t"]),
        ).sort_index(),
        QD_step_2a_w=pd.Series(
            diagnostics["step_2a_weight"],
            index=pd.to_datetime(diagnostics["step_2a_t"]),
        ).sort_index(),
        y_smooth=np.asarray(diagnostics["step_1d_node_value"], dtype=float),
    )


if __name__ == "__main__":
    """
    Run the local-CSV example and write diagnostic plots with the V2 engine.

    This example demonstrates the full `XYZ -> NEZ -> baseline -> XYZ` path:

    1. rotate the raw `X/Y/Z` data into local magnetic `N/E/Z`
    2. estimate baselines on `N`, `E`, and `Z` with `baseline_v2`
    3. rotate the estimated baselines and corrected signals back into `X/Y/Z`

    The detailed Step 1 and Step 2 figures remain focused on the rotated
    northward component so they are easy to compare with the paper.
    """
    args = parse_args()
    t, x, y, z = load_real_data(CSV_PATH)
    station_mlat = compute_station_mlat(t, GLAT, GLON)

    rotator = CoordinateRotator(t, x, y, z)
    rotator.rotate()
    bn, be, bu = rotator.get_components()

    variance = ModernVarianceEngine().fit(
        VarianceInputs(
            t=t,
            n=bn,
            e=be,
            z=bu,
            mlat=station_mlat,
            cadence_seconds=CADENCE_SECONDS,
        )
    )

    be_e = None
    be_n = None
    be_u = None

    for component, values in (("E", be), ("N", bn), ("Z", bu)):
        result = ModernBaselineEngine(
            ModernBaselineConfig(
                step_1c_method=args.mode,
                step_1d_sigma_days=STEP_1D_SIGMA_DAYS,
                step_2b_sigma_days=STEP_2B_SIGMA_DAYS,
                step_1c_min_window_days=STEP_1C_MIN_WINDOW_DAYS,
                step_1c_checkpoint_path=step_1c_checkpoint_path(
                    CSV_PATH,
                    component,
                    args.mode,
                    STEP_1C_MIN_WINDOW_DAYS,
                ),
                reuse_step_1c_checkpoint=REUSE_STEP_1C_CHECKPOINT,
                write_step_1c_checkpoint=WRITE_STEP_1C_CHECKPOINT,
                verbose=True,
                progress_label=component,
            )
        ).fit_component(
            BaselineInputs(
                t=t,
                x=values,
                component=component,
                mlat=station_mlat,
                cadence_seconds=CADENCE_SECONDS,
            ),
            variance,
        )
        estimator = build_plot_estimator(result)
        if component == "E":
            be_e = estimator
        elif component == "N":
            be_n = estimator
        else:
            be_u = estimator

    inverse_rotator = InverseCoordinateRotator(rotator)
    xyz_products = inverse_rotator.rotate_baselines(be_e, be_n, be_u)

    n_points = len(t)
    short_slice = day_slice(0, min(7, max(1, n_points // MINUTES_PER_DAY)), n_points)
    detail_start_day = 5 if n_points >= 7 * MINUTES_PER_DAY else 0
    detail_days = 2 if n_points >= 2 * MINUTES_PER_DAY else max(1, n_points // MINUTES_PER_DAY)
    detail_slice = day_slice(detail_start_day, detail_days, n_points)
    detail_half_hour_slice = slice(
        detail_start_day * HALF_HOURS_PER_DAY,
        min(len(be_n.QD_step_1c), (detail_start_day + detail_days) * HALF_HOURS_PER_DAY),
    )
    long_days = min(180, max(1, n_points // MINUTES_PER_DAY))
    long_slice = day_slice(0, long_days, n_points)

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(t[short_slice], x[short_slice], label="X")
    axs[0].plot(t[short_slice], rotator.df["N"].values[short_slice], label="N")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(t[short_slice], y[short_slice], label="Y")
    axs[1].plot(t[short_slice], rotator.df["E"].values[short_slice], label="E")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, f"real_rotation_{args.mode}.png")

    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(t[detail_slice], xyz_products["X"][detail_slice], label="X")
    axs[0].plot(t[detail_slice], xyz_products["X_corr"][detail_slice], label="X corrected")
    axs[0].plot(t[detail_slice], xyz_products["baseline_X"][detail_slice], label="X baseline")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(t[detail_slice], xyz_products["Y"][detail_slice], label="Y")
    axs[1].plot(t[detail_slice], xyz_products["Y_corr"][detail_slice], label="Y corrected")
    axs[1].plot(t[detail_slice], xyz_products["baseline_Y"][detail_slice], label="Y baseline")
    axs[1].legend()
    axs[1].set_ylabel("Magnetic field [nT]")

    axs[2].plot(t[detail_slice], xyz_products["Z"][detail_slice], label="Z")
    axs[2].plot(t[detail_slice], xyz_products["Z_corr"][detail_slice], label="Z corrected")
    axs[2].plot(t[detail_slice], xyz_products["baseline_Z"][detail_slice], label="Z baseline")
    axs[2].legend()
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Magnetic field [nT]")
    save_figure(fig, f"real_xyz_baseline_{args.mode}.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][short_slice], be_n.df["x"][short_slice], label="Observed magnetic field")
    plt.plot(be_n.QD_step_1a.iloc[:min(7, len(be_n.QD_step_1a))], ".", label="Daily typical value")
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    plt.legend()
    save_figure(fig, f"real_step_1a_{args.mode}.png")

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][short_slice], be_n.df["x"][short_slice], label="Observed magnetic field")
    axs[0].plot(be_n.df["datetime"][short_slice], be_n.df["step_1b"][short_slice], label="Weighted fit to daily typical value")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][short_slice], be_n.df["residual_step_1"][short_slice], label="Difference")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, f"real_step_1b_{args.mode}.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["residual_step_1"][detail_slice], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c.iloc[detail_half_hour_slice], ".", label="Semi-hourly typical values")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, f"real_step_1c_{args.mode}.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["residual_step_1"][detail_slice], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c.iloc[detail_half_hour_slice], ".", label="Semi-hourly typical values")
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["QD"][detail_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, f"real_step_1d_{args.mode}.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["x"][detail_slice], label="Observed signal")
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["x_QD"][detail_slice], label="Without daily variation")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    save_figure(fig, f"real_step_1e_{args.mode}.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a.iloc[:long_days], ".", label="Daily typical value")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, f"real_step_2a_{args.mode}.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a.iloc[:long_days], ".", label="Daily typical value")
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["QY"][long_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, f"real_step_2b_{args.mode}.png")

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Without daily variation")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][long_slice], be_n.df["x_QD_QY"][long_slice], label="Without daily and yearly variation")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, f"real_step_2c_{args.mode}.png")
