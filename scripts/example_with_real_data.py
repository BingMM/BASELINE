from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from apexpy import Apex
from baseline import (
    BaselineEstimator,
    CoordinateRotator,
    InverseCoordinateRotator,
    VarianceEstimator,
)
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "DMH_1min_2025.csv"
GLAT = 76.77
GLON = 341.37
MINUTES_PER_DAY = 24 * 60
HALF_HOURS_PER_DAY = 48


def save_figure(fig, filename):
    """Save and close a matplotlib figure in the repository figures folder."""
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def day_slice(start_day, num_days, num_points):
    """Return a sample slice defined in units of whole days."""
    start = start_day * MINUTES_PER_DAY
    stop = min(num_points, (start_day + num_days) * MINUTES_PER_DAY)
    return slice(start, stop)


def load_real_data(csv_path):
    """Load one local CSV file and return timestamps and raw `X/Y/Z` components."""
    data = pd.read_csv(csv_path)
    data["date_UTC"] = pd.to_datetime(data["date_UTC"])
    t = data["date_UTC"].to_numpy()
    x = data["X"].to_numpy(dtype=float)
    y = data["Y"].to_numpy(dtype=float)
    z = data["Z"].to_numpy(dtype=float)
    return t, x, y, z


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
    """
    Run the local-CSV example and write diagnostic plots.

    This example demonstrates the full `XYZ -> NEZ -> baseline -> XYZ` path:

    1. rotate the raw `X/Y/Z` data into local magnetic `N/E/Z`
    2. estimate baselines on `N`, `E`, and `Z`
    3. rotate the estimated baselines and corrected signals back into `X/Y/Z`

    The detailed Step 1 and Step 2 figures remain focused on the rotated
    northward component so they are easy to compare with the paper.
    """
    t, x, y, z = load_real_data(CSV_PATH)
    mlat = compute_mlat(t, GLAT, GLON)

    rotator = CoordinateRotator(t, x, y, z)
    rotator.rotate()
    bn, be, bu = rotator.get_components()

    ve = VarianceEstimator(t, bn, be, bu, mlat)
    ve.estimate()

    # Estimate baselines for all three rotated components before mapping the
    # result back into the original XYZ frame.
    be_e = BaselineEstimator(t, be, ve.df["uE"].values, mlat, component="E")
    be_e.get_baseline()

    be_n = BaselineEstimator(t, bn, ve.df["uN"].values, mlat, component="N")
    be_n.get_baseline()

    be_u = BaselineEstimator(t, bu, ve.df["uZ"].values, mlat, component="Z")
    be_u.get_baseline()

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
    save_figure(fig, "real_rotation.png")

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
    save_figure(fig, "real_xyz_baseline.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][short_slice], be_n.df["x"][short_slice], label="Observed magnetic field")
    plt.plot(be_n.QD_step_1a.iloc[:min(7, len(be_n.QD_step_1a))], ".", label="Daily typical value")
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    plt.legend()
    save_figure(fig, "real_step_1a.png")

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][short_slice], be_n.df["x"][short_slice], label="Observed magnetic field")
    axs[0].plot(be_n.df["datetime"][short_slice], be_n.df["step_1b"][short_slice], label="Weighted fit to daily typical value")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][short_slice], be_n.df["residual_step_1"][short_slice], label="Difference")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, "real_step_1b.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["residual_step_1"][detail_slice], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c.iloc[detail_half_hour_slice], ".", label="Semi-hourly typical values")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "real_step_1c.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["residual_step_1"][detail_slice], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c.iloc[detail_half_hour_slice], ".", label="Semi-hourly typical values")
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["QD"][detail_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "real_step_1d.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["x"][detail_slice], label="Observed signal")
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["x_QD"][detail_slice], label="Without daily variation")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    save_figure(fig, "real_step_1e.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a.iloc[:long_days], ".", label="Daily typical value")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "real_step_2a.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a.iloc[:long_days], ".", label="Daily typical value")
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["QY"][long_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "real_step_2b.png")

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Without daily variation")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][long_slice], be_n.df["x_QD_QY"][long_slice], label="Without daily and yearly variation")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, "real_step_2c.png")
