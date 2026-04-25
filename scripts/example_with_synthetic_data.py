import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from baseline import BaselineEstimator, VarianceEstimator

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
MINUTES_PER_DAY = 24 * 60
HALF_HOURS_PER_DAY = 48


def save_figure(fig, filename):
    """Save and close a matplotlib figure in the repository figures folder."""
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    """Run the synthetic-data example and write diagnostic plots."""
    be_truth = np.load(DATA_DIR / "Be_truth.npy")
    bn_truth = np.load(DATA_DIR / "Bn_truth.npy")
    _bu_truth = np.load(DATA_DIR / "Bu_truth.npy")

    be = np.load(DATA_DIR / "Be.npy")
    bn = np.load(DATA_DIR / "Bn.npy")
    bu = np.load(DATA_DIR / "Bu.npy")

    mlat = np.load(DATA_DIR / "mlat.npy")

    s_since_2000 = np.load(DATA_DIR / "s_since_2000.npy")
    t = np.array([datetime(2000, 1, 1) + timedelta(seconds=float(s)) for s in s_since_2000])

    ve = VarianceEstimator(t, bn, be, bu, mlat)
    ve.estimate()

    be_n = BaselineEstimator(t, bn, ve.df["uN"].values, mlat, component="N")
    be_n.get_baseline(step_1d_sigma_days=1/72)

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][:7 * MINUTES_PER_DAY], be_n.df["x"][:7 * MINUTES_PER_DAY], label="Observed magnetic field")
    plt.plot(be_n.QD_step_1a[:7], ".", label="Daily typical value")
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    plt.legend()
    save_figure(fig, "step_1a.png")

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][:7 * MINUTES_PER_DAY], be_n.df["x"][:7 * MINUTES_PER_DAY], label="Observed magnetic field")
    axs[0].plot(be_n.df["datetime"][:7 * MINUTES_PER_DAY], be_n.df["step_1b"][:7 * MINUTES_PER_DAY], label="Weighted fit to daily typical value")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][:7 * MINUTES_PER_DAY], be_n.df["residual_step_1"][:7 * MINUTES_PER_DAY], label="Difference")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, "step_1b.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], be_n.df["residual_step_1"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c[5 * HALF_HOURS_PER_DAY:7 * HALF_HOURS_PER_DAY], ".", label="Semi-hourly typical values")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    save_figure(fig, "step_1c.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], be_n.df["residual_step_1"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c[5 * HALF_HOURS_PER_DAY:7 * HALF_HOURS_PER_DAY], ".", label="Semi-hourly typical values")
    plt.plot(be_n.df["datetime"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], be_n.df["QD"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    save_figure(fig, "step_1d.png")

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], be_n.df["x"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], label="Observed signal")
    axs[0].plot(be_n.df["datetime"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], be_n.df["x_QD"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], label="Without daily variation")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], (bn_truth + 35000)[5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], label="Truth")
    axs[1].plot(be_n.df["datetime"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], be_n.df["x_QD"][5 * MINUTES_PER_DAY:7 * MINUTES_PER_DAY], label="Retrieved")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, "step_1e.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][:500 * MINUTES_PER_DAY], be_n.df["x_QD"][:500 * MINUTES_PER_DAY], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a[:500], ".", label="Daily typical value")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "step_2a.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][:500 * MINUTES_PER_DAY], be_n.df["x_QD"][:500 * MINUTES_PER_DAY], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a[:500], ".", label="Daily typical value")
    plt.plot(be_n.df["datetime"][:500 * MINUTES_PER_DAY], be_n.df["QY"][:500 * MINUTES_PER_DAY], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "step_2b.png")

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][:500 * MINUTES_PER_DAY], be_n.df["x_QD"][:500 * MINUTES_PER_DAY], label="Without daily variation")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][:500 * MINUTES_PER_DAY], be_n.df["x_QD_QY"][:500 * MINUTES_PER_DAY], label="Without daily and yearly variation")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, "step_2c.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][5 * MINUTES_PER_DAY:5 * MINUTES_PER_DAY + 200], bn_truth[5 * MINUTES_PER_DAY:5 * MINUTES_PER_DAY + 200], label="Truth")
    plt.plot(be_n.df["datetime"][5 * MINUTES_PER_DAY:5 * MINUTES_PER_DAY + 200], be_n.df["x_QD_QY"][5 * MINUTES_PER_DAY:5 * MINUTES_PER_DAY + 200], label="Retrieved")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "step_2d.png")
