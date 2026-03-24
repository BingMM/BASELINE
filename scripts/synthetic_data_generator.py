import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from apexpy import Apex

#%% Paths

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
MINUTES_PER_DAY = 24 * 60


def save_figure(fig, filename):
    """Save and close a matplotlib figure in the repository figures folder."""
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def main():
    """Generate synthetic input data and a reference overview plot."""
    t_start = datetime(2000, 1, 1)
    t_duration = 2 * 365 * MINUTES_PER_DAY
    t = np.array([t_start + timedelta(seconds=i * 60) for i in range(t_duration)])

    glat, glon = 65, 0
    mlat = np.zeros(t.size)
    s_since_2000 = np.zeros(t.size).astype(int)
    apex_by_year = {}

    for i, ti in tqdm(enumerate(t), total=t.size, desc="Generate mlat and t"):
        apex = apex_by_year.setdefault(ti.year, Apex(ti.year, refh=0))
        mlat[i] = apex.convert(glat, glon, "geo", "apex", height=0)[0]
        s_since_2000[i] = (ti - datetime(2000, 1, 1)).total_seconds()

    be = 5 * np.random.normal(0, 1, t.size)
    bn = 5 * np.random.normal(0, 1, t.size)
    bu = 5 * np.random.normal(0, 1, t.size)

    q = np.arange(MINUTES_PER_DAY)
    daily = 15e3 / (150 * np.sqrt(2)) * np.exp(-0.5 * ((q - 576) ** 2 / (150**2)))
    daily = np.tile(daily, 2 * 365)

    yearly = 5 * np.sin(np.arange(t.size) / (365 * MINUTES_PER_DAY) * np.pi)

    offset = 35000

    be_comp = be + daily + yearly + offset
    bn_comp = bn + daily + yearly + offset
    bu_comp = bu + daily + yearly + offset

    points = 10 * MINUTES_PER_DAY
    t_plot = t[:points]

    fig, axs = plt.subplots(4, 3, figsize=(15, 10), sharex=True)

    axs[0, 0].plot(t_plot, be_comp[:points])
    axs[0, 0].set_title("Be")
    axs[0, 1].plot(t_plot, bn_comp[:points])
    axs[0, 1].set_title("Bn")
    axs[0, 2].plot(t_plot, bu_comp[:points])
    axs[0, 2].set_title("Bu")

    axs[1, 0].plot(t_plot, be[:points])
    axs[1, 0].set_title("Signal")
    axs[1, 1].plot(t_plot, bn[:points])
    axs[1, 1].set_title("Signal")
    axs[1, 2].plot(t_plot, bu[:points])
    axs[1, 2].set_title("Signal")

    axs[2, 0].plot(t_plot, daily[:points])
    axs[2, 0].set_title("Daily variation")
    axs[2, 1].plot(t_plot, daily[:points])
    axs[2, 1].set_title("Daily variation")
    axs[2, 2].plot(t_plot, daily[:points])
    axs[2, 2].set_title("Daily variation")

    axs[3, 0].plot(t_plot, yearly[:points])
    axs[3, 0].set_title("Yearly variation")
    axs[3, 1].plot(t_plot, yearly[:points])
    axs[3, 1].set_title("Yearly variation")
    axs[3, 2].plot(t_plot, yearly[:points])
    axs[3, 2].set_title("Yearly variation")
    save_figure(fig, "synthetic_data.png")

    np.save(DATA_DIR / "s_since_2000.npy", s_since_2000)
    np.save(DATA_DIR / "mlat.npy", mlat)

    np.save(DATA_DIR / "Be_truth.npy", be)
    np.save(DATA_DIR / "Bn_truth.npy", bn)
    np.save(DATA_DIR / "Bu_truth.npy", bu)

    np.save(DATA_DIR / "Be.npy", be_comp)
    np.save(DATA_DIR / "Bn.npy", bn_comp)
    np.save(DATA_DIR / "Bu.npy", bu_comp)

if __name__ == "__main__":
    main()
