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
FIGURE_DIR.mkdir(exist_ok=True)

SM_PATH_no_BS = DATA_DIR / "DMH_SM_1min_2024_no_BS.netcdf"
SM_PATH_no_QD = DATA_DIR / "DMH_SM_1min_2024_no_QD.netcdf"
SM_PATH = DATA_DIR / "DMH_SM_1min_2024.netcdf"
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
    
    n_points = len(t)
    short_slice = day_slice(0, min(7, max(1, n_points // MINUTES_PER_DAY)), n_points)
    detail_start_day = 5 if n_points >= 7 * MINUTES_PER_DAY else 0
    detail_days = 2 if n_points >= 2 * MINUTES_PER_DAY else max(1, n_points // MINUTES_PER_DAY)
    detail_slice = day_slice(detail_start_day, detail_days, n_points)
    detail_half_hour_slice = slice(
        detail_start_day * HALF_HOURS_PER_DAY,
        min(len(t), (detail_start_day + detail_days) * HALF_HOURS_PER_DAY),
    )
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
    
    mlat = compute_mlat(t, GLAT, GLON)

    ve = VarianceEstimator(t, bn, be, bu, mlat)
    ve.estimate()

    be_e = BaselineEstimator(t, be, ve.df["uE"].values, mlat, component="E", 
                             typical_value_method='mode', step_1c_plot_diagnostics=False, step_1c_plot_dir="figures/QD_diag")
    be_e.get_baseline(step_1d_sigma_days=1/72,
                      step_1d_adaptive_sigma=True,
                      step_1d_max_sigma_multiplier=6,
                      )

    be_n = BaselineEstimator(t, bn, ve.df["uN"].values, mlat, component="N", 
                             typical_value_method='mode', step_1c_plot_diagnostics=False, step_1c_plot_dir="figures/QD_diag")
    be_n.get_baseline(step_1d_sigma_days=1/72,
                      step_1d_adaptive_sigma=True,
                      step_1d_max_sigma_multiplier=2)
    
    be_u = BaselineEstimator(t, bu, ve.df["uZ"].values, mlat, component="Z", 
                             typical_value_method='mode', step_1c_plot_diagnostics=False, step_1c_plot_dir="figures/QD_diag")
    be_u.get_baseline(step_1d_sigma_days=1/72,
                      step_1d_adaptive_sigma=True,
                      step_1d_max_sigma_multiplier=6)

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][short_slice], be_n.df["x"][short_slice], label="Observed magnetic field")
    plt.plot(be_n.QD_step_1a.iloc[:min(7, len(be_n.QD_step_1a))], ".", label="Daily typical value")
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    plt.legend()
    save_figure(fig, "SM_step_1a.png")

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

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["residual_step_1"][detail_slice], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c.iloc[detail_half_hour_slice], ".", label="Semi-hourly typical values")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "SM_step_1c.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["residual_step_1"][detail_slice], label="Field minus daily fit")
    plt.plot(be_n.QD_step_1c.iloc[detail_half_hour_slice], ".", label="Semi-hourly typical values")
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["QD"][detail_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "SM_step_1d.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["x"][detail_slice], label="Observed signal")
    plt.plot(be_n.df["datetime"][detail_slice], be_n.df["x_QD"][detail_slice], label="Without daily variation")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Magnetic field [nT]")
    save_figure(fig, "SM_step_1e.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a.iloc[:long_days], ".", label="Daily typical value")
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "SM_step_2a.png")

    fig = plt.figure(figsize=(15, 9))
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Observed signal without daily variation")
    plt.plot(be_n.QD_step_2a.iloc[:long_days], ".", label="Daily typical value")
    plt.plot(be_n.df["datetime"][long_slice], be_n.df["QY"][long_slice], label="Weighted fit", color="tab:red", linewidth=2)
    plt.ylabel("Magnetic field [nT]")
    plt.xlabel("Time")
    plt.legend()
    save_figure(fig, "SM_step_2b.png")

    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(be_n.df["datetime"][long_slice], be_n.df["x_QD"][long_slice], label="Without daily variation")
    axs[0].legend()
    axs[0].set_ylabel("Magnetic field [nT]")

    axs[1].plot(be_n.df["datetime"][long_slice], be_n.df["x_QD_QY"][long_slice], label="Without daily and yearly variation")
    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Magnetic field [nT]")
    save_figure(fig, "SM_step_2c.png")
    
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

    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(t[detail_slice], be_QD[detail_slice])
    f = (be_e.QD_step_1c.index >= t[detail_slice][0]) & (be_e.QD_step_1c.index <= t[detail_slice][-1])
    axs[0].errorbar(be_e.QD_step_1c.index[f], be_e.QD_step_1c.iloc[f], yerr=1/np.sqrt(be_e.QD_step_1c_w.iloc[f]), ls='dotted')
    axs[0].plot(t[detail_slice], be_e.df['QD'][detail_slice].values)
    axs[1].plot(t[detail_slice], bn_QD[detail_slice])
    f = (be_n.QD_step_1c.index >= t[detail_slice][0]) & (be_n.QD_step_1c.index <= t[detail_slice][-1])
    axs[1].errorbar(be_n.QD_step_1c.index[f], be_n.QD_step_1c.iloc[f], yerr=1/np.sqrt(be_n.QD_step_1c_w.iloc[f]), ls='dotted')
    axs[1].plot(t[detail_slice], be_n.df['QD'][detail_slice].values)
    axs[2].plot(t[detail_slice], bu_QD[detail_slice])
    f = (be_u.QD_step_1c.index >= t[detail_slice][0]) & (be_u.QD_step_1c.index <= t[detail_slice][-1])
    axs[2].errorbar(be_u.QD_step_1c.index[f], be_u.QD_step_1c.iloc[f], yerr=1/np.sqrt(be_u.QD_step_1c_w.iloc[f]), ls='dotted')
    axs[2].plot(t[detail_slice], be_u.df['QD'][detail_slice].values)
    axs[0].set_title('Be')
    axs[1].set_title('Bn')
    axs[2].set_title('Bu')
    save_figure(fig, "SM_QD_comp.png")
    
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
