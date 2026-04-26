from pathlib import Path

import numpy as np
import pandas as pd
from apexpy import Apex
from baseline import BaselineEstimator, VarianceEstimator
from datetime import datetime
import netCDF4 as nc
from tqdm import tqdm

from supermag_example_plotting import (
    build_step_1a_figure,
    build_step_1b_figure,
    build_step_1c_figure,
    build_step_1d_figure,
    build_step_1e_figure,
    build_step_2a_figure,
    build_step_2b_figure,
    build_step_2c_figure,
    save_chunked_component_comparison,
    save_chunked_component_triplet,
    save_chunked_qd_comp_triplet,
    save_step_chunks,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
EXAMPLE_FIGURE_DIR = FIGURE_DIR / "SM_example"
EXAMPLE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SM_PATH_no_BS = DATA_DIR / "DMH_SM_1min_2024_no_BS.netcdf"
SM_PATH_no_QD = DATA_DIR / "DMH_SM_1min_2024_no_QD.netcdf"
SM_PATH = DATA_DIR / "DMH_SM_1min_2024.netcdf"
STEP_1C_PLOT_DIR = "figures/SM_example/QD_diag"
GLAT = 76.77
GLON = 341.37
MINUTES_PER_DAY = 24 * 60
STEP_1_CONTEXT_CHUNK_DAYS = 7
STEP_1_DETAIL_CHUNK_DAYS = 2
STEP_2_CHUNK_DAYS = 60
STEP_2B_SIGMA_DAYS = 15.0
STEP_1D_SIGMA_DAYS = 1/12
STEP_1C_MIN_WINDOW_DAYS = 5
STEP_1C_CHECKPOINT_DIR = DATA_DIR / "cache" / "step_1c"
REUSE_STEP_1C_CHECKPOINT = False
WRITE_STEP_1C_CHECKPOINT = True
STEP_1C_PLOT_DIAGNOSTICS = True
COMPONENT_TITLES = ("Be", "Bn", "Bu")
COMPONENT_CONFIGS = (
    ("E", "uE"),
    ("N", "uN"),
    ("Z", "uZ"),
)
def step_1c_checkpoint_path(component, min_window_days):
    """Return a stable checkpoint filename for one component configuration."""
    filename = (
        f"DMH_2024_{component}_"
        "paper_mode_"
        f"w{int(min_window_days)}_step1c.pkl"
    )
    return STEP_1C_CHECKPOINT_DIR / filename


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
    detail_days = 2 if n_points >= 2 * MINUTES_PER_DAY else max(1, n_points // MINUTES_PER_DAY)
    detail_stop = d_start + pd.Timedelta(days=detail_days)
    raw_components = (be, bn, bu)
    db_components = (dbe, dbn, dbu)
    qd_components = (be_QD, bn_QD, bu_QD)
    qy_components = (be_QY, bn_QY, bu_QY)

    for subdir_name, components, chunk_days in (
        ("SM", raw_components, STEP_1_CONTEXT_CHUNK_DAYS),
        ("SM_db", db_components, STEP_1_CONTEXT_CHUNK_DAYS),
        ("SM_QD", qd_components, STEP_1_DETAIL_CHUNK_DAYS),
        ("SM_QY", qy_components, STEP_2_CHUNK_DAYS),
    ):
        save_chunked_component_triplet(
            EXAMPLE_FIGURE_DIR,
            t,
            components,
            COMPONENT_TITLES,
            subdir_name,
            subdir_name,
            chunk_days,
        )
    
    mlat = compute_mlat(t, GLAT, GLON)

    ve = VarianceEstimator(t, bn, be, bu, mlat)
    ve.estimate()
    be_e = None
    be_n = None
    be_u = None

    for component, variance_col in COMPONENT_CONFIGS:
        if component == "E":
            values = be
        elif component == "N":
            values = bn
        else:
            values = bu

        estimator = BaselineEstimator(
            t,
            values,
            ve.df[variance_col].values,
            mlat,
            component=component,
            step_1c_min_window_days=STEP_1C_MIN_WINDOW_DAYS,
            step_1c_plot_diagnostics=STEP_1C_PLOT_DIAGNOSTICS,
            step_1c_diagnostic_time_range=(d_start, detail_stop),
            step_1c_plot_dir=STEP_1C_PLOT_DIR,
        )
        estimator.get_baseline(
            step_1d_sigma_days=STEP_1D_SIGMA_DAYS,
            step_2b_sigma_days=STEP_2B_SIGMA_DAYS,
            step_1c_checkpoint_path=step_1c_checkpoint_path(
                component=component,
                min_window_days=estimator.step_1c_min_window_days,
            ),
            reuse_step_1c_checkpoint=REUSE_STEP_1C_CHECKPOINT,
            write_step_1c_checkpoint=WRITE_STEP_1C_CHECKPOINT,
        )

        if component == "E":
            be_e = estimator
        elif component == "N":
            be_n = estimator
        else:
            be_u = estimator

    for subdir_name, chunk_days, finite_values, figure_builder in (
        ("SM_step_1a", STEP_1_CONTEXT_CHUNK_DAYS, be_n.df["x"].values, build_step_1a_figure),
        ("SM_step_1b", STEP_1_CONTEXT_CHUNK_DAYS, be_n.df["x"].values, build_step_1b_figure),
        ("SM_step_1c", STEP_1_DETAIL_CHUNK_DAYS, be_n.df["x"].values, build_step_1c_figure),
        ("SM_step_1d", STEP_1_DETAIL_CHUNK_DAYS, be_n.df["x"].values, build_step_1d_figure),
        ("SM_step_1e", STEP_1_CONTEXT_CHUNK_DAYS, be_n.df["x"].values, build_step_1e_figure),
        ("SM_step_2a", STEP_2_CHUNK_DAYS, be_n.df["x_QD"].values, build_step_2a_figure),
        ("SM_step_2b", STEP_2_CHUNK_DAYS, be_n.df["x_QD"].values, build_step_2b_figure),
        ("SM_step_2c", STEP_2_CHUNK_DAYS, be_n.df["x_QD"].values, build_step_2c_figure),
    ):
        save_step_chunks(
            EXAMPLE_FIGURE_DIR,
            be_n,
            t,
            chunk_days,
            finite_values,
            subdir_name,
            figure_builder,
        )

    db_component_pairs = (
        (dbe, be_e.df["x_QD_QY"]),
        (dbn, be_n.df["x_QD_QY"]),
        (dbu, be_u.df["x_QD_QY"]),
    )
    save_chunked_component_comparison(
        EXAMPLE_FIGURE_DIR,
        t,
        db_component_pairs,
        COMPONENT_TITLES,
        "SM_db_comp",
        "SM_db_comp",
        STEP_1_DETAIL_CHUNK_DAYS,
    )

    save_chunked_qd_comp_triplet(
        EXAMPLE_FIGURE_DIR,
        t,
        (
            (be, be_QD, be_e, "Be"),
            (bn, bn_QD, be_n, "Bn"),
            (bu, bu_QD, be_u, "Bu"),
        ),
        STEP_1_DETAIL_CHUNK_DAYS,
        error_plot,
    )

    qy_component_pairs = (
        (be_QY, be_e.df["QY"]),
        (bn_QY, be_n.df["QY"]),
        (bu_QY, be_u.df["QY"]),
    )
    save_chunked_component_comparison(
        EXAMPLE_FIGURE_DIR,
        t,
        qy_component_pairs,
        COMPONENT_TITLES,
        "SM_QY_comp",
        "SM_QY_comp",
        STEP_2_CHUNK_DAYS,
    )
