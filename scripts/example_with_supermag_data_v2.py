"""
Run the SuperMAG-facing example workflow with the native `baseline_v2` engine.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import netCDF4 as nc
import numpy as np
import pandas as pd
from apexpy import Apex

from baseline_v2 import ModernBaselineConfig, ModernBaselineEngine, ModernVarianceEngine
from baseline_v2.types import BaselineInputs, BaselineResult, VarianceInputs
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

BASE_DIR = REPO_ROOT
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
EXAMPLE_FIGURE_ROOT = FIGURE_DIR / "SM_example_v2"

SM_PATH_NO_BS = DATA_DIR / "DMH_SM_1min_2024_no_BS.netcdf"
SM_PATH_NO_QD = DATA_DIR / "DMH_SM_1min_2024_no_QD.netcdf"
SM_PATH = DATA_DIR / "DMH_SM_1min_2024.netcdf"
GLAT = 76.77
GLON = 341.37
CADENCE_SECONDS = 60
MINUTES_PER_DAY = 24 * 60
STEP_1_CONTEXT_CHUNK_DAYS = 7
STEP_1_DETAIL_CHUNK_DAYS = 2
STEP_2_CHUNK_DAYS = 60
STEP_2B_SIGMA_DAYS = 15.0
STEP_1D_SIGMA_DAYS = 1 / 24
STEP_1C_MIN_WINDOW_DAYS = 5
STEP_1C_CHECKPOINT_DIR = DATA_DIR / "cache" / "step_1c_v2"
REUSE_STEP_1C_CHECKPOINT = False
WRITE_STEP_1C_CHECKPOINT = True
COMPONENT_TITLES = ("Be", "Bn", "Bu")
COMPONENT_CONFIGS = (
    ("E", "u_e"),
    ("N", "u_n"),
    ("Z", "u_z"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the native baseline_v2 SuperMAG example."
    )
    parser.add_argument(
        "--mode",
        choices=("reference", "robust"),
        default="robust",
        help="Which Step 1c mode to run.",
    )
    return parser.parse_args()


def get_example_figure_dir(mode: str) -> Path:
    path = EXAMPLE_FIGURE_ROOT / mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def step_1c_checkpoint_path(component: str, min_window_days: int, mode: str) -> Path:
    filename = f"DMH_2024_{mode}_{component}_w{int(min_window_days)}_step1c.pkl"
    return STEP_1C_CHECKPOINT_DIR / filename


def load_real_data(sm_path: Path, sm_path_no_qd: Path, sm_path_no_bs: Path):
    """
    Load three SuperMAG NetCDF variants and derive comparison products.

    Returns:

    - the common timestamp array
    - the baseline-subtracted comparison series (`dbe/dbn/dbu`)
    - the SuperMAG `QD` and `QY` components recovered by differencing files
    - the fully processed SuperMAG `be/bn/bu` series used as local inputs
    """
    dataset_no_bs = nc.Dataset(sm_path_no_bs, "r")
    dataset_no_qd = nc.Dataset(sm_path_no_qd, "r")
    dataset = nc.Dataset(sm_path, "r")

    t = np.array(
        [
            datetime(int(yy), int(mm), int(dd), int(hh), int(mt), int(ss))
            for yy, mm, dd, hh, mt, ss in zip(
                dataset.variables["time_yr"][:].filled(np.nan),
                dataset.variables["time_mo"][:].filled(np.nan),
                dataset.variables["time_dy"][:].filled(np.nan),
                dataset.variables["time_hr"][:].filled(np.nan),
                dataset.variables["time_mt"][:].filled(np.nan),
                dataset.variables["time_sc"][:].filled(np.nan),
            )
        ],
        dtype="datetime64[s]",
    )

    be_no_bs = dataset_no_bs.variables["dbe_geo"][:].filled(np.nan).flatten()
    bn_no_bs = dataset_no_bs.variables["dbn_geo"][:].filled(np.nan).flatten()
    bu_no_bs = -dataset_no_bs.variables["dbz_geo"][:].filled(np.nan).flatten()

    be_no_qd = dataset_no_qd.variables["dbe_geo"][:].filled(np.nan).flatten()
    bn_no_qd = dataset_no_qd.variables["dbn_geo"][:].filled(np.nan).flatten()
    bu_no_qd = -dataset_no_qd.variables["dbz_geo"][:].filled(np.nan).flatten()

    be = dataset.variables["dbe_geo"][:].filled(np.nan).flatten()
    bn = dataset.variables["dbn_geo"][:].filled(np.nan).flatten()
    bu = -dataset.variables["dbz_geo"][:].filled(np.nan).flatten()

    dataset.close()
    dataset_no_qd.close()
    dataset_no_bs.close()

    be_qd = be_no_qd - be
    bn_qd = bn_no_qd - bn
    bu_qd = bu_no_qd - bu

    be_qy = be_no_bs - be_no_qd
    bn_qy = bn_no_bs - bn_no_qd
    bu_qy = bu_no_bs - bu_no_qd

    return t, be, bn, bu, be_qd, bn_qd, bu_qd, be_qy, bn_qy, bu_qy, be_no_bs, bn_no_bs, bu_no_bs


def compute_station_mlat(t: np.ndarray, glat: float, glon: float) -> float:
    """Compute one scalar magnetic latitude for the single-year record."""
    years = np.unique(pd.to_datetime(t).year)
    if years.size != 1:
        raise ValueError("example_with_supermag_data_v2.py expects a single-year record")
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
    Run the SuperMAG V2 example and write chunked comparison plots.

    The local estimator is run componentwise on the SuperMAG-provided
    `be/bn/bu` series and compared back against the SuperMAG `QD` and `QY`
    products recovered from the companion files.
    """
    args = parse_args()
    example_figure_dir = get_example_figure_dir(args.mode)
    t, dbe, dbn, dbu, be_qd, bn_qd, bu_qd, be_qy, bn_qy, bu_qy, be, bn, bu = load_real_data(
        SM_PATH,
        SM_PATH_NO_QD,
        SM_PATH_NO_BS,
    )

    d_start = pd.Timestamp("2024-03-06")
    error_plot = True
    n_points = len(t)
    detail_days = 2 if n_points >= 2 * MINUTES_PER_DAY else max(1, n_points // MINUTES_PER_DAY)
    detail_stop = d_start + pd.Timedelta(days=detail_days)
    raw_components = (be, bn, bu)
    db_components = (dbe, dbn, dbu)
    qd_components = (be_qd, bn_qd, bu_qd)
    qy_components = (be_qy, bn_qy, bu_qy)

    for subdir_name, components, chunk_days in (
        ("SM", raw_components, STEP_1_CONTEXT_CHUNK_DAYS),
        ("SM_db", db_components, STEP_1_CONTEXT_CHUNK_DAYS),
        ("SM_QD", qd_components, STEP_1_DETAIL_CHUNK_DAYS),
        ("SM_QY", qy_components, STEP_2_CHUNK_DAYS),
    ):
        save_chunked_component_triplet(
            example_figure_dir,
            t,
            components,
            COMPONENT_TITLES,
            subdir_name,
            subdir_name,
            chunk_days,
        )

    mlat = compute_station_mlat(t, GLAT, GLON)
    variance = ModernVarianceEngine().fit(
        VarianceInputs(
            t=t,
            n=bn,
            e=be,
            z=bu,
            mlat=mlat,
            cadence_seconds=CADENCE_SECONDS,
        )
    )

    be_e = None
    be_n = None
    be_u = None

    for component, _variance_col in COMPONENT_CONFIGS:
        if component == "E":
            values = be
        elif component == "N":
            values = bn
        else:
            values = bu

        result = ModernBaselineEngine(
            ModernBaselineConfig(
                step_1c_method=args.mode,
                step_1d_sigma_days=STEP_1D_SIGMA_DAYS,
                step_2b_sigma_days=STEP_2B_SIGMA_DAYS,
                step_1c_min_window_days=STEP_1C_MIN_WINDOW_DAYS,
                step_1c_checkpoint_path=step_1c_checkpoint_path(
                    component=component,
                    min_window_days=STEP_1C_MIN_WINDOW_DAYS,
                    mode=args.mode,
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
                mlat=mlat,
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
            example_figure_dir,
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
        example_figure_dir,
        t,
        db_component_pairs,
        COMPONENT_TITLES,
        "SM_db_comp",
        "SM_db_comp",
        STEP_1_DETAIL_CHUNK_DAYS,
    )

    save_chunked_qd_comp_triplet(
        example_figure_dir,
        t,
        (
            (be, be_qd, be_e, "Be"),
            (bn, bn_qd, be_n, "Bn"),
            (bu, bu_qd, be_u, "Bu"),
        ),
        STEP_1_DETAIL_CHUNK_DAYS,
        error_plot,
    )

    qy_component_pairs = (
        (be_qy, be_e.df["QY"]),
        (bn_qy, be_n.df["QY"]),
        (bu_qy, be_u.df["QY"]),
    )
    save_chunked_component_comparison(
        example_figure_dir,
        t,
        qy_component_pairs,
        COMPONENT_TITLES,
        "SM_QY_comp",
        "SM_QY_comp",
        STEP_2_CHUNK_DAYS,
    )
