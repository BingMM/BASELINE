from __future__ import annotations

import numpy as np
import pandas as pd

from .types import BaselineInputs, BaselineResult, VarianceInputs, VarianceResult


def infer_cadence_seconds(t) -> int:
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(t)))
    if len(index) < 2:
        raise ValueError("at least two timestamps are required to infer cadence")
    deltas = np.diff(index.view("i8"))
    if len(deltas) == 0:
        raise ValueError("at least two timestamps are required to infer cadence")
    cadence_ns = int(deltas[0])
    if cadence_ns <= 0:
        raise ValueError("timestamps must be strictly increasing")
    if np.any(deltas != cadence_ns):
        raise ValueError("timestamps must be regularly spaced")
    return cadence_ns // 1_000_000_000


def build_baseline_inputs(
    t,
    x,
    *,
    component: str,
    mlat: float,
    cadence_seconds: int | None = None,
) -> BaselineInputs:
    if cadence_seconds is None:
        cadence_seconds = infer_cadence_seconds(t)
    return BaselineInputs(
        t=np.asarray(pd.to_datetime(t)),
        x=np.asarray(x, dtype=float),
        component=component,
        mlat=float(mlat),
        cadence_seconds=int(cadence_seconds),
    )


def build_variance_inputs(
    t,
    n,
    e,
    z,
    *,
    mlat: float,
    cadence_seconds: int | None = None,
) -> VarianceInputs:
    if cadence_seconds is None:
        cadence_seconds = infer_cadence_seconds(t)
    return VarianceInputs(
        t=np.asarray(pd.to_datetime(t)),
        n=np.asarray(n, dtype=float),
        e=np.asarray(e, dtype=float),
        z=np.asarray(z, dtype=float),
        mlat=float(mlat),
        cadence_seconds=int(cadence_seconds),
    )


def variance_result_to_frame(result: VarianceResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(result.t),
            "v": result.v,
            "uN": result.u_n,
            "uE": result.u_e,
            "uZ": result.u_z,
        }
    )


def baseline_result_to_frame(result: BaselineResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(result.t),
            "component": result.component,
            "x": result.x,
            "u": result.u,
            "QD": result.qd,
            "QY": result.qy,
            "x_QD_QY": result.residual,
        }
    )
