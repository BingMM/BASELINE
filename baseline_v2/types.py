from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _as_1d_array(values, *, name: str, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _require_same_length(reference: np.ndarray, other: np.ndarray, *, name: str) -> None:
    if other.shape != reference.shape:
        raise ValueError(
            f"{name} must have shape {reference.shape}, got {other.shape}"
        )


@dataclass
class BaselineInputs:
    t: np.ndarray
    x: np.ndarray
    component: str
    mlat: float
    cadence_seconds: int
    finite_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.t = _as_1d_array(self.t, name="t")
        self.x = _as_1d_array(self.x, name="x", dtype=float)
        _require_same_length(self.t, self.x, name="x")
        if self.component not in {"N", "E", "Z"}:
            raise ValueError("component must be one of 'N', 'E', or 'Z'")
        self.cadence_seconds = int(self.cadence_seconds)
        if self.cadence_seconds <= 0:
            raise ValueError("cadence_seconds must be positive")
        if self.finite_mask is None:
            self.finite_mask = np.isfinite(self.x)
        else:
            self.finite_mask = _as_1d_array(
                self.finite_mask, name="finite_mask", dtype=bool
            )
            _require_same_length(self.t, self.finite_mask, name="finite_mask")


@dataclass
class VarianceInputs:
    t: np.ndarray
    n: np.ndarray
    e: np.ndarray
    z: np.ndarray
    mlat: float
    cadence_seconds: int

    def __post_init__(self) -> None:
        self.t = _as_1d_array(self.t, name="t")
        self.n = _as_1d_array(self.n, name="n", dtype=float)
        self.e = _as_1d_array(self.e, name="e", dtype=float)
        self.z = _as_1d_array(self.z, name="z", dtype=float)
        _require_same_length(self.t, self.n, name="n")
        _require_same_length(self.t, self.e, name="e")
        _require_same_length(self.t, self.z, name="z")
        self.cadence_seconds = int(self.cadence_seconds)
        if self.cadence_seconds <= 0:
            raise ValueError("cadence_seconds must be positive")


@dataclass
class Step1CResult:
    t: np.ndarray
    value: np.ndarray
    weight: np.ndarray
    sigma: np.ndarray
    confidence: np.ndarray
    status: np.ndarray
    window_days: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = _as_1d_array(self.t, name="t")
        self.value = _as_1d_array(self.value, name="value", dtype=float)
        self.weight = _as_1d_array(self.weight, name="weight", dtype=float)
        self.sigma = _as_1d_array(self.sigma, name="sigma", dtype=float)
        self.confidence = _as_1d_array(self.confidence, name="confidence", dtype=float)
        self.status = _as_1d_array(self.status, name="status")
        self.window_days = _as_1d_array(self.window_days, name="window_days", dtype=float)
        for name, array in (
            ("value", self.value),
            ("weight", self.weight),
            ("sigma", self.sigma),
            ("confidence", self.confidence),
            ("status", self.status),
            ("window_days", self.window_days),
        ):
            _require_same_length(self.t, array, name=name)


@dataclass
class VarianceResult:
    t: np.ndarray
    v: np.ndarray
    u_n: np.ndarray
    u_e: np.ndarray
    u_z: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = _as_1d_array(self.t, name="t")
        self.v = _as_1d_array(self.v, name="v", dtype=float)
        self.u_n = _as_1d_array(self.u_n, name="u_n", dtype=float)
        self.u_e = _as_1d_array(self.u_e, name="u_e", dtype=float)
        self.u_z = _as_1d_array(self.u_z, name="u_z", dtype=float)
        for name, array in (
            ("v", self.v),
            ("u_n", self.u_n),
            ("u_e", self.u_e),
            ("u_z", self.u_z),
        ):
            _require_same_length(self.t, array, name=name)

    def u_for_component(self, component: str) -> np.ndarray:
        if component == "N":
            return self.u_n.copy()
        if component == "E":
            return self.u_e.copy()
        if component == "Z":
            return self.u_z.copy()
        raise ValueError("component must be one of 'N', 'E', or 'Z'")


@dataclass
class BaselineResult:
    t: np.ndarray
    component: str
    x: np.ndarray
    u: np.ndarray
    qd: np.ndarray
    qy: np.ndarray
    residual: np.ndarray
    step1c: Step1CResult
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = _as_1d_array(self.t, name="t")
        self.x = _as_1d_array(self.x, name="x", dtype=float)
        self.u = _as_1d_array(self.u, name="u", dtype=float)
        self.qd = _as_1d_array(self.qd, name="qd", dtype=float)
        self.qy = _as_1d_array(self.qy, name="qy", dtype=float)
        self.residual = _as_1d_array(self.residual, name="residual", dtype=float)
        for name, array in (
            ("x", self.x),
            ("u", self.u),
            ("qd", self.qd),
            ("qy", self.qy),
            ("residual", self.residual),
        ):
            _require_same_length(self.t, array, name=name)
        if self.component not in {"N", "E", "Z"}:
            raise ValueError("component must be one of 'N', 'E', or 'Z'")


@dataclass
class ModernBaselineConfig:
    step_1c_method: str = "reference"
    step_1d_a: float = -0.5
    step_1d_sigma_days: float = 1 / 24
    step_2b_a: float = -0.5
    step_2b_sigma_days: float = 15.0
    step_1c_min_window_days: int = 5
    step_1c_checkpoint_path: str | Path | None = None
    reuse_step_1c_checkpoint: bool = False
    write_step_1c_checkpoint: bool = False
    step_1c_plot_diagnostics: bool = False
    step_1c_diagnostic_time_range: tuple[Any, Any] | None = None
    step_1c_plot_dir: str | Path = "figures/QD_diag"
    verbose: bool = False
    progress_label: str | None = None
    progress_every_days: int = 14

    def __post_init__(self) -> None:
        self.step_1c_method = str(self.step_1c_method)
        if self.step_1c_method not in {"reference", "robust"}:
            raise ValueError("step_1c_method must be 'reference' or 'robust'")
        self.step_1c_min_window_days = int(self.step_1c_min_window_days)
        if self.step_1c_min_window_days <= 0 or self.step_1c_min_window_days % 2 == 0:
            raise ValueError("step_1c_min_window_days must be a positive odd integer")
        if self.step_1c_checkpoint_path is not None:
            self.step_1c_checkpoint_path = Path(self.step_1c_checkpoint_path)
        self.step_1c_plot_dir = Path(self.step_1c_plot_dir)
        self.verbose = bool(self.verbose)
        if self.progress_label is not None:
            self.progress_label = str(self.progress_label)
        self.progress_every_days = int(self.progress_every_days)
        if self.progress_every_days <= 0:
            raise ValueError("progress_every_days must be positive")
