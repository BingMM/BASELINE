from importlib import import_module

from .types import (
    BaselineInputs,
    BaselineResult,
    ModernBaselineConfig,
    Step1CResult,
    VarianceInputs,
    VarianceResult,
)

__all__ = [
    "BaselineInputs",
    "BaselineResult",
    "ModernBaselineConfig",
    "ModernBaselineEngine",
    "ModernVarianceEngine",
    "Step1CResult",
    "Step1CDayBinCache",
    "VarianceInputs",
    "VarianceResult",
    "baseline_result_to_frame",
    "build_step1c_target_index",
    "build_baseline_inputs",
    "build_variance_inputs",
    "collect_step1c_window_chunks",
    "compute_step1c_bin30",
    "expand_step1c_window",
    "get_max_odd_window_size",
    "infer_cadence_seconds",
    "prepare_step1c_day_bin_cache",
    "validate_odd_window_days",
    "variance_result_to_frame",
]


def __getattr__(name: str):
    if name in {
        "baseline_result_to_frame",
        "build_baseline_inputs",
        "build_variance_inputs",
        "infer_cadence_seconds",
        "variance_result_to_frame",
    }:
        module = import_module(".adapters", __name__)
        return getattr(module, name)
    if name == "ModernBaselineEngine":
        module = import_module(".pipeline", __name__)
        return getattr(module, name)
    if name == "ModernVarianceEngine":
        module = import_module(".variance", __name__)
        return getattr(module, name)
    if name in {
        "Step1CDayBinCache",
        "build_step1c_target_index",
        "collect_step1c_window_chunks",
        "compute_step1c_bin30",
        "expand_step1c_window",
        "get_max_odd_window_size",
        "prepare_step1c_day_bin_cache",
        "validate_odd_window_days",
    }:
        module = import_module(".step1c_prepare", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
