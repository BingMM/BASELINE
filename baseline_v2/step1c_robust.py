from __future__ import annotations

from .reference_math import get_typical_value_dominant_region
from .step1c_reference import run_component_with_local_estimator
from .types import BaselineInputs, BaselineResult, ModernBaselineConfig, VarianceResult


def run_robust_component(
    inputs: BaselineInputs,
    variance: VarianceResult,
    config: ModernBaselineConfig,
) -> BaselineResult:
    """
    Run the V2 pipeline with the experimental dominant-region Step 1c estimator.

    The surrounding pipeline is intentionally still the same as the current
    reference path. The first experimental change in V2 is localized to the
    semi-hourly Step 1c estimator.
    """
    return run_component_with_local_estimator(
        inputs=inputs,
        variance=variance,
        config=config,
        local_estimator=get_typical_value_dominant_region,
    )
