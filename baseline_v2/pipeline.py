from __future__ import annotations

from .types import BaselineInputs, BaselineResult, ModernBaselineConfig, VarianceResult


class ModernBaselineEngine:
    """Orchestrate one-component baseline estimation for `baseline_v2`."""

    def __init__(self, config: ModernBaselineConfig | None = None):
        self.config = ModernBaselineConfig() if config is None else config

    def fit_component(
        self,
        inputs: BaselineInputs,
        variance: VarianceResult,
    ) -> BaselineResult:
        if self.config.step_1c_method == "reference":
            from .step1c_reference import run_reference_component

            return run_reference_component(inputs, variance, self.config)
        if self.config.step_1c_method == "robust":
            from .step1c_robust import run_robust_component

            return run_robust_component(inputs, variance, self.config)
        raise ValueError(f"Unsupported Step 1c method: {self.config.step_1c_method}")
