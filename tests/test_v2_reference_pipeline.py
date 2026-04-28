from __future__ import annotations

import tempfile
import unittest

import numpy as np

from baseline_v2.pipeline import ModernBaselineEngine
from baseline_v2.types import BaselineInputs, ModernBaselineConfig, VarianceInputs
from baseline_v2.variance import ModernVarianceEngine


class ReferencePipelineTests(unittest.TestCase):
    def test_reference_pipeline_runs_end_to_end(self):
        t, n, e, z = _build_synthetic_nez(days=19)
        variance = ModernVarianceEngine().fit(
            VarianceInputs(
                t=t,
                n=n,
                e=e,
                z=z,
                mlat=52.0,
                cadence_seconds=30 * 60,
            )
        )
        result = ModernBaselineEngine(
            ModernBaselineConfig(step_1c_method="reference")
        ).fit_component(
            BaselineInputs(
                t=t,
                x=n,
                component="N",
                mlat=52.0,
                cadence_seconds=30 * 60,
            ),
            variance,
        )

        self.assertEqual(result.qd.shape, n.shape)
        self.assertEqual(result.qy.shape, n.shape)
        self.assertEqual(result.residual.shape, n.shape)
        self.assertEqual(result.step1c.t.shape, (19 * 48,))
        self.assertIn("step_1c_diagnostics", result.diagnostics)
        self.assertTrue(np.any(result.step1c.status == "ok"))

    def test_reference_pipeline_reuses_step1c_checkpoint(self):
        t, n, e, z = _build_synthetic_nez(days=19)
        variance = ModernVarianceEngine().fit(
            VarianceInputs(
                t=t,
                n=n,
                e=e,
                z=z,
                mlat=55.0,
                cadence_seconds=30 * 60,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = tmpdir
            config_write = ModernBaselineConfig(
                step_1c_method="reference",
                step_1c_checkpoint_path=checkpoint_dir,
                write_step_1c_checkpoint=True,
            )
            config_read = ModernBaselineConfig(
                step_1c_method="reference",
                step_1c_checkpoint_path=checkpoint_dir,
                reuse_step_1c_checkpoint=True,
            )
            inputs = BaselineInputs(
                t=t,
                x=n,
                component="N",
                mlat=55.0,
                cadence_seconds=30 * 60,
            )
            first = ModernBaselineEngine(config_write).fit_component(inputs, variance)
            second = ModernBaselineEngine(config_read).fit_component(inputs, variance)

            np.testing.assert_allclose(first.qd, second.qd, equal_nan=True)
            np.testing.assert_allclose(first.qy, second.qy, equal_nan=True)
            np.testing.assert_allclose(first.step1c.value, second.step1c.value, equal_nan=True)
            np.testing.assert_array_equal(first.step1c.status, second.step1c.status)

    def test_robust_pipeline_runs_end_to_end(self):
        t, n, e, z = _build_synthetic_nez(days=19)
        variance = ModernVarianceEngine().fit(
            VarianceInputs(
                t=t,
                n=n,
                e=e,
                z=z,
                mlat=52.0,
                cadence_seconds=30 * 60,
            )
        )
        result = ModernBaselineEngine(
            ModernBaselineConfig(step_1c_method="robust")
        ).fit_component(
            BaselineInputs(
                t=t,
                x=n,
                component="N",
                mlat=52.0,
                cadence_seconds=30 * 60,
            ),
            variance,
        )

        self.assertEqual(result.qd.shape, n.shape)
        self.assertEqual(result.step1c.t.shape, (19 * 48,))
        self.assertIn("step_1c_diagnostics", result.diagnostics)


def _build_synthetic_nez(days: int):
    samples_per_day = 48
    n_samples = days * samples_per_day
    t = np.arange(
        np.datetime64("2024-03-01T00:00"),
        np.datetime64("2024-03-01T00:00") + n_samples * np.timedelta64(30, "m"),
        np.timedelta64(30, "m"),
    )
    phase = np.arange(n_samples, dtype=float)
    daily = 15.0 * np.sin(2.0 * np.pi * phase / samples_per_day)
    slow = 2.0 * np.sin(2.0 * np.pi * phase / (samples_per_day * 4))
    trend = np.linspace(-1.0, 1.0, n_samples)
    n = daily + slow + trend
    e = 0.6 * daily - 0.3 * slow
    z = -0.4 * daily + 0.2 * trend
    return t, n, e, z


if __name__ == "__main__":
    unittest.main()
