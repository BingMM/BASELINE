# Project Brief

`BASELINE` is an open-source Python implementation of the baseline-removal
workflow described by J. W. Gjerloev in *The SuperMAG data processing
technique* (JGR, 2012).

Primary reference:

- `documentation/SuperMAG_data_processing_paper.pdf`

## Scientific goal

Reproduce and improve the separation of ground-magnetometer measurements into:

1. daily baseline `QD`;
2. yearly trend `QY`;
3. residual offset `QO`, which is not implemented.

The difficult scientific area is the local "typical value" estimator and its
use in Step 1c of daily-baseline determination, especially for sparse,
multimodal, skewed, or gapped data.

## Implementation tracks

- `baseline/` is the reference track. It aims to remain close to the published
  SuperMAG workflow and supports historical comparisons.
- `baseline_v2/` is the modern experimental track. It uses array-first
  internals and may deviate from the paper when that improves robustness,
  speed, or agreement with reference data.

The live branch and code determine which track is active. At the last verified
snapshot, development was paused on the `baseline_v2` branch.

## Primary entry points

- `baseline/baseline_estimator.py`
- `baseline_v2/pipeline.py`
- `baseline_v2/step1c_reference.py`
- `baseline_v2/step1c_robust.py`
- `scripts/example_with_supermag_data.py`
- `scripts/example_with_supermag_data_v2.py`
- `documentation/BASELINE_algorithm.md`
- `documentation/BASELINE_v2_design.md`

## Validation goal

The immediate research question is whether V2's robust Step 1c path is
materially faster and at least as scientifically useful as the reference path.
That requires reproducible runtime and output comparisons, not visual or commit
message claims alone.
