# BASELINE

BASELINE is an open source Python implementation of the baseline-removal workflow described in the SuperMAG data processing paper:

- J. W. Gjerloev, *The SuperMAG data processing technique*, JGR, 2012

The goal of this repository is to provide a transparent and reusable implementation of the core baseline estimation steps used for ground magnetometer data. The paper describes the overall approach, but several lower-level implementation details are not fully specified, so some parts of this repository necessarily use pragmatic approximations.

## Current Scope

Implemented:

- Coordinate rotation from `X/Y/Z` to local magnetic `N/E/Z`
- Variance estimation corresponding to equations 11, 12, and 13
- Daily baseline estimation, `QD`
- Yearly trend estimation, `QY`

Not implemented yet:

- Step 3 residual offset term, `QO`

## Installation

```bash
pip install .
```

The package currently depends on:

- `numpy`
- `pandas`
- `scipy`
- `tqdm`

## Package Usage

```python
from baseline import BaselineEstimator, CoordinateRotator, VarianceEstimator

rotator = CoordinateRotator(t, Bx, By, Bz)
rotator.rotate()
bn, be, bz = rotator.get_components()

variance = VarianceEstimator(t, bn, be, bz, mlat)
variance.estimate()

baseline_n = BaselineEstimator(t, bn, variance.df["uN"].values, mlat, component="N")
baseline_n.get_baseline()

qd = baseline_n.df["QD"]
qy = baseline_n.df["QY"]
residual = baseline_n.df["x_QD_QY"]
```

`BaselineEstimator` must be run separately for each component (`"N"`, `"E"`, or `"Z"`).

## Repository Layout

- [`baseline/`](baseline/) contains the installable package
- [`baseline_v2/`](baseline_v2/) contains the next-generation experimental package
- [`scripts/`](scripts/) contains local example and data-generation scripts
- [`documentation/`](documentation/) contains the reference paper and project
  design notes
- [`data/`](data/) contains synthetic example data

## Notes on the Implementation

- The paper does not fully specify all weighting and typical-value details.
- The current `main` branch uses a paper-style histogram-based typical-value
  estimator for Step 1c.
- The weighting used in the smoothing stages is still a modeling choice and should be treated as provisional.

Additional documentation:

- [`documentation/BASELINE_algorithm.md`](documentation/BASELINE_algorithm.md)
  documents the current reference implementation.
- [`documentation/BASELINE_v2_design.md`](documentation/BASELINE_v2_design.md)
  describes the proposed next-generation `baseline_v2` architecture.

## V2 Status

The `baseline_v2` branch now contains a self-contained native V2 pipeline with:

- typed V2 inputs/results/configuration
- a native modified-variance engine
- a native reference baseline pipeline
- an experimental robust Step 1c mode based on a dominant histogram region

The current V2 example script is:

- [`scripts/example_with_supermag_data_v2.py`](scripts/example_with_supermag_data_v2.py)

## Development Status

Known follow-up work is tracked in [`TODO.md`](TODO.md).
