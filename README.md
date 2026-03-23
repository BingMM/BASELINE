# BASELINE

BASELINE is an open source Python implementation of the baseline-removal workflow described in the SuperMAG data processing paper:

- J. W. Gjerloev, *The SuperMAG data processing technique*, JGR, 2012

The goal of this repository is to provide a transparent and reusable implementation of the core baseline estimation steps used for ground magnetometer data. The paper describes the overall approach, but several lower-level implementation details are not fully specified, so some parts of this repository necessarily use pragmatic approximations.

## Current Scope

Implemented:

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
from baseline import BaselineEstimator, VarianceEstimator

variance = VarianceEstimator(t, Bn, Be, Bz, mlat)
variance.estimate()

baseline_n = BaselineEstimator(t, Bn, variance.df["uN"].values, mlat, component="N")
baseline_n.get_baseline()

qd = baseline_n.df["QD"]
qy = baseline_n.df["QY"]
residual = baseline_n.df["x_QD_QY"]
```

`BaselineEstimator` must be run separately for each component (`"N"`, `"E"`, or `"Z"`).

## Repository Layout

- [`baseline/`](/home/bing/Dropbox/work/code/repos/BASELINE/baseline) contains the installable package
- [`scripts/`](/home/bing/Dropbox/work/code/repos/BASELINE/scripts) contains local example and data-generation scripts
- [`documentation/`](/home/bing/Dropbox/work/code/repos/BASELINE/documentation) contains the reference paper
- [`data/`](/home/bing/Dropbox/work/code/repos/BASELINE/data) contains synthetic example data

## Notes on the Implementation

- The paper does not fully specify all weighting and typical-value details.
- The current implementation uses an IRLS-based robust typical-value estimator rather than reproducing the paper's histogram-mode logic exactly.
- The weighting used in the smoothing stages is still a modeling choice and should be treated as provisional.

## Development Status

Known follow-up work is tracked in [`TODO.md`](/home/bing/Dropbox/work/code/repos/BASELINE/TODO.md).
