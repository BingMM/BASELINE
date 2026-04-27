# BASELINE Algorithm Documentation

## Purpose

This repository implements a practical reproduction of the SuperMAG baseline
workflow described in `SuperMAG_data_processing_paper.pdf`.

The implemented pipeline has three main parts:

1. optional coordinate rotation from `X/Y/Z` to local magnetic `N/E/Z`
2. modified variance estimation for yearly-baseline weighting
3. baseline estimation for one component at a time

The code is organized around three public classes:

- `CoordinateRotator`
- `VarianceEstimator`
- `BaselineEstimator`

An additional helper, `InverseCoordinateRotator`, can rotate estimated
baseline products back from `N/E/Z` into `X/Y/Z`.

This document describes:

- what the original paper does
- what this repository implements
- where the implementation deliberately or accidentally differs from the paper

## High-Level Workflow

The intended end-to-end workflow is:

1. Start with magnetic field time series on a regular time grid.
2. If needed, rotate `X/Y/Z -> N/E/Z`.
3. Estimate the modified variance `u` from `N/E/Z`.
4. For each component separately, estimate:
   - daily baseline `QD`
   - yearly baseline `QY`
5. Optionally rotate `QD`, `QY`, and corrected signals back to `X/Y/Z`.

## Inputs and Assumptions

### Time Series

The core code assumes:

- a regular cadence
- synchronized component time series
- one station at a time

### Coordinate System

`BaselineEstimator` itself is component-wise. It does not enforce a coordinate
system. However, the SuperMAG method is defined for local magnetic `N/E/Z`, and
several parameterizations in this repository assume `N`, `E`, and `Z`:

- `FWHM_stat` curves are component-specific
- `VarianceEstimator` uses component-specific latitude scaling

So:

- for SuperMAG-style reproduction, use `N/E/Z`
- for practical use on other component systems, the code can still run, but it
  is no longer a strict reproduction of the paper

## 1. Coordinate Rotation

Implemented in:

- `baseline/coordinate_rotator.py`

### Forward Rotation: `CoordinateRotator`

Purpose:

- rotate geographic-like `X/Y/Z` into local magnetic `N/E/Z`

Implemented steps:

1. Compute an instantaneous horizontal angle:
   - `q_raw = unwrap(arctan2(Y, X))`
2. Estimate one typical declination value per day using a symmetric odd-day
   window.
3. Smooth the daily declination values.
4. Interpolate the smoothed declination back to the full time grid.
5. Rotate:
   - `N = X cos(q) + Y sin(q)`
   - `E = -X sin(q) + Y cos(q)`
   - `Z = Z`

### Inverse Rotation: `InverseCoordinateRotator`

Purpose:

- rotate finished baseline products from `N/E/Z` back into `X/Y/Z`

It accepts:

- one completed `CoordinateRotator`
- three completed `BaselineEstimator` objects for `E`, `N`, and `Z`

It returns a dataframe containing:

- original `X/Y/Z`
- `QD_X/Y/Z`
- `QY_X/Y/Z`
- combined baseline `baseline_X/Y/Z`
- corrected signal `X_corr/Y_corr/Z_corr`

Inverse rotation:

- `X = N cos(q) - E sin(q)`
- `Y = N sin(q) + E cos(q)`
- `Z = Z`

### Deviation from the Paper

The paper sketches the declination procedure but does not fully specify it.
This implementation follows the same general idea, but the exact declination
estimator and smoothing choices are implementation choices in this repository.

## 2. Modified Variance Estimation

Implemented in:

- `baseline/variance_estimator.py`

Class:

- `VarianceEstimator`

Purpose:

- compute the modified variance `u` used for yearly-baseline weighting

### Implemented Steps

1. Compute `v` from rolling sums of squared deviations over a 24-hour window.
2. Compute latitude-dependent scaling factors:
   - `fN = |cos(mlat)|`
   - `fE = 0`
   - `fZ = |sin(mlat)|`
3. Compute delayed-memory terms `dN`, `dE`, `dZ` using a causal 8-day kernel.
4. Compute:
   - `uN = v + dN`
   - `uE = v + dE`
   - `uZ = v + dZ`

### Notes

- This part is intended to follow equations 11-13 in the paper.
- The implementation uses FFT-based convolution for the delayed-memory term.

## 3. Baseline Estimation

Implemented in:

- `baseline/baseline_estimator.py`

Class:

- `BaselineEstimator`

Purpose:

- estimate baseline terms for one component at a time

The implemented output terms are:

- `QD`: daily baseline
- `QY`: yearly baseline

The placeholder Step 3 `QO` term is not implemented.

## 3.1 FWHM Threshold

Method:

- `get_FWHM_stat()`

Purpose:

- compute the empirical latitude-dependent threshold used in the acceptance
  tests

This is implemented separately for `N`, `E`, and `Z`.

## 3.2 Daily Baseline `QD`

The daily baseline follows the paper's Step 1 workflow.

### Step 1a

Method:

- `step_1a()`

Purpose:

- estimate one typical value per day

Implementation:

- group by day
- run the typical-value estimator on the day's samples
- place the result at noon

### Step 1b

Method:

- `step_1b()`

Purpose:

- interpolate the daily values from Step 1a back to the native cadence

Implementation:

- cubic convolution interpolation

Residual after Step 1b:

- `residual_step_1 = x - step_1b`

### Step 1c

Method:

- `step_1c()`

Purpose:

- estimate one semi-hourly typical value for each day and 30-minute bin

#### Implemented Logic

For each target day and half-hour bin:

1. Start from a minimum odd window size:
   - current code default in `BaselineEstimator`: `5` days
2. Collect all residual samples in that half-hour bin across the current
   symmetric day window.
3. Estimate a typical value and fitted spread from the histogram-based
   estimator.
4. Accept when:
   - `sigma <= FWHM_stat`
5. Otherwise widen the window by 2 days and retry.

If the target day has no finite input samples in that half-hour bin:

- the node is marked `missing_input`
- no estimate is attempted from neighboring days

#### Current Performance Optimization

Step 1c was optimized to reduce repeated pandas work.

The current implementation:

- precomputes per-day/per-bin residual arrays
- precomputes target-day finite-sample counts
- precomputes summed/count `FWHM_stat` values on the same grid
- widens windows incrementally instead of rebuilding dataframe masks each time

The helper functions are:

- `_prepare_step_1c_day_bin_cache`
- `_collect_step_1c_window_chunks`
- `_expand_step_1c_window`

This is intended as a performance optimization only. It should not change the
algorithmic meaning of Step 1c.

#### Step 1c Weighting

Each accepted Step 1c node gets a weight for Step 1d smoothing.

The weight uses:

- `sigma_weight = get_weight_sigma(...)`
- the accepted window size

Current formula:

- `weight ~ 1 / (sigma_weight^2 * window_days / step_1c_min_window_days)`

#### Deviation from the Paper

1. The paper starts from a 3-day Step 1c window.
   - Current code default: `5` days.
2. The exact Step 1d uncertainty formula is not specified in the paper.
   - Current weighting formula is a repository implementation choice.
3. Missing-input bins are skipped explicitly.
   - This behavior is useful and intentional, but the paper does not spell it
     out in this form.

### Step 1d

Method:

- `step_1d()`

Purpose:

- smooth the accepted semi-hourly values and interpolate them back to the
  native cadence

Implementation:

1. weighted Gaussian smoothing on the semi-hourly node grid
2. cubic convolution interpolation back to the full cadence

Current behavior:

- nodes with Step 1c status `missing_input` are forced back to `NaN` after
  smoothing so that true data gaps remain gaps
- isolated rejected Step 1c nodes are allowed to smooth through from their
  neighbors

#### Deviation from the Paper

The paper indicates that the weighted fit should use:

- information about the spread in the data
- information about the width of the window used

The exact weighting formula is not given for Step 1d. The current weighting
used here is therefore an implementation choice.

Also:

- an adaptive-sigma Step 1d variant existed earlier in this repository
- it has been removed
- the current code uses one fixed smoothing width only

### Step 1e

Method:

- `step_1e()`

Purpose:

- subtract `QD` from the original component:
  - `x_QD = x - QD`

## 3.3 Typical Value Estimator

Implemented in:

- `get_typical_value()`
- `_get_typical_value_paper_mode()`

This repository now has one typical-value estimator only:

- the paper-oriented histogram/mode method

Older alternative estimator branches were removed.

### Current Implemented Method

1. Build a fixed-width histogram with:
   - `1 nT` bin width
2. Define the mode from the bins with maximum count.
3. Fit a Gaussian to the full histogram distribution.
4. Use the fitted Gaussian sigma as the spread estimate used in Step 1c
   acceptance.
5. Apply the repository's current interpretation of equation 4:
   - compare the fitted Gaussian peak height with the local three-bin average
     around the modal region
   - if the condition is not met, replace the typical value with the Gaussian
     center

### Important Interpretation Notes

#### Equation 4

The paper text around equation 4 is internally difficult to interpret. The
current implementation uses this rule:

- keep the histogram mode when the fitted Gaussian peak height exceeds the
  local three-bin average around the mode
- otherwise use the Gaussian center to avoid isolated-bin spikes

This is a best-effort interpretation, not a guaranteed exact reconstruction of
the SuperMAG production code.

#### Equation 5

The code currently defines a multi-bin mode as:

- the mean of all bins tied for the maximum count

This is a literal implementation, but it may be narrower than what the paper
intended by saying the mode is "not necessarily a single number."

This remains an open interpretation question.

### Known Deviation from the Paper

The paper is not specific enough to uniquely determine:

- the exact modal-region definition
- the exact equation 4 spike rule
- the exact histogram ambiguity handling

So the current typical-value estimator is best described as:

- article-aligned in structure
- not guaranteed identical to the original SuperMAG implementation

## 3.4 Yearly Baseline `QY`

The yearly baseline follows the paper's Step 2 workflow.

### Step 2a

Method:

- `step_2a()`

Purpose:

- estimate one daily value from `x_QD`

Implementation:

1. start from a 17-day odd window
2. estimate one typical value for the window
3. accept when:
   - `sigma <= FWHM_stat`
4. widen by 2 days if needed

If the target day has no finite `x_QD` samples:

- the day is skipped
- value becomes `NaN`
- weight becomes `0`

### Step 2b

Method:

- `step_2b()`

Purpose:

- smooth the daily Step 2a values and interpolate back to the native cadence

Implementation:

1. weighted Gaussian smoothing on the daily node grid
2. cubic convolution interpolation back to the full cadence
3. mask output where `x_QD` is missing

### Step 2c

Method:

- `step_2c()`

Purpose:

- subtract `QY` from `x_QD`:
  - `x_QD_QY = x_QD - QY`

## 4. Checkpointing

Implemented in:

- `save_step_1c_checkpoint()`
- `load_step_1c_checkpoint()`

Purpose:

- avoid recomputing expensive Step 1c results when only Step 1d or Step 2
  parameters are being tuned

Stored fields:

- `QD_step_1c`
- `QD_step_1c_w`
- `QD_step_1c_status`
- `QD_step_1c_diagnostics`

Validation on load:

- component name
- expected semi-hourly target timestamp index

## 5. Diagnostics

The repository includes detailed Step 1c diagnostics.

Features:

- optional histogram plot generation
- optional time-range restriction for those plots
- per-day contribution heatmap under each histogram
- embedded fit diagnostics on the figure
- minimum histogram x-axis span of `[-100, 100]` nT

These are repository diagnostics and not part of the original paper.

## 6. Missing-Data Handling

The current code explicitly handles missing data.

Implemented behavior:

- Step 1c skips half-hour bins with no target-day input
- Step 1d preserves true raw-data gaps in `QD`
- Step 2a skips days with no finite `x_QD`
- Step 2b preserves missing `x_QD` regions in `QY`

This behavior is important for real data and for chunked plotting.

## 7. Summary of Known Deviations from the Paper

The main current deviations are:

1. **Step 1c starting window**
   - paper: 3 days
   - current estimator default: 5 days

2. **Equation 4 interpretation**
   - current implementation uses a local three-bin comparison heuristic
   - exact original SuperMAG behavior is not known from the paper alone

3. **Equation 5 interpretation**
   - current code averages only exact tied highest bins
   - the paper may have intended a broader modal region or plateau

4. **Step 1d weighting**
   - current code uses a practical weight from fitted/core spread and window
     size
   - the paper suggests a more nuanced weighting but does not fully specify it

5. **Rotation details**
   - the paper describes the purpose and broad approach
   - the exact declination-estimation implementation here is repository-specific

6. **Missing-data policy**
   - the repository has explicit missing-input and masking rules
   - these are practical implementation choices

7. **No implemented Step 3 / `QO`**
   - this repository currently implements `QD` and `QY`
   - not the full baseline decomposition described in the broader SuperMAG
     workflow

## 8. Recommended Reading Order For New Users

1. `documentation/SuperMAG_data_processing_paper.pdf`
2. `baseline/coordinate_rotator.py`
3. `baseline/variance_estimator.py`
4. `baseline/baseline_estimator.py`
5. `scripts/example_with_supermag_data.py`
6. `scripts/example_with_real_data.py`

## 9. Current Practical Status

The repository is best understood as:

- a serious, code-level reconstruction of the SuperMAG baseline workflow
- with several parts closely aligned to the paper
- but not yet guaranteed identical to the original SuperMAG production
  implementation in all details

The most important open scientific questions are still:

- the exact intended interpretation of equations 4 and 5
- whether Step 1d weighting should be made more article-faithful
- whether tracked cache artifacts should remain in the repository
