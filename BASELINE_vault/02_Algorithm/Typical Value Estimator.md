# Typical Value Estimator

Source: SuperMAG paper section 4.

The article defines a "typical" value because ground magnetometer distributions are often skewed. Average and median are not considered sufficient.

Article description:

- derive a probability distribution;
- determine the mode;
- fit a Gaussian by least squares;
- define the typical value as the mode unless the spike-handling requirement is fulfilled;
- if multiple modes exist, define the mode as the simple average of the individual modal values.

## Figure 4 Interpretation

The right panel of Figure 4 shows:

- a broad, mostly single-peaked distribution;
- a long negative tail from disturbed values;
- average pulled far negative;
- median, Gaussian center, and mode clustered near the main peak;
- a Gaussian fit over the broad central distribution, not a narrow fit to one histogram bar.

Important conclusion from 2026-04-20 discussion:

The Gaussian should likely be fit to the whole distribution, or at least the main full distribution, not a tiny local window around the modal bin. If the distribution is multimodal or dominated by disturbance values, the fit should become broad or fail so Step 1c widens the time window.

## Current Implementation Direction

Use the built-in paper-style typical-value path.

Current `paper_mode` behavior:

- fixed 1 nT histogram bins, based on visual inspection of Figure 4;
- mode is the returned typical value;
- Gaussian is fit to the full histogram distribution;
- equation (4) is applied through a nearest-neighbor average check on the modal region;
- Gaussian sigma is used for Step 1c acceptance;
- if distribution is broad or multimodal, full-fit sigma should be large, causing Step 1c to widen/reject;
- constant values and too few samples are invalid rather than accepted as zero-sigma quiet values.

## Earlier Problem

The first `paper_mode` attempt used a local Gaussian fit around the modal peak. Histogram inspection showed that this was too permissive: sparse multimodal distributions could pass because one narrow local peak produced a small sigma. That was changed to full-histogram fitting.

## Equation 4 Status

The article wording around equation (4) is internally inconsistent, but the current implementation follows the only interpretation that matches the spike-handling explanation:

- compute the local average of the modal neighborhood;
- keep the mode when the fitted Gaussian peak height exceeds that local average;
- otherwise use the Gaussian center instead.

This is closer to the paper than the earlier isolated-spike heuristic, but it still needs validation against the SuperMAG comparison plots.

## Current Code State

The old `irls` and `mode` branches have been removed. The estimator now exposes only the paper-style implementation.
