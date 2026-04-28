# BASELINE V2 Design

## Purpose

This document describes a proposed next-generation baseline engine for this
repository.

The current `main` branch is a practical reproduction of the SuperMAG
workflow, with the daily baseline `QD` implementation now behaving reasonably
well. The proposed `baseline_v2` branch should serve a different purpose:

- preserve compatibility with the current workflow where useful;
- allow scientifically motivated deviations from the paper;
- improve robustness on sparse and multimodal windows;
- simplify the numerical core;
- reduce runtime substantially.

This is not a proposal to replace the current implementation in-place. It is a
proposal to build a second engine with a cleaner architecture and a more modern
statistical formulation.

## Branch Positioning

The repository should explicitly maintain two tracks:

- `main`: reference / SuperMAG reproduction track
- `baseline_v2`: modernized research and production track

The practical rule should be:

- `main` optimizes for fidelity to the published method and historical
  comparisons with SuperMAG;
- `baseline_v2` optimizes for accuracy, robustness, and speed, even when that
  requires clear deviations from the paper-era implementation.

## Design Goals

The V2 engine should optimize for the following:

1. Clear separation between data adaptation, numerical kernels, diagnostics,
   and plotting.
2. Array-first implementation in the hot path, with minimal pandas use inside
   the estimation core.
3. Step 1c logic that is more robust to multimodality, skew, tails, and heavy
   missing-data windows.
4. Explicit uncertainty and confidence propagation instead of purely binary
   accept/reject decisions.
5. Fast repeatability through compact checkpoints and deterministic regression
   tests.

## Non-Goals

At least initially, V2 should not attempt to:

- reproduce every internal field or plotting side effect of the current
  estimator classes;
- keep the current `DataFrame`-centric internals;
- implement a full Bayesian model before a deterministic robust baseline is
  working well;
- optimize for multiple stations or distributed compute before the
  single-station core is stable.

## Proposed Package Layout

The new engine should live in a separate package subtree:

```text
baseline_v2/
  __init__.py
  adapters.py
  checkpoints.py
  diagnostics.py
  pipeline.py
  smoothing.py
  step1c_reference.py
  step1c_robust.py
  types.py
  variance.py
```

Supporting files:

```text
documentation/BASELINE_v2_design.md
scripts/example_with_supermag_data_v2.py
tests/test_v2_*.py
tests/benchmarks/
```

The current `baseline/` package should remain intact.

## Core Data Model

The numerical core should work on typed array containers rather than pandas
objects. Pandas can remain the boundary format for scripts and external users.

Suggested core types:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class BaselineInputs:
    t: np.ndarray
    x: np.ndarray
    finite_mask: np.ndarray
    component: str
    mlat: float
    cadence_seconds: int

@dataclass
class Step1CResult:
    value: np.ndarray
    sigma: np.ndarray
    confidence: np.ndarray
    status: np.ndarray
    window_days: np.ndarray

@dataclass
class BaselineResult:
    qd: np.ndarray
    qy: np.ndarray
    step1c: Step1CResult
    diagnostics: dict[str, np.ndarray]
```

Additional small config classes should hold algorithm settings rather than
having one very large estimator constructor.

## Execution Model

The V2 pipeline should have one orchestrator for one component:

1. adapt inputs into `BaselineInputs`
2. estimate or load modified variance
3. run Step 1 local quiet-time estimation
4. smooth or infer `QD`
5. estimate and smooth `QY`
6. return structured outputs and diagnostics

This should be exposed through a single object such as:

```python
engine = ModernBaselineEngine(config)
result = engine.fit_component(inputs, variance)
```

`CoordinateRotator` and the inverse rotation path can stay conceptually
separate from the core component-wise baseline engine.

## Step 1c: V2 Reference Path

The first milestone should not change the science. It should only change the
architecture.

The initial V2 Step 1c path should:

- reproduce current logic as closely as practical;
- pre-index day/bin membership once;
- avoid dataframe masking inside the inner loop;
- update histograms incrementally as the day window expands;
- reuse the current acceptance logic and status model where possible.

This provides:

- a fast compatibility path;
- a numerical regression target;
- a stable baseline for later scientific changes.

## Step 1c: V2 Robust Path

The main scientific improvement should be a new Step 1c estimator that is more
stable on sparse and ambiguous distributions.

### Motivation

The current paper-style implementation behaves reasonably well, but the known
failure modes are concentrated in Step 1c:

- competing modes across neighboring days;
- mode hopping between adjacent bins;
- broad or skewed distributions with visually plausible quiet-time centers;
- tails that inflate a full-distribution Gaussian fit;
- fragile behavior in small windows.

### Proposed Estimator

Rather than treating the tallest histogram bin as the full definition of the
quiet-time estimate, the robust path should identify a dominant quiet-time
region.

Candidate elements:

1. fixed residual histogram bins, likely still `1 nT`;
2. optional target-day-centered weighting when accumulating the window
   histogram;
3. identification of the dominant contiguous high-density region, not just a
   single modal bin;
4. robust center estimate from that region;
5. robust spread estimate from that region;
6. confidence score combining local concentration, inter-day consistency, and
   separation from competing regions.

This path should produce a continuous confidence metric, not just a hard
accept/reject gate.

### Acceptance Philosophy

The V2 robust path should not collapse all ambiguity into one threshold on one
fitted `sigma`.

Instead, it should compute:

- `value`
- `spread`
- `confidence`
- `window_days`
- `status`

Then later smoothing or inference should decide how strongly to trust each
local estimate.

## Step 1d and Step 2 in V2

The current pipeline treats local estimation and smoothing as distinct steps,
but V2 should move toward a unified uncertainty-aware formulation.

Near-term deterministic design:

- Step 1c produces local node estimates plus uncertainty/confidence;
- Step 1d smooths semi-hourly nodes using those uncertainties directly;
- Step 2 uses daily nodes and modified variance with the same uncertainty-aware
  machinery;
- true missing input and ambiguous-but-observed estimates remain distinct.

This should eliminate some of the current heuristic boundaries between:

- accepted nodes,
- rejected nodes,
- gap nodes.

## Performance Strategy

The main performance gains should come from architecture, not from exotic
optimization.

Expected high-impact changes:

- remove pandas from hot loops;
- precompute day/bin indexing once;
- use dense arrays for cached histogram accumulation;
- avoid repeated generic nonlinear `curve_fit()` calls where a robust closed
  form or local optimization is adequate;
- use compact checkpoint arrays rather than partially serialized estimator
  state;
- add optional `numba` acceleration only after the array-first design is in
  place.

The first extra dependency worth considering is:

- `numba`

It should remain optional unless it proves clearly valuable.

## Diagnostics Model

Diagnostics should be structured data first and plots second.

V2 diagnostics should be emitted as arrays or small tables that include:

- local estimate value
- local spread
- confidence
- accepted window size
- region separation metrics
- inter-day consistency metrics
- final smoothing weights

Plotting code should consume those diagnostics, not generate them inline from
the estimator internals.

## Validation Strategy

Before changing the science, the repo should freeze a benchmark corpus.

The benchmark set should include:

- the current SuperMAG example year;
- known problematic intervals in `E`, `N`, and `Z`;
- synthetic cases:
  - unimodal narrow;
  - unimodal broad;
  - bimodal balanced;
  - bimodal unbalanced;
  - skewed with tail;
  - sparse with gaps.

Every major V2 change should be compared on:

- runtime;
- Step 1c accepted/rejected status stability;
- window-size behavior;
- `QD` agreement with the current reference path;
- `QD` agreement with SuperMAG;
- behavior through missing-data intervals.

## Migration Plan

The recommended order of work is:

1. Create `baseline_v2` package skeleton and typed containers.
2. Build array-first adapters around the current scripts and data.
3. Implement a V2 reference Step 1c path that reproduces current logic.
4. Add regression and benchmark tests against the current implementation.
5. Replace the Step 1c inner loop with incremental histogram accumulation.
6. Add the robust dominant-region Step 1c estimator behind a config switch.
7. Move Step 1d and Step 2 toward uncertainty-aware smoothing.
8. Decide whether a more global latent-state model is justified.

## The Truly Modern Direction

The fully modern endpoint should not treat Step 1c as the final estimator. It
should treat Step 1c as a local observation model for a latent baseline.

The underlying signal decomposition is:

```text
x(t) = qd(t) + qy(t) + disturbance(t) + noise(t)
```

where:

- `qd(t)` is a smooth semi-hourly baseline process;
- `qy(t)` is a slower varying long-timescale baseline process;
- disturbance and noise are not well modeled by a single Gaussian;
- observation quality changes strongly over time.

### Modern Deterministic Formulation

The most practical first modern formulation is a robust optimization problem.

In this view:

- local windowed histogram analysis produces candidate observations of `qd`;
- each candidate observation carries uncertainty or confidence;
- the final `qd` and `qy` curves are solved globally by minimizing a robust
  objective.

A representative objective would combine:

- robust data fidelity to local quiet-time observations;
- smoothness penalty on `qd`;
- stronger smoothness penalty on `qy`;
- optional penalties on implausible local jumps;
- missing-data-aware masking.

This is a major improvement over the current chained
estimate-interpolate-smooth-estimate-smooth structure, while still remaining
tractable and debuggable.

### Modern Probabilistic Formulation

The longer-term endpoint is a latent-state or state-space model.

In that view:

- `qd` and `qy` are hidden states;
- local quiet-time windows generate noisy observations of those states;
- uncertainty is time-varying and component-dependent;
- inference is performed with a Kalman-style smoother, robust iterative
  reweighting, or an EM-like procedure.

Possible advantages:

- uncertainty is propagated naturally through the pipeline;
- gap handling becomes part of the inference problem, not an edge case;
- Step 1c ambiguity no longer needs to be represented only as a binary accept
  or reject decision.

### Recommendation

The probabilistic route is the cleanest long-term architecture, but it is not
the right starting point for the branch.

The recommended progression is:

1. array-first compatibility engine;
2. robust deterministic Step 1c estimator;
3. uncertainty-aware global smoothing;
4. optional experimental latent-state prototype.

This preserves debuggability while still moving toward a truly modern model.

## Open Questions

The main design questions that V2 should answer experimentally are:

1. Is center-day weighting materially helpful in ambiguous windows?
2. Is a dominant-region estimator enough, or is a global latent-state model
   needed to remove the remaining instability?
3. Which uncertainty summary is most predictive for downstream smoothing:
   robust spread, window size, inter-day consistency, or a learned combination?
4. Should `QY` remain a separate stage, or should `QD` and `QY` eventually be
   estimated jointly?

## Initial Success Criteria

The `baseline_v2` branch should be considered successful if it can show all of
the following:

- materially lower runtime than the current estimator on the SuperMAG example;
- more stable Step 1c behavior on known ambiguous windows;
- at least comparable, and preferably better, `QD` agreement with SuperMAG;
- cleaner code boundaries than the current monolithic estimator class;
- a path to future latent-state modeling without another full rewrite.
