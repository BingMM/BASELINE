# Handoff Archive - 2026-04-28

> Archived on 2026-07-26 when the live handoff was consolidated. This file
> preserves historical detail and may contain superseded blockers or next
> steps. It is not startup context.

Last updated: 2026-04-28

## New Planning Context - V2 Branch

The user is now fairly happy with the current SuperMAG-style implementation and
wants the next phase to be a new, better, faster branch rather than more small
adjustments to `main`.

That design work is now documented in:

- `documentation/BASELINE_v2_design.md`

Current planning stance:

- keep `main` as the reference / paper-reproduction track;
- build `baseline_v2` as a separate modern track that is allowed to deviate
  from the paper when it improves robustness, speed, or overall fit.

Recommended first implementation step for the next session:

- create the `baseline_v2/` package skeleton and typed result/config/input
  containers before changing the science.

Status update:

- that first implementation step is now done on branch `baseline_v2`.

What exists now:

- typed containers in `baseline_v2/types.py`;
- adapters and frame conversion helpers in `baseline_v2/adapters.py`;
- native array-first `ModernVarianceEngine` in `baseline_v2/variance.py`;
- a self-contained native `ModernBaselineEngine` reference path;
- native Step 1c cache/indexing helpers in `baseline_v2/step1c_prepare.py`;
- a real reference baseline path in `baseline_v2/step1c_reference.py`;
- a real experimental robust Step 1c path in `baseline_v2/step1c_robust.py`;
- simple checkpoint helpers and a minimal V2 example script.

Additional implementation detail:

- `baseline_v2/__init__.py` and `baseline_v2/pipeline.py` now use lazy imports
  so the pure-array V2 pieces can be imported without immediately pulling in
  the pandas-based legacy reference path.

Verification now recorded:

- V2 variance code compiles and passes `tests/test_v2_variance.py`;
- V2 Step 1c preparation code compiles and passes
  `tests/test_v2_step1c_prepare.py`;
- the native V2 reference pipeline passes end-to-end tests, including Step 1c
  checkpoint reuse;
- the first robust Step 1c estimator has focused unit coverage and end-to-end
  pipeline coverage;
- `scripts/example_with_supermag_data_v2.py` is now the real SuperMAG-facing
  V2 example workflow;
- the optional direct comparison against the legacy variance estimator is
  present but skipped in this shell because `pandas` is unavailable.

Recommended next implementation step:

- run `scripts/example_with_supermag_data_v2.py` on the real 2024 DMH dataset
  and inspect the generated `reference` and `robust` outputs under
  `figures/SM_example_v2/`.

Status update:

- that real-data V2 example port is now implemented.

Current script:

- `scripts/example_with_supermag_data_v2.py`

Current output layout:

- `figures/SM_example_v2/shared/`
- `figures/SM_example_v2/reference/`
- `figures/SM_example_v2/robust/`

Current example behavior:

- runs the native V2 variance engine and baseline engine on the 2024 DMH
  SuperMAG data;
- supports `--mode reference`, `--mode robust`, or `--mode both`;
- uses V2 Step 1c checkpoints under `data/cache/step_1c_v2/`;
- adapts V2 results into the legacy plotting interface so existing chunked
  figure builders can still be used.

Usability update:

- the V2 example now keeps plotting output quiet and shows `tqdm` progress bars
  only for the long compute stages, matching the old script more closely;
- `scripts/example_with_supermag_data.py` was restored to legacy-only;
- `scripts/example_with_supermag_data_v2.py` now follows the older script's
  linear structure more closely and exposes only `--mode reference|robust`;
- `scripts/example_with_real_data_v2.py` now exists as the local CSV analogue
  of the V2 example and defaults to `--mode robust`;
- this was added because the script could otherwise look hung for long periods,
  especially while writing shared plots or doing a first uncached Step 1c run.

Current blocker:

- this shell could not execute the real-data V2 example because `netCDF4` is
  not installed here;
- the next session should reopen in an environment with `netCDF4` and
  `apexpy`, then run `python scripts/example_with_supermag_data_v2.py`.

Immediate next step for the next session:

- run the V2 example on the real 2024 DMH dataset;
- choose one V2 mode explicitly, for example
  `python scripts/example_with_supermag_data_v2.py --mode reference`;
- inspect `figures/SM_example_v2/reference/` versus
  `figures/SM_example_v2/robust/`, especially:
  - `SM_QD_comp/`
  - `SM_step_1c/`
  - `SM_step_1d/`

## Latest Step 1c Note

The current `main` branch uses the cached/incremental Step 1c implementation in
`baseline/baseline_estimator.py`:

- `_prepare_step_1c_day_bin_cache`
- `_collect_step_1c_window_chunks`
- `_expand_step_1c_window`

Important follow-up from the code review on 2026-04-26:

- the speed-up initially left one stale control-flow path from the old
  implementation;
- when `paper_mode` returned an invalid `(mu, sigma)`, Step 1c still did a bare
  `window_days += 2`, which no longer updates the cached window state;
- this has now been fixed so the `typical_value_failed` path also goes through
  `_expand_step_1c_window(...)`.

Small cleanup done at the same time:

- removed the redundant local `need_diagnostics` alias in `step_1c`;
- removed one duplicate `get_weight_sigma(...)` call on accepted nodes;
- updated the Step 1c docstrings so the cached/incremental behavior is
  documented explicitly.

## Formal Public Documentation

A formal public-facing algorithm document now exists at:

- `documentation/BASELINE_algorithm.md`

It is intended for new users of the repository, not just for session handoff.
It documents:

- coordinate rotation
- variance estimation
- baseline estimation steps 1 and 2
- the current paper-style typical-value estimator
- checkpointing
- diagnostics
- missing-data behavior
- known deviations from the SuperMAG paper

## Read First

This repo is trying to reproduce the SuperMAG webservice baseline algorithm. Current focus is the daily baseline `QD`, specifically the typical value estimator and Step 1c acceptance behavior.

## Current Code Direction

Use the paper-style typical-value path.

It is intended to better follow the article:

- fixed 1 nT histogram bins;
- typical value is the mode;
- Gaussian sigma comes from a full-histogram fit;
- broad or multimodal windows should fail acceptance and force Step 1c window expansion.

The old `mode` and `irls` methods have been removed from the codebase.

The estimator API has also been simplified:

- `BaselineEstimator` no longer carries dead `typical_value_method` or `typical_value_histogram_bins` plumbing;
- `get_typical_value()` now directly calls the paper-style implementation;
- the placeholder Step 3 `get_QO()` hook was removed from the baseline run path.

The SuperMAG example script was also structurally reduced:

- repeated component-estimator setup now runs through one helper;
- repeated triplet plots and Step 1a-2c plotting blocks were consolidated;
- the script dropped from 791 lines to 697 lines without changing exported filenames or chunk layout.

That refactor was then taken one step further:

- plotting and chunk-export helpers were moved into `scripts/supermag_example_plotting.py`;
- `scripts/example_with_supermag_data.py` now mainly orchestrates loading, estimation, and plot calls;
- the main script is now 295 lines, with behavior and output paths preserved.

## Immediate Next Step

Run the full SuperMAG example after the latest `paper_mode` change:

```bash
python scripts/example_with_supermag_data.py
```

Then inspect:

- `figures/SM_QD_comp.png`;
- `figures/QD_diag/N/`;
- `figures/QD_diag/Z/`;
- `figures/QD_diag/E/`.

Look for whether the afternoon N/Z mismatches improve and whether formerly ambiguous histograms now widen/reject rather than pass.

## Latest Preliminary Inspection

E-component histograms were inspected after the full-histogram `paper_mode` revision. They look mixed:

- some E windows are reasonably single-peaked;
- some remain multimodal or tailed but still accepted;
- at least one E window used a very wide 41-day Step 1c window;
- `SM_QD_comp.png` appeared older than the inspected histograms, so regenerate the comparison figure before drawing final conclusions.

User domain calibration: lower E modes can be the quiet signal, so do not treat lower-mode selection as automatically bad. The specific `2024-03-07 05:15` and `2024-03-06 18:15` E histograms were considered encouraging by the user. The current concern is many visually scattered histograms passing with only 3-day windows.

The histogram PNGs now include embedded diagnostics and use a minimum x-axis range of `[-100, 100]` nT. Regenerate plots before inspecting this change.

## Latest Result Interpretation

In `figures/SM_QD_comp.png`, E around `2024-03-06 06:00-12:00` suggests the current mismatch is mainly a Step 1d smoothing issue, not a Step 1c typical-value issue:

- orange Step 1c nodes roughly follow SuperMAG;
- green local QD underestimates the crest.

Primary suspect:

- `weighted_gaussian_smooth` with `step_1d_adaptive_sigma=True`, because the current implementation effectively uses source-node-dependent smoothing widths and may let low-confidence nodes flatten nearby peaks.

Recommended next test:

- run E with `step_1d_adaptive_sigma=False` and compare the same panel before changing Step 1c again.

## Newer Result

The user tested:

- `step_1d_adaptive_sigma=False`
- `step_1d_sigma_days=1/12`

This improved `figures/SM_QD_comp.png` substantially compared with the earlier smoothing setup.

Updated interpretation:

- Step 1d smoothing was a real issue, but it is no longer the dominant remaining problem.
- The remaining mismatches now mostly appear to come from Step 1c accepted nodes that are still ambiguous, broad, or only weakly modal.

What to inspect next:

- accepted windows with `sigma/threshold` close to 1,
- accepted windows with low `mode_dominance`,
- `spike_replaced` windows in N,
- accepted Z windows near threshold with broad or tailed histograms.

## Mode Dominance Experiment

A standalone Step 1c mode-dominance rejection gate was tried and then removed.

Reason:

- it rejected too many nodes, already obvious from the E-component partial run;
- `mode_dominance` is too blunt as a standalone rejection criterion.

Current state:

- no active mode-dominance rejection in code;
- `mode_dominance` remains available as a diagnostic on the histogram plots;
- `core_fraction` remains on the TODO as the next ambiguity diagnostic to try.

## New Interpretation After Re-reading The Paper

The earlier idea that Figure 4 implies a smoothed histogram is probably wrong. The user is confident the paper simply connects the tops of the bins, and the plot only looks relatively smooth because it uses 16 days of data.

Updated hypothesis:

- the remaining mismatch is more likely in how equations (4) and (5) are interpreted on sparse 3-day histograms;
- equation (5) likely refers to a dominant modal region/plateau rather than exact ties between highest-count bins;
- equation (4) has now been updated to an explicit nearest-neighbor comparison, but it still needs to be evaluated on the SuperMAG output.

This is probably the next big place to investigate if Step 1c remains unstable.

## Current Risk

Full-histogram Gaussian fitting may be too strict if tails dominate even when the quiet-time peak is visually clear. If so, the next adjustment should not return to local fitting. Instead consider a robust full-distribution fit or fitting the dominant contiguous distribution after objectively identifying it.

## Important Correction

A previous quick read from a reduced-size N-component contact sheet was wrong for at least one panel.

- `figures/QD_diag/N/N_20240307T164500.png` is not a large-offset replacement case. In the full-size plot, the histogram mode, Gaussian center, and typical value nearly overlap, and the diagnostic box reports `fit mu - mode=-0.4 nT`.
- The large negative offset previously attributed to `16:45` was actually from `figures/QD_diag/N/N_20240307T174500.png`, which reports `fit mu - mode=-42.8 nT`.

Do not trust tiny montage text for panel-specific numbers. Read the individual full-size PNGs before drawing numerical conclusions about equation-(4) behavior.

## N Afternoon Diagnosis

For `figures/SM_QD_comp.png` around `2024-03-07 12:00-18:00` in the N component, do not attribute the alternating sign mainly to Gaussian-center replacement.

Current best reading from the full-size diagnostics:

- the accepted Step 1c modal values themselves alternate sign across adjacent half-hour bins on sparse multimodal histograms;
- some adjacent windows widen to 7-17 days and settle on a positive population, while nearby 3-day windows still accept a negative population;
- the instability therefore looks more like modal-bin hopping between competing populations than a simple `mu_fit` vs mode sign mismatch.

This strengthens the case that equation (5) interpretation, or more generally the definition of the dominant modal region in sparse 1 nT histograms, is the next place to investigate.

## Weighting Note From The Paper

The article wording supports the user's recall that the weighted fit uses more than the fitted Gaussian `sigma`.

- In Step 1d (`[37]`), the paper says each semi-hourly value provides knowledge of the "spread in the data" and the "width of the window used."
- In Step 2 (`[41]` and section `7.1`), the weighting is explicitly based on a non-trivial uncertainty with two parts: instantaneous variance `v` and a history-dependent term `d`, combined as `U = v + d`.

Practical implication:

- For Step 1d, the paper likely expects a weight based on more than just the fitted Gaussian width. Window width is explicitly part of the uncertainty, and "spread in the data" may refer to the broader distribution scatter rather than only the Gaussian-fit `sigma`.
- Our current Step 1d weighting in code is therefore an implementation choice, not something clearly specified by the paper.

## Candidate Extension - Center-Day Weighting In Step 1c

The user proposed a plausible extension for the N-component instability: because Step 1c combines data from multiple days to estimate the QD for one target day, the target day itself may deserve higher weight than neighboring days.

Why this may matter:

- the problematic N histograms often show two clear populations of opposite sign, which looks more like competing day-to-day states than a main mode plus outliers;
- equal counting across days may let neighboring days dominate the modal bin even when the target day favors the other population.

Status:

- this is not described in the paper and should be treated as an extension, not part of the paper-faithful baseline;
- if tried later, implement it by using day-distance weights when accumulating the Step 1c histogram rather than by post-processing the selected mode.

## New Diagnostic - Per-Day Step 1c Heatmap

The Step 1c histogram diagnostics now include a second panel showing per-day contributions using the same histogram bins as the main plot.

- x-axis: residual field bins
- y-axis: day offset from the target day
- color: count contributed by that day to that bin

Use this before changing the algorithm again. It should tell us whether the competing N-component modes are coming from different days in the window or whether each day already contains both populations.

Note:

- the per-day panel no longer includes the long offset-to-date text strip; that overlay extended outside the axes and caused saved diagnostic PNGs to become excessively wide.
- the heatmap also no longer includes a colorbar.

## Current Step 1c Start Window

The initial Step 1c window is now configurable with `step_1c_min_window_days`.

- default in the estimator: `3`
- validation: positive odd integer only
- current example-script setting: `5` for E, N, and Z

## Current Best Result

The run with `step_1c_min_window_days=5` looks materially better than the earlier 3-day-start runs.

Most notable improvement:

- `Bn` around `2024-03-07 12:00-18:00` is much more stable;
- several formerly unstable N windows now widen to larger accepted windows instead of passing early with competing-sign modes;
- inspected examples include `N_20240307T154500.png` widening to `55 days` and `N_20240307T164500.png` / `N_20240307T171500.png` widening to `11 days`.

Current interpretation:

- the instability was partly due to poor statistics in the initial 3-day Step 1c window;
- a 5-day starting window appears to be a strong candidate default for further testing.

## Z Failure Diagnosis

The remaining Z failures around `2024-03-06 13:00-15:00` and `2024-03-07 13:00-15:00` are not the same as the earlier N sign-hopping problem.

Current best reading from the full-size diagnostics:

- the Z histograms often have a reasonable-looking central positive mode but also a long positive tail;
- the full-histogram Gaussian fit broadens to include that tail and ends up with `sigma > FWHM_stat`, causing rejection even at very wide windows;
- neighboring accepted Z windows often sit just below the threshold, so the current pass/fail boundary is marginal in those intervals.

This means the main Z issue is the broad skewed distribution and the full-distribution sigma criterion, not mode sign instability.

## Step 2 Gap Handling

There was a real Step 2 bug visible in `figures/SM_step_2b.png`:

- Step 2a was generating daily values for days with no finite `x_QD` samples on the target day;
- those gap-day nodes then influenced the Step 2b smoothing.

Current behavior after the fix:

- if a target day has zero finite `x_QD` samples, `step_2a()` now writes `NaN` and weight `0.0` for that day and does not attempt a windowed estimate.

## Full-Year Example Plots

The example script now writes chunked non-diagnostic plots across the full record into:

- `figures/SM_example/SM/`
- `figures/SM_example/SM_db/`
- `figures/SM_example/SM_QD/`
- `figures/SM_example/SM_QY/`
- `figures/SM_example/SM_step_1a/` through `SM_step_2c/`
- `figures/SM_example/SM_db_comp/`, `SM_QD_comp/`, and `SM_QY_comp/`

Current chunk sizes:

- Step 1 context: `7` days
- Step 1 detail: `2` days
- Step 2: `60` days

Top-level summary PNGs in `figures/` are still produced.

## Missing-Day Behavior In QD Plots

Gray shading in `SM_QD_comp` marks missing raw component data, not just missing diagnostics.

Current behavior after the latest fix:

- Step 1c does not estimate a semi-hourly value for a target bin if that target bin has zero finite input samples on the target day;
- Step 1d masks `QD` to `NaN` anywhere the original component `x` is missing;
- Step 2b masks `QY` to `NaN` anywhere `x_QD` is missing.

So fully missing days should no longer show local baseline estimates through the gap.

## Chunk Plot Skipping

The chunked `figures/SM_example/...` export now skips fully empty windows:

- Step 1 exports skip a chunk if the plotted component `x` has no finite values there.
- Step 2 exports skip a chunk if `x_QD` has no finite values there.
- Triplet plots skip only when all relevant component series are fully missing in the chunk.

This is intentional; missing chunks in the export folders may simply mean the entire plotted window had no data.

## QY Comparison Gaps

The observed case where `SM_QY_comp` showed local `QY` through SuperMAG `QY` gaps is the same missing-data issue as the QD case, not a separate algorithmic bug.

Verified from the source NetCDFs:

- the SuperMAG `QY` breaks checked in early 2024 correspond to intervals with zero finite raw samples.

Therefore:

- after the current masking fix, regenerated `QY` comparison plots should also break through those gaps.
- if an existing `SM_QY_comp` plot still shows continuity there, treat it as stale output.

## Step 2 Tuning Knob

The example script now exposes the yearly-trend smoothing width as:

- `STEP_2B_SIGMA_DAYS = 30.0`

and passes it to all three estimator runs.

Use this when tuning against `SM_QY_comp`; SuperMAG `QY` currently has more structure than the local estimate.

## Fast Reruns - Step 1c Checkpoints

The expensive part of the run is Step 1c. The estimator now supports checkpointing that stage and reusing it on later runs.

API:

- `step_1c_checkpoint_path`
- `reuse_step_1c_checkpoint`
- `write_step_1c_checkpoint`

Current example-script defaults:

- `STEP_1C_CHECKPOINT_DIR = DATA_DIR / "cache" / "step_1c"`
- `REUSE_STEP_1C_CHECKPOINT = True`
- `WRITE_STEP_1C_CHECKPOINT = True`

Behavior:

- on the first run, Step 1c is computed and written;
- on later runs with the same component/method/min-window setup, Step 1c is loaded and the script resumes from Step 1d onward.

## QD Timing Mismatch

The bad timing seen in `SM_QD_comp_20240202_20240203.png` is not a plotting artifact.

- `plot_qd_component()` plots both the SuperMAG series and the local `QD` against the same time axis `t[view_slice]`.
- There is no extra time transform in the plotting layer.

So when the local curve looks shifted by hours, treat that as an estimator problem, not a figure-rendering problem.

## Post-Gap Distortion Bug

The major QD failure after the first long missing interval was traced to interpolation/smoothing, not cache reuse.

Root cause:

- `cubic_convolution_interpolate()` was dropping NaN nodes and then assuming the remaining nodes were still uniformly spaced;
- after a long gap, that invalidates the interpolation index basis and corrupts the curve after the gap;
- `weighted_gaussian_smooth()` also filled missing target nodes with synthesized values, which created support points inside gaps.

Current fix:

- preserve NaN nodes in the interpolation grid and ignore them only locally in the stencil;
- keep missing/zero-weight smoothing targets as NaN.

Implication:

- rerunning with `REUSE_STEP_1C_CHECKPOINT = True` should be enough to test this fix, because Step 1c itself is not the part that changed.

## Current Step 1d Side Effect

After the post-gap interpolation fix, isolated rejected Step 1c nodes can now create local discontinuities in the final `QD`.

Confirmed example:

- `SM_QD_comp_20240218_20240219.png`, `Bu`, around `2024-02-19 13:45`

Details:

- raw data are present there;
- Step 1c has `status='fwhm_rejected'` at `2024-02-19 13:45` with zero weight;
- the current `weighted_gaussian_smooth()` leaves zero-weight target nodes as `NaN`, so the Step 1d curve breaks at that time.

This is different from the raw-data-gap bug. The next fix should treat true missing-input bins differently from isolated rejected Step 1c bins.

Status update:

- this has now been patched;
- Step 1d smooths through isolated rejected Step 1c nodes again;
- only nodes with Step 1c status `missing_input` are forced to remain NaN in the smoothed semi-hourly series.

## Important User Preference

The user wants the vault to serve AI continuity. Keep it updated when decisions or test outcomes change.

## Step 1c Runtime Clarification

The later cleanup/refactor commits are not what made Step 1c much faster.

The actual speedup source is earlier commit `319ae83`, which changed Step 1c in two material ways:

- bins with no finite target-day input are now skipped immediately as `missing_input`;
- acceptance now uses `sigma <= FWHM_stat` directly, instead of the older stricter `2.355 * sigma <= FWHM_stat`.

Current 2024 data have heavy gaps:

- `7,225 / 17,568` semi-hourly targets per component are skipped immediately (`41.1%`).

So if Step 1c runtime is discussed again, do not attribute that change to the code cleanup or plotting-module refactor.

## Step 1c Weighting Update

Step 1c weight normalization now uses the configured minimum window:

- old: `window_days / 3`
- new: `window_days / step_1c_min_window_days`

This keeps the weighting consistent with runs that start from `5` days instead of `3`.

## SuperMAG Example Plot Outputs

The SuperMAG example no longer writes the legacy root-level summary PNGs in `figures/`.

Current behavior:

- non-diagnostic example plots are chunked only and written under `figures/SM_example/`.
- SuperMAG Step 1c diagnostics are written under `figures/SM_example/QD_diag/`.

## Inverse XYZ Rotation

There is now a separate inverse helper in `baseline/coordinate_rotator.py`:

- `InverseCoordinateRotator(rotator)`
- `rotate_baselines(be_e, be_n, be_u)`

It takes:

- a completed `CoordinateRotator` with `q(t)` already estimated;
- finished `BaselineEstimator` objects for `E`, `N`, and vertical `Z`.

It returns one dataframe with:

- original `X`, `Y`, `Z`
- `q`
- `QD_X`, `QD_Y`, `QD_Z`
- `QY_X`, `QY_Y`, `QY_Z`
- `baseline_X`, `baseline_Y`, `baseline_Z`
- `X_corr`, `Y_corr`, `Z_corr`

The real-data example now exercises this path and writes `real_xyz_baseline.png`.

## Inline Docs Status

Core inline documentation was refreshed after the recent estimator/rotation changes.

Files updated:

- `baseline/baseline_estimator.py`
- `baseline/coordinate_rotator.py`
- `scripts/example_with_supermag_data.py`
- `scripts/example_with_real_data.py`

So if documentation is questioned again, the main stale docstrings from the old API/layout should already be fixed.

## Step 1c Optimization Work

Active branch:

- `optimize-step1c`

First pass already done in `baseline/baseline_estimator.py`:

- precompute Step 1c day/bin caches once;
- remove repeated dataframe masking from the inner loop;
- expand windows by updating cached aggregates;
- skip building paper-mode diagnostics when they are not requested.

This is meant to speed up Step 1c without changing the scientific logic.

The next obvious hotspot, if more speed is needed, is still the repeated histogram + `curve_fit()` work inside `_get_typical_value_paper_mode()`.

Current branch state:

- later experiments (profiling hooks, warm-start, custom Gaussian fitter) were
  reverted;
- only the first safe Step 1c optimization remains on `optimize-step1c`.
