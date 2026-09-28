# Current State

Reviewed: 2026-09-28
Code snapshot: `baseline_v2` at `954ba50` with an uncommitted coordinate-rotation fix
Vault integration: versioned in the repository; local `.obsidian/` state ignored

## Current position

Broader development remains paused on the modern V2 track. The committed
implementation has not changed since 2026-04-28; the working tree now contains
a bounded coordinate-rotation fix made on 2026-09-28. The later committed
vault-integration change remains documentation-only.

The repository currently contains:

- the reference implementation under `baseline/`;
- a self-contained array-first V2 implementation under `baseline_v2/`;
- V2 reference and robust Step 1c modes;
- focused V2 tests under `tests/`;
- real-data V2 scripts for the 2024 DMH SuperMAG dataset and local CSV data;
- committed reference and robust checkpoints and plot outputs.

## Confirmed

- `baseline_v2` is the checked-out branch and tracks its remote.
- `baseline_v2/step1c_reference.py` and
  `baseline_v2/step1c_robust.py` are both implemented.
- `scripts/example_with_supermag_data_v2.py` supports explicit `reference` and
  `robust` modes.
- Existing V2 outputs are present under
  `figures/SM_example_v2/reference/` and
  `figures/SM_example_v2/robust/`.
- The coordinate rotator now evaluates declination with a configurable
  angle-specific histogram width of `0.1 degree` instead of applying the
  baseline estimator's `1 nT` default directly to radians. The regression in
  `tests/test_coordinate_rotator.py` recovers the expected `21.8 degree`
  orientation for mean components near X=7500 nT and Y=3000 nT.
- On 2026-09-28, all 16 focused tests passed and the broad Python syntax check
  passed.
- On the full 525,600-sample `DMH_1min_2025.csv` input, the repaired rotator
  produced a median declination of -12.77 degrees. Over the plotted first week,
  the mean Y component changed from -1654.7 nT to a rotated E mean of 1.0 nT.
  `figures/real_rotation.png` was regenerated and visually inspected.
- The old handoff claim that real-data execution was blocked by missing
  `netCDF4` is superseded as a project-state claim. Those libraries remain
  optional environment prerequisites.

## Not yet confirmed

- The latest commit message reports roughly 100x faster execution with
  "sameish" results, but this documentation review did not reproduce that
  benchmark.
- Scientific equivalence or improvement of robust V2 over the reference path
  has not been established quantitatively.
- Baseline-estimation figures and checkpoints have not been regenerated during
  this review; only the coordinate-rotation figure was regenerated.
- The coordinate-rotation fix and its project-memory checkpoint are not yet
  committed.

## Current research focus

Establish a reproducible comparison between V2 `reference` and `robust`:

1. inspect the already committed output around known difficult intervals;
2. record exact configurations and checkpoint provenance;
3. benchmark runtime on the same input and environment;
4. quantify QD agreement with both the reference path and SuperMAG;
5. decide whether the robust dominant-region estimator should remain
   experimental or become the V2 default.

## Risks and maintenance debt

- `documentation/BASELINE_v2_design.md` still reads partly as a proposal even
  though much of the package now exists.
- `TODO.md` contains legacy items that appear superseded and needs a separate
  code-grounded audit before it is used as the work queue.
- `netCDF4` and `apexpy` are required by the SuperMAG example but are not
  declared in the base `pyproject.toml`.
- Large generated checkpoints and figures are committed; do not assume they
  represent the current configuration without checking provenance.
