# Current State

Reviewed: 2026-07-26
Code snapshot: `baseline_v2` at `d979b94`
Vault integration: versioned in the repository; local `.obsidian/` state ignored

## Current position

Development is paused on the modern V2 track. The repository has no commits
containing implementation changes after 2026-04-28, so the pause did not
conceal newer code changes. The later vault-integration commit is
documentation-only and does not change scientific or implementation state.

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
- On 2026-07-26, the focused V2 suite passed 14 tests with 1 optional
  comparison skipped, and the broad Python syntax check passed.
- The old handoff claim that real-data execution was blocked by missing
  `netCDF4` is superseded as a project-state claim. Those libraries remain
  optional environment prerequisites.

## Not yet confirmed

- The latest commit message reports roughly 100x faster execution with
  "sameish" results, but this documentation review did not reproduce that
  benchmark.
- Scientific equivalence or improvement of robust V2 over the reference path
  has not been established quantitatively.
- Committed figures and checkpoints have not been regenerated during this
  review.

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
