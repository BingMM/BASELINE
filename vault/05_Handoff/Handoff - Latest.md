# Handoff - Latest

Last updated: 2026-07-26
Verified against: `baseline_v2` code at `d979b94`

## Project in one paragraph

BASELINE reproduces the baseline-removal workflow from the 2012 SuperMAG
data-processing paper. `baseline/` is the paper-oriented reference track.
`baseline_v2/` is a faster array-first experimental track with both reference
and robust Step 1c estimators. Work is currently paused on the V2 branch.

## Live repository snapshot

At the 2026-07-26 documentation review:

- the worktree was clean before the onboarding changes;
- `baseline_v2` matched `origin/baseline_v2`;
- the latest code commit was `d979b94` from 2026-04-28;
- the project vault and root `AGENTS.md` were integrated as versioned
  documentation while local `.obsidian/` state remained ignored;
- the vault integration made no source-code or scientific changes;
- no later project work was found;
- V2 unit tests, SuperMAG-facing scripts, checkpoints, and reference/robust
  figure outputs were present;
- the focused V2 suite passed 14 tests with 1 skipped, and the broad syntax
  check passed.

Inspect Git and the live code again before making changes. This snapshot is
orientation, not authority.

## Current implementation focus

The next research task is to validate V2's robust dominant-region Step 1c
estimator against the V2 reference path:

1. inspect existing `reference` and `robust` plots around the known difficult
   March 6-7 intervals;
2. identify the exact configuration and checkpoint provenance;
3. benchmark both modes on identical inputs and hardware;
4. quantify QD agreement with the reference pipeline and SuperMAG;
5. decide whether robust Step 1c is sufficiently accurate and stable to remain
   the main V2 direction.

Do not begin by redesigning the estimator. First establish the reproducible
comparison that the historical work did not record formally.

## Relevant entry points

- `baseline_v2/step1c_reference.py`
- `baseline_v2/step1c_robust.py`
- `baseline_v2/types.py`
- `scripts/example_with_supermag_data_v2.py`
- `tests/test_v2_reference_pipeline.py`
- `tests/test_v2_robust_step1c.py`
- `figures/SM_example_v2/reference/`
- `figures/SM_example_v2/robust/`
- `documentation/BASELINE_v2_design.md`

## Verification commands

Focused tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Full-year examples, only when their outputs are intentionally in scope:

```bash
python scripts/example_with_supermag_data_v2.py --mode reference
python scripts/example_with_supermag_data_v2.py --mode robust
```

The example requires `netCDF4` and `apexpy`, which are not base project
dependencies.

## Known risks

- The historical "100x faster, sameish result" statement is not yet a
  reproducible benchmark.
- Scientific equivalence of the robust estimator is unconfirmed.
- Committed plots and checkpoints may outlive the configuration that produced
  them.
- `TODO.md` and parts of `documentation/BASELINE_v2_design.md` need a separate
  live-code audit because they still contain proposal-era or legacy wording.

## Portfolio impact

- Central update needed: No
- Changes: None
- Sync summary: BASELINE remains paused with no recorded deadline. The next
  portfolio decision remains whether robust V2 Step 1c is sufficiently
  accurate, stable, and reproducibly faster than the reference path.

## Historical detail

The former 632-line handoff is preserved at
`vault/04_Sessions/Handoff Archive - 2026-04-28.md`. Read it only when
investigating a specific prior decision or result.
