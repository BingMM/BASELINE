# Repository guidance

## Purpose and project tracks

BASELINE implements the baseline-removal workflow described in the 2012
SuperMAG data-processing paper.

Keep the two implementation tracks distinct:

- `baseline/` is the reference, paper-oriented implementation.
- `baseline_v2/` is the experimental array-first implementation and may
  deliberately deviate from the paper for robustness, speed, or fit quality.

Never infer the active track from old notes alone. Check the current branch,
worktree, recent commits, and live code before editing.

## Project memory

Before substantive project work, read:

1. `BASELINE_vault/01_Project/Current State.md`
2. `BASELINE_vault/05_Handoff/Handoff - Latest.md`
3. `BASELINE_vault/01_Project/Project Brief.md`

Then read the relevant decision or algorithm note only as needed. Treat
`BASELINE_vault/04_Sessions/` as history, not mandatory startup context.

The authority order is:

1. live code, Git state, tests, and regenerated results;
2. `Current State.md` and the latest handoff;
3. decision and algorithm notes;
4. dated session archives.

If a note conflicts with the repository, follow the repository and update the
live memory notes when the task changes project understanding.

## Working rules

- Preserve unrelated or pre-existing worktree changes.
- Keep scientific reproduction claims separate from hypotheses and V2
  experiments.
- Label durable scientific statements as `Confirmed`, `Hypothesis`, or
  `Superseded` when their status could be ambiguous.
- Use repository-relative paths in documentation.
- Do not treat committed figures or checkpoints as current evidence until
  their inputs, configuration, and commit are verified.
- Do not run the expensive full-year examples unless the task requires them.
- Keep `Handoff - Latest.md` concise and replace obsolete content instead of
  appending indefinitely.

## Verification

Run the focused V2 unit suite with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

A broad syntax check is:

```bash
python -m py_compile baseline/*.py baseline_v2/*.py scripts/*.py
```

The SuperMAG V2 workflow is:

```bash
python scripts/example_with_supermag_data_v2.py --mode reference
python scripts/example_with_supermag_data_v2.py --mode robust
```

Those examples require `netCDF4` and `apexpy`, which are not declared in the
base `pyproject.toml`, and they can generate or overwrite many checkpoints and
figures. Confirm the environment and intended output scope before running them.

## Memory closeout

After a meaningful project session:

1. create one session note named for the actual date;
2. update `Current State.md` if the live project state changed;
3. append only durable decisions to the decision log;
4. replace the latest handoff with the next actionable state;
5. update algorithm notes only when the scientific interpretation changed.
