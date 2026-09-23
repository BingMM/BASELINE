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

1. `vault/01_Project/Current State.md`
2. `vault/05_Handoff/Handoff - Latest.md`
3. `vault/01_Project/Project Brief.md`

Then read the relevant decision or algorithm note only as needed. Treat
`vault/04_Sessions/` as history, not mandatory startup context.

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

## Scientific coding style

- Write for a small research group. The expected reader is a student or
  scientist who should be able to follow the calculation from top to bottom.
- Write for a scientist reading the code interactively in an editor. Make
  workflows visually scannable with `#%%` sections where appropriate, blank
  lines between conceptual stages, and short comments that identify each block.
- Keep the main calculation linear and visible from top to bottom. Prefer
  familiar intermediate variables and explicit operations over nested
  expressions, generic plumbing, or compressed control flow.
- Extract a helper when it represents a distinct calculation or removes
  meaningful repetition. Do not hide a few obvious sequential steps behind an
  abstraction merely to shorten the main function.
- Comments may serve as navigational headings even when the underlying Python
  is straightforward. Treat formatter conventions and line-length targets as
  secondary to human readability, while avoiding ambiguous or unwieldy code.
- Use nearby code as the stylistic baseline, not as a ceiling. Do not copy weak
  patterns blindly: identify scientific, numerical, or code choices that could
  be improved, explain the tradeoff, and propose a clearer or safer alternative.
- Adopt improvements when they materially improve correctness,
  reproducibility, clarity, or demonstrated performance. Do not add complexity
  merely because it is conventional in large production systems.
- Let complexity follow the science, numerical method, or actual reuse
  requirements. Prefer direct functions, NumPy arrays, ordinary loops and
  dictionaries, and keep the main calculation visible in execution order.
- Unless current requirements justify them, avoid dataclasses, manager or
  factory classes, generic schemas, version and compatibility frameworks,
  checkpoint/resume machinery, and speculative extension points.
- Retain scientific rigor: make units, coordinates, assumptions, provenance,
  and uncertainty explicit, and add focused tests or reference comparisons for
  consequential calculations.
- Scale packaging, validation, documentation, and abstractions to the code's
  real reuse. A reusable package may justify more structure, but that structure
  should solve a current, explained need.
- If a nominally small feature grows beyond roughly 200 lines or more than two
  new source files, pause and explain why before continuing.

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

## Automatic memory checkpoints

Project-memory maintenance is a default responsibility. Do not wait for the
user to request a vault update or announce that a session is ending.

Checkpoint after a verified fix or result, a durable implementation or
scientific decision, a changed blocker or next action, and any milestone that
would otherwise leave important understanding only in the conversation.

At a meaningful checkpoint:

1. create a session note named for the actual date only when history is worth
   preserving;
2. update `Current State.md` if verified live state changed;
3. append only durable decisions to the decision log;
4. replace the latest handoff with the next actionable state;
5. update algorithm notes only when the scientific interpretation changed;
6. refresh the handoff's `Portfolio impact` section, using `Central update
   needed: No` when no portfolio-level information changed.

Do not write raw logs, transient speculation, or unchanged state into the
vault. An explicit read-only or no-file-changes request disables automatic
memory writes for that task. Do not edit the central second brain directly;
communicate portfolio changes through the latest handoff.
