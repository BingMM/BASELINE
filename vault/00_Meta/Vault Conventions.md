# Vault Conventions

This vault optimizes for fast, reliable reorientation. It is durable project
memory, not an exhaustive transcript and not authority over live code.

## Source-of-truth order

1. Live code, Git state, tests, and regenerated results.
2. `01_Project/Current State.md` and `05_Handoff/Handoff - Latest.md`.
3. Durable decision and algorithm notes.
4. Dated session archives.

When these disagree, verify the repository and repair the higher-level live
notes. Do not silently carry a contradiction forward.

## Note ownership

- `START HERE - AI Onboarding.md`: human and Obsidian entry point.
- `01_Project/Project Brief.md`: stable purpose, scope, and track definitions.
- `01_Project/Current State.md`: one authoritative snapshot of the live state.
- `02_Algorithm/`: current scientific interpretations and implementation maps.
- `03_Decisions/Decision Log.md`: durable decisions with rationale.
- `04_Sessions/`: dated history, one file per actual session date.
- `05_Handoff/Handoff - Latest.md`: concise next-session operational handoff.

## Status and evidence

Use explicit labels when a scientific or implementation claim could be
misread:

- `Confirmed`: supported by cited code, test, benchmark, or inspected result.
- `Hypothesis`: plausible but not adequately tested.
- `Superseded`: retained for history but no longer current.

Record the branch and commit for important results. A committed figure is not
self-validating; note the relevant input and configuration.

## Update style

Prefer short dated summaries containing:

- what changed and why;
- evidence or verification;
- what remains untested;
- repository-relative code, test, or figure paths.

Avoid raw terminal output, pasted diffs, unlabeled speculation, and duplicated
narratives. Link to the note that owns the detail.

## Live-note limits

- Rewrite `Current State.md`; do not append a running diary.
- Keep `Handoff - Latest.md` below roughly 100 lines.
- Move superseded handoff details into a dated session archive.
- Never add later dates to an older session file.

## Session closeout

After a meaningful project session:

1. Create `04_Sessions/YYYY-MM-DD.md` for the actual date.
2. Update `Current State.md` if live state changed.
3. Append durable decisions to the decision log.
4. Replace the latest handoff with the next actionable state.
5. Update algorithm notes only when interpretation changed.

A documentation-only session may update only the affected meta notes and
handoff. Do not manufacture scientific progress.
