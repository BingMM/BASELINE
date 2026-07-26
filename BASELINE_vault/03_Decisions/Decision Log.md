# Decision Log

## 2026-04-20 - Keep Vault As AI Project Memory

Decision: This vault is for AI continuity, not the user's notes. Future AI sessions should read it to recover project context and should update it when durable project understanding changes.

Rationale: Conversations can terminate. The user remembers discussion and tests, but the AI will not.

## 2026-04-20 - Add Partial Step 1c Histogram Diagnostics

Decision: Add `step_1c_diagnostic_time_range=(start, stop)` to `BaselineEstimator`.

Rationale: Plotting histograms for every Step 1c node is too much. The user wants histograms corresponding to the `SM_QD_comp.png` inspection window.

Implementation:

- Constructor argument in `baseline/baseline_estimator.py`.
- `_should_plot_step_1c_diagnostic`.
- `scripts/example_with_supermag_data.py` passes `(d_start, detail_stop)`.

## 2026-04-20 - Use Fixed 1 nT Histogram Bins For Paper-Mode Typical Value

Decision: New `paper_mode` estimator uses fixed 1 nT histogram bins.

Rationale: Visual inspection of Figure 4 suggests approximately 1 nT bins. Adaptive bins caused modal jitter and made results unstable.

## 2026-04-20 - Return Mode, Use Full Gaussian Fit For Acceptance

Decision: `paper_mode` returns the histogram mode but fits a Gaussian to the full histogram distribution to estimate sigma.

Rationale: The article's Figure 4 shows a broad Gaussian fit to the central distribution, not a local fit to one modal bar. If the distribution is multimodal or broad, Step 1c should widen the window until quiet-time behavior dominates.

Rejected earlier approach:

- local modal-peak Gaussian fit;
- it accepted ambiguous multimodal histograms because a narrow local peak produced small sigma.

## 2026-04-20 - Compare Sigma Directly To FWHM_stat

Decision: Step 1c and Step 2a compare `sigma <= FWHM_stat`, not `2.355 * sigma <= FWHM_stat`.

Rationale: Article equation 8 uses `sigma` as the standard deviation of the Gaussian fit and compares it to the empirical curve named `FWHM_statistical`.

## 2026-04-27 - Keep V2 As A Separate Modern Track

Decision: The proposed `baseline_v2` work should live as a separate branch and
package track, not as an in-place rewrite of the current reference
implementation.

Rationale:

- the current `main` branch now serves as a reasonably working SuperMAG-style
  reference;
- the next branch should be free to optimize for robustness, speed, and modern
  statistical design, even when that means explicit deviations from the paper;
- keeping the tracks separate preserves a stable regression target.

## 2026-04-27 - First V2 Experimental Change Is Localized To Step 1c

Decision: The first true V2 algorithmic deviation should be a new Step 1c
local estimator, while Step 1d and Step 2 initially remain shared with the
reference-style pipeline.

Rationale:

- Step 1c is the main scientific and runtime hotspot in the current code;
- changing the local estimator first isolates the experiment and preserves a
  controlled comparison against the native reference pipeline;
- this allows `reference` and `robust` modes to share the same downstream
  smoothing/trend stages until a clear benefit from broader changes is shown.

## 2026-07-26 - Use AGENTS.md As The Automatic Entry Point

Decision: Add a concise repository-root `AGENTS.md` and keep the vault as
linked, versioned project memory.

Rationale:

- Codex automatically loads repository guidance from `AGENTS.md`;
- the vault remains useful for scientific interpretations, decisions, and
  historical continuity;
- separating concise operating guidance from deeper memory reduces startup
  context and prevents the handoff from becoming an archive.

The Markdown vault is now intended to be tracked in Git. Only local Obsidian
workspace state under `BASELINE_vault/.obsidian/` remains ignored.
