# Daily Baseline QD

Source: SuperMAG paper section 5.1.

The daily baseline `QD` is determined in five steps:

1. **Step 1a**: Estimate one typical value per day from the rotated component. This removes variations longer than one day.
2. **Step 1b**: Interpolate daily values to 1-minute cadence using cubic convolution, then subtract to form residual `temp(t)`.
3. **Step 1c**: For each day and each 30-minute bin, estimate a typical value from the residual. Start with a 3-day window: target day plus neighboring days. If there is too much scatter, symmetrically widen the window by 2 days. Repeat until a solution is found or the data range is exhausted.
4. **Step 1d**: Smooth the semi-hourly values with a weighted fit.
5. **Step 1e**: Subtract `QD` from the original component.

## Key Interpretation

The article says Step 1c accepts a solution if the Gaussian fit to the probability distribution satisfies:

```text
sigma < FWHM_statistical
```

Although the right-hand side is named `FWHM_statistical`, the text explicitly says `sigma` is the standard deviation of the Gaussian fit. The code currently compares `sigma <= FWHM_stat`, not `2.355 * sigma <= FWHM_stat`.

This matters. A previous implementation compared Gaussian FWHM instead, which was stricter and not aligned with the text.

## Diagnostic Figure

The user primarily inspects:

- `figures/SM_QD_comp.png`

This figure overlays:

- blue: SuperMAG QD;
- green: local QD;
- orange: local semi-hourly Step 1c typical values with error bars.

Histograms for the currently selected diagnostic window are under:

- `figures/QD_diag/E/`
- `figures/QD_diag/N/`
- `figures/QD_diag/Z/`
