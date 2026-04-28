from __future__ import annotations

from typing import Any

import numpy as np


def dataframe_to_numpy_dict(df) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"t": df.index.to_numpy()}
    for column in df.columns:
        diagnostics[column] = df[column].to_numpy()
    return diagnostics


def sigma_ratio_confidence(
    sigma: np.ndarray,
    threshold: np.ndarray,
    status: np.ndarray,
) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    status = np.asarray(status)
    confidence = np.zeros_like(sigma, dtype=float)
    valid = (
        np.isfinite(sigma)
        & np.isfinite(threshold)
        & (threshold > 0)
        & (status == "ok")
    )
    ratio = np.full_like(sigma, np.nan, dtype=float)
    ratio[valid] = sigma[valid] / threshold[valid]
    confidence[valid] = np.clip(1.0 - ratio[valid], 0.0, 1.0)
    confidence[(status == "missing_input") & ~valid] = np.nan
    return confidence
