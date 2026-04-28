from __future__ import annotations

import pickle
from pathlib import Path

from .types import BaselineResult


def save_baseline_result(path: str | Path, result: BaselineResult) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(result, handle)


def load_baseline_result(path: str | Path) -> BaselineResult:
    with Path(path).open("rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, BaselineResult):
        raise TypeError("checkpoint does not contain a BaselineResult")
    return result
