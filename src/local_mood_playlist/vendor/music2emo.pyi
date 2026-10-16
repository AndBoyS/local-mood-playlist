"""Stub for the vendored Music2Emotion inference module.
"""

from pathlib import Path
from typing import TypedDict


class Music2emoPrediction(TypedDict):
    valence: float
    arousal: float
    predicted_moods: list[str]
    mood_scores: dict[str, float]


class Music2emo:
    def __init__(
        self,
        model_weights: str | Path | None = None,
        work_dir: str | Path | None = None,
        chord_model_weights: str | Path | None = None,
        *,
        mert_snapshot: str | Path,
        device: str,
    ) -> None: ...

    def predict(self, audio: str | Path, threshold: float = 0.5) -> Music2emoPrediction: ...


__all__ = ["Music2emo", "Music2emoPrediction"]
