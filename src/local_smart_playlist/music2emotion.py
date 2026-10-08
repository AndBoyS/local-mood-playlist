"""In-process adapter for vendored Music2Emotion inference."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable

from local_smart_playlist.predictions import TrackPrediction

MODEL_VERSION = "m2e-c5f7984-j95926289-btc11c6ed-tags-c53cee67-mert12af15fe-patch2"


@runtime_checkable
class _Music2EmotionRuntime(Protocol):
    def predict(self, audio: str | Path, threshold: float = 0.5) -> dict[str, object]: ...


def _load_model() -> _Music2EmotionRuntime:
    repository_root = Path(__file__).resolve().parents[2]
    vendor_path = repository_root / "vendor" / "music2emotion"
    vendor_directory = str(vendor_path)
    if vendor_directory not in sys.path:
        sys.path.insert(0, vendor_directory)

    module = import_module("music2emo")
    model_factory: object = getattr(module, "Music2emo", None)
    if not callable(model_factory):
        raise TypeError("vendored module has no Music2emo model factory")
    model: object = model_factory()
    if not isinstance(model, _Music2EmotionRuntime):
        raise TypeError("vendored Music2emo model does not implement predict()")
    return model


class Music2Emotion:
    """Load one MERT/Music2Emotion model in current Python process."""

    def __init__(self) -> None:
        self._model = _load_model()

    def predict(self, audio_path: Path) -> TrackPrediction:
        result = self._model.predict(audio_path.expanduser().resolve(strict=True), threshold=0.5)
        scores = result.get("mood_scores")
        valence = result.get("valence")
        arousal = result.get("arousal")
        if not isinstance(scores, dict):
            raise TypeError("Music2Emotion output has no mood_scores mapping")
        if (
            not isinstance(valence, (int, float))
            or isinstance(valence, bool)
            or not isinstance(arousal, (int, float))
            or isinstance(arousal, bool)
        ):
            raise TypeError("Music2Emotion output has invalid valence/arousal")

        validated_scores: dict[str, float] = {}
        for tag, score in scores.items():
            if not isinstance(tag, str) or not isinstance(score, (int, float)) or isinstance(score, bool):
                raise TypeError("Music2Emotion output has invalid mood score")
            validated_scores[tag] = float(score)
        return TrackPrediction(validated_scores, float(valence), float(arousal))
