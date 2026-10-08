"""Persistent JSON-lines worker for vendored Music2Emotion (Python 3.10 env)."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import traceback
from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Music2EmotionModel(Protocol):
    def predict(self, audio: str | Path, threshold: float = 0.5) -> dict[str, object]: ...


def _load_model(work_dir: Path) -> Music2EmotionModel:
    repository_root = Path(__file__).resolve().parents[1]
    vendor_path = repository_root / "vendor" / "music2emotion"
    sys.path.insert(0, str(vendor_path))
    with contextlib.redirect_stdout(sys.stderr):
        module = import_module("music2emo")
        model_factory: object = getattr(module, "Music2emo", None)
        if not callable(model_factory):
            raise TypeError("vendored module has no Music2emo model factory")
        model: object = model_factory(work_dir=work_dir)
    if not isinstance(model, Music2EmotionModel):
        raise TypeError("vendored Music2emo model does not implement predict()")
    return model


def _predict(model: Music2EmotionModel, request: object) -> dict[str, object]:
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")
    audio_path = request.get("audio_path")
    if not isinstance(audio_path, str) or len(audio_path) == 0:
        raise TypeError("request must contain non-empty audio_path")

    with contextlib.redirect_stdout(sys.stderr):
        result = model.predict(audio_path, threshold=0.5)
    scores = result.get("mood_scores")
    valence = result.get("valence")
    arousal = result.get("arousal")
    if not isinstance(scores, dict):
        raise TypeError("model output has no mood_scores mapping")
    if (
        not isinstance(valence, (int, float))
        or isinstance(valence, bool)
        or not isinstance(arousal, (int, float))
        or isinstance(arousal, bool)
    ):
        raise TypeError("model output has invalid valence/arousal")

    validated_scores: dict[str, float] = {}
    for tag, score in scores.items():
        if not isinstance(tag, str) or not isinstance(score, (int, float)) or isinstance(score, bool):
            raise TypeError("model output has invalid mood score")
        validated_scores[tag] = float(score)

    return {
        "mood_scores": validated_scores,
        "valence": float(valence),
        "arousal": float(arousal),
    }


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="sp-music2emotion-") as temporary_directory:
            model = _load_model(Path(temporary_directory))
            for line in sys.stdin:
                try:
                    request: object = json.loads(line)
                    response = _predict(model, request)
                except Exception as error:
                    response = {"error": str(error)}
                    traceback.print_exc(file=sys.stderr)
                print(json.dumps(response, allow_nan=False), flush=True)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
