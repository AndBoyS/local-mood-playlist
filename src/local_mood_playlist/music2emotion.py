"""In-process adapter for vendored Music2Emotion inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from local_mood_playlist.assets import decompress_mert_weights, ensure_bundles, mert_cache_dir
from local_mood_playlist.predictions import TrackPrediction

if TYPE_CHECKING:
    from local_mood_playlist.vendor.music2emo import Music2emo

MODEL_VERSION = "m2e-c5f7984-j95926289-btc11c6ed-tags-c53cee67-mert12af15fe-patch2"


def _load_model() -> Music2emo:
    bundles = ensure_bundles()

    from local_mood_playlist.vendor.music2emo import Music2emo  # noqa: PLC0415

    with decompress_mert_weights(mert_cache_dir()) as snapshot:
        return Music2emo(
            model_weights=bundles["music2emo/J_all.safetensors"],
            chord_model_weights=bundles["music2emo/btc_model_large_voca.safetensors"],
            mert_snapshot=snapshot,
        )


class Music2Emotion:
    """Load one MERT/Music2Emotion model in current Python process."""

    def __init__(self) -> None:
        self._model = _load_model()

    def predict(self, audio_path: Path) -> TrackPrediction:
        result = self._model.predict(audio_path.expanduser().resolve(strict=True), threshold=0.5)
        return TrackPrediction(result["mood_scores"], result["valence"], result["arousal"])
