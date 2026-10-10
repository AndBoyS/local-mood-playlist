"""In-process adapter for vendored Music2Emotion inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from local_mood_playlist.assets import MERT_CACHE_DIR, decompress_mert_weights, ensure_bundles
from local_mood_playlist.moods import MOOD_NAMES, MoodName
from local_mood_playlist.predictions import TrackPrediction

if TYPE_CHECKING:
    from local_mood_playlist.vendor.music2emo import Music2emo

MODEL_VERSION = "m2e-c5f7984-j95926289-btc11c6ed-tags-c53cee67-mert12af15fe-patch2-plainnames"

# Canonical mood tag format used by Music2Emotion (see vendor/music2emo.py,
# which loads inference/data/tag_list.npy).
TAG_PREFIX = "mood/theme---"
# Model tag list says "dream"; mood vocabulary says "dreamy".
CANONICAL_NAME_OVERRIDES = {"dream": "dreamy"}


def to_cli_moods(canonical_tag: str) -> MoodName:
    name = canonical_tag.removeprefix(TAG_PREFIX)
    name = CANONICAL_NAME_OVERRIDES.get(name, name)
    if name not in MOOD_NAMES:
        raise ValueError(f"unexpected mood tag from model: {canonical_tag}")
    return name


def _load_model() -> Music2emo:
    bundles = ensure_bundles()

    from local_mood_playlist.vendor.music2emo import Music2emo  # noqa: PLC0415

    with decompress_mert_weights(MERT_CACHE_DIR) as snapshot:
        return Music2emo(
            model_weights=bundles["music2emo/J_all.safetensors"],
            chord_model_weights=bundles["music2emo/btc_model_large_voca.safetensors"],
            mert_snapshot=snapshot,
        )


class Music2Emotion:
    """Music2Emotion model wrapper."""

    def __init__(self) -> None:
        self._model = _load_model()

    def predict(self, audio_path: Path) -> TrackPrediction:
        result = self._model.predict(audio_path.expanduser().resolve(strict=True), threshold=0.5)
        mood_scores = {to_cli_moods(tag): score for tag, score in result["mood_scores"].items()}
        return TrackPrediction(mood_scores, result["valence"], result["arousal"])
