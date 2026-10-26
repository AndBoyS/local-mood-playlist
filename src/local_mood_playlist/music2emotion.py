"""In-process adapter for vendored Music2Emotion inference."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from local_mood_playlist.assets import decompress_mert_weights, get_asset_paths, mert_cache_dir
from local_mood_playlist.moods import MOOD_NAMES, MoodName
from local_mood_playlist.paths import ResolvedPath
from local_mood_playlist.predictions import TrackPrediction

if TYPE_CHECKING:
    from local_mood_playlist.vendor.music2emo import Music2emo, PreparedAudio

MODEL_VERSION = "m2e-c5f7984-j95926289-btc11c6ed-tags-c53cee67-mert12af15fe-patch2-plainnames"

# Inference device: "auto" picks the first available of CUDA, MPS, then CPU
Device = Literal["auto", "cpu", "mps", "cuda"]

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


def _load_model(cache_root: ResolvedPath, *, device: Device) -> Music2emo:
    from local_mood_playlist.vendor.music2emo import Music2emo  # noqa: PLC0415

    asset_paths = get_asset_paths(cache_root)

    with decompress_mert_weights(mert_cache_dir(cache_root)) as snapshot:
        return Music2emo(
            model_weights=asset_paths["music2emo/J_all.safetensors"],
            chord_model_weights=asset_paths["music2emo/btc_model_large_voca.safetensors"],
            mert_snapshot=snapshot,
            device=device,
        )


class Music2Emotion:
    """Music2Emotion model wrapper."""

    def __init__(self, cache_root: ResolvedPath, *, device: Device) -> None:
        self._model = _load_model(cache_root, device=device)

    def preprocess(self, audio_path: ResolvedPath) -> PreparedAudio:
        """CPU-only preprocessing (decode, resample, CQT); safe to run off the model thread."""
        return self._model.prepare(audio_path.path)

    def predict_preprocessed(self, preprocessed: PreparedAudio) -> TrackPrediction:
        """Prediction for a track from :meth:`preprocess`; runs on the model's device."""
        result = self._model.infer(preprocessed)
        mood_scores = {to_cli_moods(tag): score for tag, score in result["mood_scores"].items()}
        return TrackPrediction.from_raw(
            {"mood_scores": mood_scores, "valence": result["valence"], "arousal": result["arousal"]}
        )

    @contextmanager
    def preprocess_pipeline(
        self, audio_paths: Sequence[ResolvedPath]
    ) -> Generator[Iterator[tuple[ResolvedPath, PreparedAudio | Exception]]]:
        """Preprocess tracks one ahead so CPU preprocessing overlaps prediction."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moodpl-preprocess")

        def resolved(
            entry: tuple[ResolvedPath, Future[PreparedAudio]],
        ) -> tuple[ResolvedPath, PreparedAudio | Exception]:
            audio_path, future = entry
            try:
                return audio_path, future.result()
            except Exception as error:  # noqa: BLE001 - surfaced per track, not per run
                return audio_path, error

        def preprocessed() -> Generator[tuple[ResolvedPath, PreparedAudio | Exception]]:
            try:
                upcoming: tuple[ResolvedPath, Future[PreparedAudio]] | None = None
                for audio_path in audio_paths:
                    current = (audio_path, executor.submit(self.preprocess, audio_path))
                    if upcoming is not None:
                        # `current` is already preprocessing while the consumer
                        # works on `upcoming`.
                        yield resolved(upcoming)
                    upcoming = current
                if upcoming is not None:
                    yield resolved(upcoming)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        pipeline = preprocessed()
        try:
            yield pipeline
        finally:
            pipeline.close()
