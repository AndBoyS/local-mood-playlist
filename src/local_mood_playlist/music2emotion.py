"""In-process adapter for vendored Music2Emotion inference."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

from local_mood_playlist.assets import decompress_mert_weights, ensure_bundles, mert_cache_dir
from local_mood_playlist.moods import MOOD_NAMES, MoodName
from local_mood_playlist.paths import ResolvedPath
from local_mood_playlist.predictions import TrackPrediction

MODEL_VERSION = "m2e-c5f7984-j95926289-btc11c6ed-tags-c53cee67-mert12af15fe-patch2-plainnames"

# Inference device: "auto" picks the first available of CUDA, MPS, then CPU
Device = Literal["auto", "cpu", "mps", "cuda"]

# Canonical mood tag format used by Music2Emotion (see vendor/music2emo.py,
# which loads inference/data/tag_list.npy).
TAG_PREFIX = "mood/theme---"
# Model tag list says "dream"; mood vocabulary says "dreamy".
CANONICAL_NAME_OVERRIDES = {"dream": "dreamy"}


class PreparedTrack(Protocol):
    """CPU-staged audio for one track (vendored ``Music2emo.prepare`` output).

    Structural contract only: the vendored module is excluded from type
    checking, so consumers treat staged tracks as opaque and hand them back to
    :meth:`Music2Emotion.predict_prepared`.
    """

    audio_path: str


class ModelOutput(TypedDict):
    """Raw inference output of the vendored model for one track."""

    mood_scores: dict[str, float]
    valence: float
    arousal: float


class _Music2EmotionModel(Protocol):
    """Inference seam over the vendored (untyped) ``Music2emo`` class."""

    def prepare(self, audio: str | Path) -> PreparedTrack: ...

    def infer(self, prepared: PreparedTrack, threshold: float = 0.5) -> ModelOutput: ...


def to_cli_moods(canonical_tag: str) -> MoodName:
    name = canonical_tag.removeprefix(TAG_PREFIX)
    name = CANONICAL_NAME_OVERRIDES.get(name, name)
    if name not in MOOD_NAMES:
        raise ValueError(f"unexpected mood tag from model: {canonical_tag}")
    return name


def _load_model(cache_root: ResolvedPath, *, device: Device) -> _Music2EmotionModel:
    bundles = ensure_bundles(cache_root)

    from local_mood_playlist.vendor.music2emo import Music2emo  # noqa: PLC0415

    with decompress_mert_weights(mert_cache_dir(cache_root)) as snapshot:
        model = Music2emo(
            model_weights=bundles["music2emo/J_all.safetensors"],
            chord_model_weights=bundles["music2emo/btc_model_large_voca.safetensors"],
            mert_snapshot=snapshot,
            device=device,
        )
    return cast("_Music2EmotionModel", model)


def _to_prediction(result: ModelOutput) -> TrackPrediction:
    mood_scores = {to_cli_moods(tag): float(score) for tag, score in result["mood_scores"].items()}
    return TrackPrediction.from_raw(
        {"mood_scores": mood_scores, "valence": result["valence"], "arousal": result["arousal"]}
    )


class Music2Emotion:
    """Music2Emotion model wrapper."""

    def __init__(self, cache_root: ResolvedPath, *, device: Device) -> None:
        self._model = _load_model(cache_root, device=device)

    def predict(self, audio_path: ResolvedPath) -> TrackPrediction:
        return self.predict_prepared(self.prepare(audio_path))

    def prepare(self, audio_path: ResolvedPath) -> PreparedTrack:
        """CPU-only staging (decode, resample, CQT); safe to run off the model thread."""
        return self._model.prepare(audio_path.path)

    def predict_prepared(self, prepared: PreparedTrack) -> TrackPrediction:
        """Inference for a track staged by :meth:`prepare`; runs on the model's device."""
        return _to_prediction(self._model.infer(prepared, threshold=0.5))

    @contextmanager
    def prepare_pipeline(
        self, audio_paths: Sequence[ResolvedPath]
    ) -> Generator[Iterator[tuple[ResolvedPath, PreparedTrack | Exception]]]:
        """Stage tracks one ahead so CPU staging overlaps model inference.

        Use as ``with model.prepare_pipeline(paths) as pipeline:`` and iterate
        ``pipeline``. Each item is ``(audio_path, staged_track)``, or an
        ``Exception`` raised by staging that track (yielded instead of raised
        so per-track failure handling stays with the consumer). Staging order
        matches ``audio_paths``.
        """
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moodpl-prepare")

        def stage(audio_path: ResolvedPath) -> PreparedTrack:
            return self.prepare(audio_path)

        def resolved(
            entry: tuple[ResolvedPath, Future[PreparedTrack]],
        ) -> tuple[ResolvedPath, PreparedTrack | Exception]:
            audio_path, future = entry
            try:
                return audio_path, future.result()
            except Exception as error:  # noqa: BLE001 - surfaced per track, not per run
                return audio_path, error

        def staged() -> Generator[tuple[ResolvedPath, PreparedTrack | Exception]]:
            try:
                upcoming: tuple[ResolvedPath, Future[PreparedTrack]] | None = None
                for audio_path in audio_paths:
                    current = (audio_path, executor.submit(stage, audio_path))
                    if upcoming is not None:
                        # `current` is already staging while the consumer works on `upcoming`.
                        yield resolved(upcoming)
                    upcoming = current
                if upcoming is not None:
                    yield resolved(upcoming)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        pipeline = staged()
        try:
            yield pipeline
        finally:
            pipeline.close()
