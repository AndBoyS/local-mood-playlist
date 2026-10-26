"""Index local audio files into the persistent Music2Emotion prediction cache."""

from collections.abc import Iterator
from dataclasses import dataclass

from local_mood_playlist.music2emotion import MODEL_VERSION, Device, Music2Emotion
from local_mood_playlist.paths import ResolvedPath
from local_mood_playlist.predictions import PredictionCache

AUDIO_SUFFIXES: tuple[str, ...] = (".flac", ".m4a", ".mp3", ".wav")
_SAVE_EVERY = 25


@dataclass(frozen=True)
class IndexFailure:
    audio_path: ResolvedPath
    error: str


@dataclass(frozen=True, slots=True)
class IndexProgress:
    processed: int
    to_process: int
    path_being_predicted: ResolvedPath


@dataclass(frozen=True, slots=True)
class IndexSummary:
    tracks_found: int
    predictions_reused: int
    predictions_computed: int
    failures: tuple[IndexFailure, ...]


def index_library(
    library_path: ResolvedPath,
    *,
    cache_path: ResolvedPath,
    cache_root: ResolvedPath,
    device: Device = "auto",
    force: bool = False,
) -> Iterator[IndexProgress | IndexSummary]:
    """Predict uncached or changed audio files; store raw scores for later ranking.

    Validates the library, scans it, and partitions against the cache
    eagerly, so callers can fail fast before expensive work; the returned
    iterator runs the model. Raises ValueError when ``library_path`` is not
    a directory or contains no supported audio files.

    Yields one :class:`IndexProgress` as each track enters prediction (tracks
    failing preprocessing yield no frame), then one final
    :class:`IndexSummary` with the run totals; an all-cached library yields
    only the summary. Interruptions still save whatever was computed.
    """
    if not library_path.path.is_dir():
        raise ValueError(f"music library path is not a directory: {library_path}")
    audio_files = sorted(
        set(
            ResolvedPath(path)
            for path in library_path.path.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )
    )
    if len(audio_files) == 0:
        raise ValueError(f"no supported audio files found in {library_path}")

    cache = PredictionCache(cache_path, model_version=MODEL_VERSION)
    reused = 0
    pending: list[ResolvedPath] = []
    for audio_file in audio_files:
        if not force and cache.get(audio_file) is not None:
            reused += 1
        else:
            pending.append(audio_file)

    return _index_pending(
        pending,
        cache,
        tracks_found=len(audio_files),
        predictions_reused=reused,
        cache_root=cache_root,
        device=device,
    )


def _index_pending(
    pending: list[ResolvedPath],
    cache: PredictionCache,
    *,
    tracks_found: int,
    predictions_reused: int,
    cache_root: ResolvedPath,
    device: Device,
) -> Iterator[IndexProgress | IndexSummary]:
    """Predict pending tracks into the cache; yield progress frames, then the summary."""
    failures: list[IndexFailure] = []
    computed = 0
    if len(pending) > 0:
        to_process = len(pending)
        try:
            model = Music2Emotion(cache_root, device=device)
            with model.preprocess_pipeline(pending) as pipeline:
                for index, (audio_file, preprocessed) in enumerate(pipeline, start=1):
                    if isinstance(preprocessed, Exception):
                        failures.append(IndexFailure(audio_file, str(preprocessed)))
                    else:
                        yield IndexProgress(processed=index - 1, to_process=to_process, path_being_predicted=audio_file)
                        try:
                            prediction = model.predict_preprocessed(preprocessed)
                        except Exception as error:
                            failures.append(IndexFailure(audio_file, str(error)))
                        else:
                            cache.put(audio_file, prediction)
                            computed += 1
                    if index % _SAVE_EVERY == 0:
                        cache.save()
        finally:
            if computed > 0:
                cache.save()

    yield IndexSummary(
        tracks_found=tracks_found,
        predictions_reused=predictions_reused,
        predictions_computed=computed,
        failures=tuple(failures),
    )
