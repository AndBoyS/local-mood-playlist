"""Index local audio files into the persistent Music2Emotion prediction cache."""

from collections.abc import Callable
from dataclasses import dataclass

from local_mood_playlist.music2emotion import MODEL_VERSION, Device, Music2Emotion
from local_mood_playlist.paths import ResolvedPath
from local_mood_playlist.predictions import PredictionCache

AUDIO_SUFFIXES: tuple[str, ...] = (".flac", ".m4a", ".mp3", ".wav")
_SAVE_EVERY = 25
ProgressUpdate = tuple[int, int, ResolvedPath | None]
ProgressCallback = Callable[[ProgressUpdate], None]


@dataclass(frozen=True)
class IndexFailure:
    audio_path: ResolvedPath
    error: str


@dataclass(frozen=True)
class IndexSummary:
    tracks_found: int
    predictions_reused: int
    predictions_computed: int
    failures: tuple[IndexFailure, ...]
    cache_path: ResolvedPath

    @property
    def predictions_failed(self) -> int:
        return len(self.failures)


def index_library(
    library_path: ResolvedPath,
    *,
    cache_path: ResolvedPath,
    cache_root: ResolvedPath,
    device: Device = "auto",
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> IndexSummary:
    """Predict uncached or changed audio files; store raw scores for later ranking.

    Raises ValueError when ``library_path`` is not a directory.
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
    pending: list[ResolvedPath] = []
    reused = 0
    for audio_file in audio_files:
        if not force and cache.get(audio_file) is not None:
            reused += 1
        else:
            pending.append(audio_file)

    computed = 0
    failures: list[IndexFailure] = []
    if len(pending) > 0:
        if on_progress is not None:
            on_progress((0, len(pending), None))
        try:
            model = Music2Emotion(cache_root, device=device)
            with model.prepare_pipeline(pending) as pipeline:
                for index, (audio_file, prepared) in enumerate(pipeline, start=1):
                    if isinstance(prepared, Exception):
                        failures.append(IndexFailure(audio_file, str(prepared)))
                    else:
                        try:
                            prediction = model.predict_prepared(prepared)
                        except Exception as error:
                            failures.append(IndexFailure(audio_file, str(error)))
                        else:
                            cache.put(audio_file, prediction)
                            computed += 1
                    if on_progress is not None:
                        on_progress((index, len(pending), audio_file))
                    if index % _SAVE_EVERY == 0:
                        cache.save()
        except Exception:
            if computed > 0:
                cache.save()
            raise
        cache.save()

    return IndexSummary(
        tracks_found=len(audio_files),
        predictions_reused=reused,
        predictions_computed=computed,
        failures=tuple(failures),
        cache_path=cache_path,
    )
