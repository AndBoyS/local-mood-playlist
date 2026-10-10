"""Index local audio files into the persistent Music2Emotion prediction cache."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from local_mood_playlist.const import CACHE_DIR
from local_mood_playlist.music2emotion import MODEL_VERSION, Music2Emotion
from local_mood_playlist.predictions import PredictionCache

_AUDIO_SUFFIXES = [".flac", ".m4a", ".mp3", ".wav"]
_SAVE_EVERY = 25
ProgressUpdate = tuple[int, int, Path | None]
ProgressCallback = Callable[[ProgressUpdate], None]


@dataclass(frozen=True)
class IndexFailure:
    audio_path: Path
    error: str


@dataclass(frozen=True)
class IndexSummary:
    tracks_found: int
    predictions_reused: int
    predictions_computed: int
    failures: tuple[IndexFailure, ...]
    cache_path: Path

    @property
    def predictions_failed(self) -> int:
        return len(self.failures)


def default_cache_path(library_path: Path) -> Path:
    """Return workspace-cache path unique to this resolved library directory."""
    resolved_library = library_path.expanduser().resolve(strict=True)
    library_id = hashlib.sha256(str(resolved_library).encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / "predictions" / f"predictions-{library_id}.tsv"


def index_library(
    library_path: Path,
    *,
    cache_path: Path | None = None,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> IndexSummary:
    """Predict uncached or changed audio files; store raw scores for later ranking."""
    root = library_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"music library path is not a directory: {root}")
    audio_files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES)
    if len(audio_files) == 0:
        raise ValueError(f"no supported audio files found in {root}")

    actual_cache_path = cache_path.expanduser() if cache_path is not None else default_cache_path(root)
    cache = PredictionCache(actual_cache_path, model_version=MODEL_VERSION)
    pending: list[Path] = []
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
            model = Music2Emotion()
            for index, audio_file in enumerate(pending, start=1):
                try:
                    prediction = model.predict(audio_file)
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
        cache_path=actual_cache_path,
    )
