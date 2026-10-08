"""Index local audio files into the persistent Music2Emotion prediction cache."""

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from local_smart_playlist.music2emotion import MODEL_VERSION, Music2EmotionWorker
from local_smart_playlist.predictions import PredictionCache

_AUDIO_SUFFIXES = frozenset({".flac", ".m4a", ".mp3", ".wav"})
_SAVE_EVERY = 25


@dataclass(frozen=True)
class IndexSummary:
    tracks_found: int
    predictions_reused: int
    predictions_computed: int
    cache_path: Path


def default_cache_path(library_path: Path) -> Path:
    """Return user-cache path unique to this resolved library directory."""
    resolved_library = library_path.expanduser().resolve(strict=True)
    library_id = hashlib.sha256(str(resolved_library).encode("utf-8")).hexdigest()[:20]
    if sys.platform == "darwin":
        cache_root = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Cache"
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "local-smart-playlist" / f"predictions-{library_id}.pkl"


def index_library(
    library_path: Path,
    *,
    cache_path: Path | None = None,
    force: bool = False,
) -> IndexSummary:
    """Predict uncached or changed audio files; store raw scores for later ranking."""
    root = library_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"music library path is not a directory: {root}")
    audio_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES
    )
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

    if len(pending) > 0:
        computed = 0
        try:
            with Music2EmotionWorker() as worker:
                for index, audio_file in enumerate(pending, start=1):
                    cache.put(audio_file, worker.predict(audio_file))
                    computed += 1
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
        predictions_computed=len(pending),
        cache_path=actual_cache_path,
    )
