"""Prediction records and persistent cache for local audio inference."""

import math
import os
import pickle
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from local_mood_playlist.moods import MOOD_NAMES, MoodName

_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrackPrediction:
    """Emotion model output for one track.

    Mood scores should use plain mood names (Mood vocabulary values) and
    probabilities in [0, 1].
    Valence and arousal are informational model outputs on Music2Emotion's
    1-9 scale; they are stored but never used for ranking, so they are not
    range-validated. Nothing in this repository consumes them yet.
    """

    mood_scores: dict[MoodName, float]
    valence: float
    arousal: float

    def __post_init__(self) -> None:
        if len(self.mood_scores) == 0:
            raise ValueError("mood_scores must not be empty")
        for tag, score in self.mood_scores.items():
            if tag not in MOOD_NAMES or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("mood scores must use known mood names and finite values in [0, 1]")


@dataclass(frozen=True)
class _CacheEntry:
    size_bytes: int
    modified_ns: int
    prediction: TrackPrediction


class _CachePayload(TypedDict):
    schema_version: int
    model_version: str
    entries: dict[Path, _CacheEntry]


class PredictionCache:
    """Pickle-backed cache keyed by canonical path and audio-file metadata.

    Only load cache files created and controlled by this application: pickle is
    executable serialization and must not be used with untrusted files.
    """

    def __init__(self, path: Path, model_version: str) -> None:
        if len(model_version.strip()) == 0:
            raise ValueError("model_version must not be empty")
        self.path = path.expanduser()
        self.model_version = model_version
        self._entries = self._load()

    def get(self, audio_path: Path) -> TrackPrediction | None:
        """Return fresh prediction for audio_path, or None on cache miss/staleness."""
        fingerprint = self._fingerprint(audio_path)
        if fingerprint is None:
            return None
        canonical_path, size_bytes, modified_ns = fingerprint
        entry = self._entries.get(canonical_path)
        if entry is None or entry.size_bytes != size_bytes or entry.modified_ns != modified_ns:
            return None
        return entry.prediction

    def iter_predictions(self) -> Iterator[tuple[Path, TrackPrediction]]:
        """Yield fresh cached predictions with canonical audio paths."""
        for audio_path in self._entries:
            prediction = self.get(audio_path)
            if prediction is not None:
                yield audio_path, prediction

    def put(self, audio_path: Path, prediction: TrackPrediction) -> None:
        """Remember prediction for current audio-file version; call save() to persist."""
        fingerprint = self._fingerprint(audio_path)
        if fingerprint is None:
            raise FileNotFoundError(audio_path)
        canonical_path, size_bytes, modified_ns = fingerprint
        self._entries[canonical_path] = _CacheEntry(size_bytes, modified_ns, prediction)

    def save(self) -> None:
        """Atomically write cache; batch put() calls before saving to avoid repeated I/O."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: _CachePayload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "model_version": self.model_version,
            "entries": self._entries,
        }

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                temp_path = Path(f.name)
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            _ = temp_path.replace(self.path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _fingerprint(self, audio_path: Path) -> tuple[Path, int, int] | None:
        try:
            canonical_path = audio_path.expanduser().resolve(strict=True)
            file_stat = canonical_path.stat()
        except FileNotFoundError:
            return None
        if not canonical_path.is_file():
            raise ValueError(f"audio path is not a file: {audio_path}")
        if canonical_path == self.path.resolve():
            raise ValueError("cache path must not point to an audio file")
        return canonical_path, file_stat.st_size, file_stat.st_mtime_ns

    def _load(self) -> dict[Path, _CacheEntry]:
        if not self.path.is_file():
            return {}
        try:
            with self.path.open("rb") as cache_file:
                payload: _CachePayload = pickle.load(cache_file)
            if payload["schema_version"] != _CACHE_SCHEMA_VERSION:
                return {}
            if payload["model_version"] != self.model_version:
                return {}
            return payload["entries"]
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            AttributeError,
            ImportError,
            ValueError,
            TypeError,
            KeyError,
        ):
            return {}
