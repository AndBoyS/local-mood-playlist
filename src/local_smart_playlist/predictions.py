"""Prediction records and persistent cache for local audio inference."""

import math
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

_CACHE_SCHEMA_VERSION = 1
_MIN_VA_SCORE = 1.0
_MAX_VA_SCORE = 9.0


@dataclass(frozen=True)
class TrackPrediction:
    """Emotion model output for one track.

    Mood scores should use canonical model tag names and probabilities in [0, 1].
    Valence and arousal use Music2Emotion's documented 1-9 scale.
    """

    mood_scores: dict[str, float]
    valence: float
    arousal: float

    def __post_init__(self) -> None:
        if len(self.mood_scores) == 0:
            raise ValueError("mood_scores must not be empty")
        for tag, score in self.mood_scores.items():
            if len(tag) == 0 or not _is_number(score) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("mood scores must have non-empty tags and finite values in [0, 1]")
        for name, value in (("valence", self.valence), ("arousal", self.arousal)):
            if (
                not _is_number(value)
                or not math.isfinite(value)
                or not _MIN_VA_SCORE <= value <= _MAX_VA_SCORE
            ):
                raise ValueError(f"{name} must be finite and in [{_MIN_VA_SCORE:g}, {_MAX_VA_SCORE:g}]")


@dataclass(frozen=True)
class _CacheEntry:
    size_bytes: int
    modified_ns: int
    prediction: TrackPrediction


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
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "model_version": self.model_version,
            "entries": {
                path: {
                    "size_bytes": entry.size_bytes,
                    "modified_ns": entry.modified_ns,
                    "mood_scores": entry.prediction.mood_scores,
                    "valence": entry.prediction.valence,
                    "arousal": entry.prediction.arousal,
                }
                for path, entry in self._entries.items()
            },
        }

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                pickle.dump(payload, temporary_file, protocol=pickle.HIGHEST_PROTOCOL)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _fingerprint(self, audio_path: Path) -> tuple[str, int, int] | None:
        try:
            canonical_path = audio_path.expanduser().resolve(strict=True)
            file_stat = canonical_path.stat()
        except FileNotFoundError:
            return None
        if not canonical_path.is_file():
            raise ValueError(f"audio path is not a file: {audio_path}")
        if canonical_path == self.path.resolve():
            raise ValueError("cache path must not point to an audio file")
        return str(canonical_path), file_stat.st_size, file_stat.st_mtime_ns

    def _load(self) -> dict[str, _CacheEntry]:
        if not self.path.is_file():
            return {}
        try:
            with self.path.open("rb") as cache_file:
                payload: object = pickle.load(cache_file)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError, ValueError, TypeError):
            return {}

        if not isinstance(payload, dict):
            return {}
        if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return {}
        if payload.get("model_version") != self.model_version:
            return {}
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            return {}

        entries: dict[str, _CacheEntry] = {}
        for raw_path, raw_entry in raw_entries.items():
            if not isinstance(raw_path, str) or not isinstance(raw_entry, dict):
                continue
            size_bytes = raw_entry.get("size_bytes")
            modified_ns = raw_entry.get("modified_ns")
            if not _is_nonnegative_int(size_bytes) or not _is_nonnegative_int(modified_ns):
                continue
            prediction = _decode_prediction(raw_entry)
            if prediction is not None:
                entries[raw_path] = _CacheEntry(size_bytes, modified_ns, prediction)
        return entries


def _decode_prediction(value: dict[object, object]) -> TrackPrediction | None:
    raw_scores = value.get("mood_scores")
    raw_valence = value.get("valence")
    raw_arousal = value.get("arousal")
    if not isinstance(raw_scores, dict):
        return None
    if not _is_number(raw_valence) or not _is_number(raw_arousal):
        return None

    scores: dict[str, float] = {}
    for tag, score in raw_scores.items():
        if not isinstance(tag, str) or not isinstance(score, (int, float)) or isinstance(score, bool):
            return None
        scores[tag] = float(score)
    try:
        return TrackPrediction(scores, float(raw_valence), float(raw_arousal))
    except ValueError:
        return None


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_nonnegative_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
