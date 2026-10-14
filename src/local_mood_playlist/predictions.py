"""Prediction records and persistent cache for local audio inference.

:class:`TrackPrediction` is a msgspec ``Struct``: the type annotations are
validated at the model-output boundary via :func:`TrackPrediction.from_raw`
(mood keys must be known mood names, scores must be finite floats in [0, 1]),
so consumers never re-check them.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Annotated, TypedDict

import msgspec

from local_mood_playlist.filesystem_utils import open_atomic
from local_mood_playlist.moods import MoodName
from local_mood_playlist.paths import ResolvedPath

_CACHE_SCHEMA_VERSION = 2

MoodScore = Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]


class TrackPrediction(msgspec.Struct, frozen=True):
    """Emotion model output for one track.

    Mood scores use plain mood names (Mood vocabulary values) and probabilities
    in [0, 1]; construct via :meth:`from_raw` to enforce this. Valence and
    arousal are informational model outputs on Music2Emotion's 1-9 scale; they
    are stored but never used for ranking, so they are not range-validated.
    Nothing in this repository consumes them yet.
    """

    mood_scores: dict[MoodName, MoodScore]
    valence: float
    arousal: float

    def __post_init__(self) -> None:
        if len(self.mood_scores) == 0:
            raise ValueError("mood_scores must not be empty")

    @classmethod
    def from_raw(cls, raw: Mapping[str, object]) -> TrackPrediction:
        return msgspec.convert(raw, cls)


@dataclass(frozen=True)
class _CacheEntry:
    size_bytes: int
    modified_ns: int
    prediction: TrackPrediction


class _CachePayload(TypedDict):
    schema_version: int
    model_version: str
    entries: dict[ResolvedPath, _CacheEntry]


class PredictionCache:
    """Pickle-backed cache keyed by canonical path and audio-file metadata.

    Audio keys are :class:`ResolvedPath`, so identity is stable across skips,
    symlinks, and trailing ``.`` — no caller can slip in a non-canonical key.
    Only load cache files created and controlled by this application: pickle is
    executable serialization and must not be used with untrusted files.
    """

    def __init__(self, path: ResolvedPath, model_version: str) -> None:
        if len(model_version.strip()) == 0:
            raise ValueError("model_version must not be empty")
        self.path = path
        self.model_version = model_version
        self._entries = self._load()

    def get(self, audio_path: ResolvedPath) -> TrackPrediction | None:
        """Return fresh prediction for audio_path, or None on cache miss/staleness."""
        fingerprint = self._fingerprint(audio_path)
        if fingerprint is None:
            return None
        size_bytes, modified_ns = fingerprint
        entry = self._entries.get(audio_path)
        if entry is None or entry.size_bytes != size_bytes or entry.modified_ns != modified_ns:
            return None
        return entry.prediction

    def iter_predictions(self) -> Iterator[tuple[ResolvedPath, TrackPrediction]]:
        """Yield fresh cached predictions with canonical audio paths."""
        for audio_path in self._entries:
            prediction = self.get(audio_path)
            if prediction is not None:
                yield audio_path, prediction

    def put(self, audio_path: ResolvedPath, prediction: TrackPrediction) -> None:
        """Remember prediction for current audio-file version; call save() to persist."""
        fingerprint = self._fingerprint(audio_path)
        if fingerprint is None:
            raise FileNotFoundError(audio_path.path)
        size_bytes, modified_ns = fingerprint
        self._entries[audio_path] = _CacheEntry(size_bytes, modified_ns, prediction)

    def save(self) -> None:
        """Atomically write cache; batch put() calls before saving to avoid repeated I/O."""
        payload: _CachePayload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "model_version": self.model_version,
            "entries": self._entries,
        }

        with open_atomic(self.path.path) as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _fingerprint(self, audio_path: ResolvedPath) -> tuple[int, int] | None:
        """Return (size, mtime) for the canonical audio path, or None if it vanished."""
        try:
            file_stat = audio_path.path.stat()
        except FileNotFoundError:
            return None
        if not audio_path.path.is_file():
            raise ValueError(f"audio path is not a file: {audio_path}")
        if audio_path == self.path:
            raise ValueError("cache path must not point to an audio file")
        return file_stat.st_size, file_stat.st_mtime_ns

    def _load(self) -> dict[ResolvedPath, _CacheEntry]:
        if not self.path.path.is_file():
            return {}
        try:
            with self.path.path.open("rb") as cache_file:
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
