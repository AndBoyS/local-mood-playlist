"""Prediction records and persistent cache for local audio inference."""

import csv
import io
import math
import os
import tempfile
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from local_mood_playlist.moods import MOOD_NAMES, MoodName

_CACHE_SCHEMA_VERSION = 1
_META_PREFIX = "#moodpl-predictions"
_SCHEMA_FIELD = f"schema={_CACHE_SCHEMA_VERSION}"
_MODEL_VERSION_FIELD = "model_version="
_META_COLUMN_COUNT = 3
_HEADER_PREFIX = ("path", "size_bytes", "modified_ns", "valence", "arousal")


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


class PredictionCache:
    """TSV-backed cache keyed by canonical path and audio-file metadata.

    Layout: one metadata row, one header row naming the mood-tag columns, then
    one row per track. An empty score cell means the tag is absent for that
    track. Only load cache files created and controlled by this application:
    cache files must never come from an untrusted source.
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
        for raw_path in self._entries:
            audio_path = Path(raw_path)
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
        payload = self._encode_tsv()

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                bytes_written = temporary_file.write(payload)
                if bytes_written != len(payload):
                    raise OSError(f"short cache write: {bytes_written} of {len(payload)} bytes")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _encode_tsv(self) -> str:
        """Serialize all entries; tag columns are the union across entries, sorted."""
        tags = sorted({tag for entry in self._entries.values() for tag in entry.prediction.mood_scores})
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            (
                _META_PREFIX,
                _SCHEMA_FIELD,
                _MODEL_VERSION_FIELD + urllib.parse.quote(self.model_version, safe=""),
            )
        )
        writer.writerow(_HEADER_PREFIX + tuple(tags))
        for canonical_path, entry in self._entries.items():
            scores = entry.prediction.mood_scores
            writer.writerow(
                (
                    canonical_path,
                    entry.size_bytes,
                    entry.modified_ns,
                    repr(entry.prediction.valence),
                    repr(entry.prediction.arousal),
                )
                + tuple(repr(scores[tag]) if tag in scores else "" for tag in tags)
            )
        return buffer.getvalue()

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
            with self.path.open("r", encoding="utf-8", newline="") as cache_file:
                return self._decode_tsv(cache_file)
        except (OSError, csv.Error, UnicodeDecodeError, EOFError):
            return {}

    def _decode_tsv(self, cache_file: TextIO) -> dict[str, _CacheEntry]:
        reader = csv.reader(cache_file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        meta = next(reader, None)
        header = next(reader, None)
        if meta is None or len(meta) != _META_COLUMN_COUNT or meta[0] != _META_PREFIX:
            return {}
        if meta[1] != _SCHEMA_FIELD or not meta[2].startswith(_MODEL_VERSION_FIELD):
            return {}
        model_version = urllib.parse.unquote(meta[2][len(_MODEL_VERSION_FIELD) :])
        if model_version != self.model_version:
            return {}
        if header is None or len(header) < len(_HEADER_PREFIX):
            return {}
        if tuple(header[: len(_HEADER_PREFIX)]) != _HEADER_PREFIX:
            return {}
        tags = tuple(header[len(_HEADER_PREFIX) :])
        unknown_tags = [tag for tag in tags if tag not in MOOD_NAMES]
        if len(unknown_tags) > 0:
            raise ValueError(
                f"prediction cache {self.path} contains unknown mood tag column(s) "
                f"{', '.join(unknown_tags)}; delete it and re-index the library"
            )

        entries: dict[str, _CacheEntry] = {}
        for row in reader:
            if len(row) == 0:
                continue
            entry = _decode_row(row, tags)
            if entry is not None:
                entries[row[0]] = entry
        return entries


def _decode_row(row: list[str], tags: tuple[str, ...]) -> _CacheEntry | None:
    expected_columns = len(_HEADER_PREFIX) + len(tags)
    if len(row) != expected_columns:
        return None
    size_bytes = _parse_nonnegative_int(row[1])
    modified_ns = _parse_nonnegative_int(row[2])
    if size_bytes is None or modified_ns is None:
        return None
    try:
        valence = float(row[3])
        arousal = float(row[4])
    except ValueError:
        return None
    scores: dict[str, float] = {}
    for tag, raw_score in zip(tags, row[len(_HEADER_PREFIX) :], strict=True):
        if raw_score == "":
            continue
        try:
            scores[tag] = float(raw_score)
        except ValueError:
            return None
    try:
        prediction = TrackPrediction(cast("dict[MoodName, float]", scores), valence, arousal)
    except ValueError:
        return None
    return _CacheEntry(size_bytes, modified_ns, prediction)


def _parse_nonnegative_int(value: str) -> int | None:
    if len(value) == 0 or not (value.isascii() and value.isdigit()):
        return None
    return int(value)
