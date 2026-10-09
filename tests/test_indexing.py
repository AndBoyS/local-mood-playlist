"""Tests for indexing progress and cache reuse."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_smart_playlist.indexing import ProgressUpdate, index_library
from local_smart_playlist.predictions import TrackPrediction


class _FakeMusic2Emotion:
    def predict(self, audio_path: Path) -> TrackPrediction:
        if audio_path.name == "02-bad.mp3":
            raise RuntimeError("audio is too short for model")
        return TrackPrediction({"mood/theme---calm": 0.8}, 6.0, 3.0)


class IndexingTests(unittest.TestCase):
    def test_reports_progress_and_reuses_cached_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio_path = root / "track.mp3"
            cache_path = root / "predictions.tsv"
            _ = audio_path.write_bytes(b"audio")
            progress: list[tuple[int, int, Path | None]] = []

            def report_progress(update: ProgressUpdate) -> None:
                progress.append(update)

            with patch("local_smart_playlist.indexing.Music2Emotion", _FakeMusic2Emotion):
                first = index_library(root, cache_path=cache_path, on_progress=report_progress)
                second = index_library(root, cache_path=cache_path, on_progress=report_progress)

            assert first.predictions_computed == 1
            assert second.predictions_reused == 1
            assert second.predictions_computed == 0
            assert progress == [(0, 1, None), (1, 1, audio_path.resolve())]
            assert second.cache_path == cache_path

    def test_track_failure_is_recorded_and_does_not_abort_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            good_path = root / "01-good.mp3"
            bad_path = root / "02-bad.mp3"
            cache_path = root / "predictions.tsv"
            _ = good_path.write_bytes(b"audio")
            _ = bad_path.write_bytes(b"audio")
            progress: list[ProgressUpdate] = []

            with patch("local_smart_playlist.indexing.Music2Emotion", _FakeMusic2Emotion):
                summary = index_library(
                    root,
                    cache_path=cache_path,
                    on_progress=progress.append,
                )

            assert summary.predictions_computed == 1
            assert summary.predictions_failed == 1
            assert summary.failures[0].audio_path == bad_path.resolve()
            assert summary.failures[0].error == "audio is too short for model"
            assert progress == [
                (0, 2, None),
                (1, 2, good_path.resolve()),
                (2, 2, bad_path.resolve()),
            ]


if __name__ == "__main__":
    _ = unittest.main()
