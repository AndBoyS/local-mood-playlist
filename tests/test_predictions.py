"""Focused tests for prediction-cache persistence and invalidation."""

import tempfile
import unittest
from pathlib import Path

from local_smart_playlist.predictions import PredictionCache, TrackPrediction


class PredictionCacheTests(unittest.TestCase):
    def test_round_trip_and_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio_path = root / "track.mp3"
            cache_path = root / "predictions.tsv"
            _ = audio_path.write_bytes(b"audio")
            prediction = TrackPrediction({"mood/theme---calm": 0.8}, 6.0, 3.0)

            cache = PredictionCache(cache_path, model_version="test-model-v1")
            cache.put(audio_path, prediction)
            cache.save()

            loaded = PredictionCache(cache_path, model_version="test-model-v1")
            assert loaded.get(audio_path) == prediction
            assert PredictionCache(cache_path, model_version="test-model-v2").get(audio_path) is None

            _ = audio_path.write_bytes(b"changed audio")
            assert loaded.get(audio_path) is None

    def test_round_trip_special_characters_in_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            weird_name = "05 - I Swallowed Hard, Like \"I\" Under\tstood.mp3"
            audio_path = root / weird_name
            cache_path = root / "predictions.tsv"
            _ = audio_path.write_bytes(b"audio")
            prediction = TrackPrediction({"mood/theme---calm": 0.8}, 6.0, 3.0)

            cache = PredictionCache(cache_path, model_version="test-model-v1")
            cache.put(audio_path, prediction)
            cache.save()

            loaded = PredictionCache(cache_path, model_version="test-model-v1")
            assert loaded.get(audio_path) == prediction

    def test_missing_tags_are_preserved_per_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_path = root / "first.mp3"
            second_path = root / "second.mp3"
            cache_path = root / "predictions.tsv"
            _ = first_path.write_bytes(b"a")
            _ = second_path.write_bytes(b"b")
            first = TrackPrediction({"mood/theme---calm": 0.8}, 6.0, 3.0)
            second = TrackPrediction({"mood/theme---calm": 0.5, "mood/theme---dark": 0.9}, 5.0, 4.0)

            cache = PredictionCache(cache_path, model_version="test-model-v1")
            cache.put(first_path, first)
            cache.put(second_path, second)
            cache.save()

            loaded = PredictionCache(cache_path, model_version="test-model-v1")
            assert loaded.get(first_path) == first
            assert loaded.get(second_path) == second

    def test_malformed_cache_file_loads_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_path = root / "predictions.tsv"
            _ = cache_path.write_text("#moodpl-predictions\tschema=1\tmodel_version=nope\n")
            assert PredictionCache(cache_path, model_version="test-model-v1")._entries == {}


if __name__ == "__main__":
    _ = unittest.main()
