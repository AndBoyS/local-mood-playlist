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
            cache_path = root / "predictions.pkl"
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


if __name__ == "__main__":
    _ = unittest.main()
