import pytest
from cli import IndexedLibrary

from local_mood_playlist.music2emotion import MODEL_VERSION
from local_mood_playlist.paths import ResolvedPath
from local_mood_playlist.predictions import PredictionCache

pytestmark = [pytest.mark.model]

VALENCE_AROUSAL_MAX = 10.0


def test_cached_predictions_match_the_pinned_structure(indexed_library: IndexedLibrary) -> None:
    cache = PredictionCache(ResolvedPath(indexed_library.music_cache), MODEL_VERSION)
    predictions = dict(cache.iter_predictions())

    assert set(predictions) == {ResolvedPath(track) for track in indexed_library.tracks}
    for prediction in predictions.values():
        assert len(prediction.mood_scores) > 0
        assert all(0.0 <= float(score) <= 1.0 for score in prediction.mood_scores.values())
        assert 0.0 <= prediction.valence <= VALENCE_AROUSAL_MAX
        assert 0.0 <= prediction.arousal <= VALENCE_AROUSAL_MAX
