"""Text-query component ranking tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from local_smart_playlist.index.store import TrackData
from local_smart_playlist.query import rank_text
from local_smart_playlist.query.axis_prompts import QueryAxis


@pytest.mark.parametrize("use_decomposed_components", [False, True])
def test_rank_text_components_splits_commas_and_uses_minimum(
    monkeypatch: pytest.MonkeyPatch,
    use_decomposed_components: bool,
) -> None:
    track_a = TrackData(rel_path="a.mp3", title="a", duration=60.0)
    track_b = TrackData(rel_path="b.mp3", title="b", duration=60.0)
    track_one_term_only = TrackData(rel_path="only-ethereal.mp3", title="only", duration=60.0)
    ethereal_vec = np.asarray([1.0], dtype=np.float32)
    dreamy_vec = np.asarray([2.0], dtype=np.float32)
    axis_bank = np.asarray([[1.0]], dtype=np.float32)
    rankings = [
        [(track_a, 0.91), (track_b, 0.94), (track_one_term_only, 0.93)],
        [(track_a, 0.97), (track_b, 0.82)],
    ]
    embed = MagicMock(side_effect=[ethereal_vec, dreamy_vec])
    rank = MagicMock(side_effect=rankings)
    store = MagicMock()
    store.track_count.return_value = 3
    monkeypatch.setattr(rank_text, "query_vector_contrast", embed)
    monkeypatch.setattr(rank_text, "rank_by_upper_cluster", rank)

    result = rank_text.rank_text_components(
        store,
        MagicMock(),
        query="ethereal, , dreamy",
        components={QueryAxis.MOOD: "ethereal, , dreamy"} if use_decomposed_components else None,
        axis_vocab_vecs={QueryAxis.MOOD: axis_bank},
        min_score=0.3,
        k=3,
    )

    assert [call.args[0] for call in embed.call_args_list] == ["ethereal mood.", "dreamy mood."]
    assert rank.call_count == 2
    assert [track.rel_path for track, _score in result] == ["a.mp3", "b.mp3"]
    assert [score for _track, score in result] == [0.91, 0.82]
    for call in rank.call_args_list:
        assert call.kwargs["min_score"] == 0.3
        assert call.kwargs["k"] == 3
        assert set(call.kwargs["query_vecs"]) == {QueryAxis.MOOD}
        assert set(call.kwargs["anchor_vecs"]) == {QueryAxis.MOOD}


def test_rank_text_components_truncates_only_after_minimum_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track_x = TrackData(rel_path="x.mp3", title="x", duration=60.0)
    track_y = TrackData(rel_path="y.mp3", title="y", duration=60.0)
    rank = MagicMock(
        side_effect=[
            [(track_x, 0.99), (track_y, 0.98)],
            [(track_y, 0.97), (track_x, 0.01)],
        ]
    )
    store = MagicMock()
    store.track_count.return_value = 2
    monkeypatch.setattr(
        rank_text,
        "query_vector_contrast",
        MagicMock(
            side_effect=[
                np.asarray([1.0], dtype=np.float32),
                np.asarray([2.0], dtype=np.float32),
            ]
        ),
    )
    monkeypatch.setattr(rank_text, "rank_by_upper_cluster", rank)

    result = rank_text.rank_text_components(
        store,
        MagicMock(),
        query="first, second",
        components={QueryAxis.MOOD: "first, second"},
        axis_vocab_vecs={QueryAxis.MOOD: np.asarray([[1.0]], dtype=np.float32)},
        min_score=0.0,
        k=1,
    )

    assert [call.kwargs["k"] for call in rank.call_args_list] == [2, 2]
    assert [(track.rel_path, score) for track, score in result] == [("y.mp3", 0.97)]


def test_rank_text_components_keeps_distinct_axes_in_one_rank_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mood_vec = np.asarray([1.0], dtype=np.float32)
    instrument_vec = np.asarray([2.0], dtype=np.float32)
    rank = MagicMock(return_value=[])
    embed = MagicMock(side_effect=[mood_vec, instrument_vec])
    monkeypatch.setattr(rank_text, "query_vector_contrast", embed)
    monkeypatch.setattr(rank_text, "rank_by_upper_cluster", rank)

    _ = rank_text.rank_text_components(
        MagicMock(),
        MagicMock(),
        query="dreamy piano",
        components={QueryAxis.MOOD: "dreamy", QueryAxis.INSTRUMENT: "piano"},
        axis_vocab_vecs={
            QueryAxis.MOOD: np.asarray([[1.0]], dtype=np.float32),
            QueryAxis.INSTRUMENT: np.asarray([[1.0]], dtype=np.float32),
        },
        min_score=0.7,
        k=10,
    )

    rank.assert_called_once()
    assert [call.args[0] for call in embed.call_args_list] == ["dreamy mood.", "piano."]
