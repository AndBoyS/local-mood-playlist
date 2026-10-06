"""Playlist command query-union tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shape_extensions import IntVar

from local_smart_playlist.commands import playlist_cmd
from local_smart_playlist.commands.playlist_cmd import merge_query_rankings
from local_smart_playlist.index.store import TrackData
from local_smart_playlist.query.prompts import LlmError


def test_merge_query_rankings_keeps_unique_tracks_and_best_score() -> None:
    first = TrackData(rel_path="first.mp3", title="first", duration=100.0)
    shared = TrackData(rel_path="shared.mp3", title="shared", duration=120.0)
    third = TrackData(rel_path="third.mp3", title="third", duration=90.0)

    merged = merge_query_rankings(
        [
            [(first, 0.8), (shared, 0.7)],
            [(shared, 0.9), (third, 0.85)],
        ]
    )

    assert [(track.rel_path, score) for track, score in merged] == [
        ("shared.mp3", 0.9),
        ("third.mp3", 0.85),
        ("first.mp3", 0.8),
    ]


def test_play_starts_vacuum_after_closing_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    db.touch()
    store = MagicMock()
    store.__enter__.return_value = store
    store.get_meta.return_value = str(tmp_path)
    store.get_track.return_value = MagicMock(rel_path="seed.mp3", vectors=MagicMock(mean_vec=object()))
    store.track_count.return_value = 1
    store.vacuum_if_needed.return_value.recommended = True
    track = TrackData(rel_path="result.mp3", title="result", duration=60.0)

    monkeypatch.setattr(playlist_cmd, "cap_torch_threads", lambda: None)

    def make_store(_db: Path) -> MagicMock:
        return store

    def fake_rank[D: IntVar](
        *_args: object, **_kwargs: object
    ) -> list[tuple[TrackData[D], float]]:
        return [(track, 0.95)]

    monkeypatch.setattr(playlist_cmd, "Store", make_store)
    monkeypatch.setattr(playlist_cmd, "rank_by_similarity", fake_rank)

    def start_background(_db_path: Path) -> bool:
        assert store.__exit__.called
        return True

    monkeypatch.setattr(playlist_cmd.vacuum_worker, "start_background", start_background)
    playlist_cmd.PlayArgs.run(
        query="seed",
        db=db,
        n=1,
        out=None,
        llm=False,
        seed_track="seed.mp3",
        alpha=playlist_cmd.PLAY_DEFAULT_ALPHA,
        min_score=0.9,
        dry=True,
    )
    store.vacuum_if_needed.assert_called_once_with(execute=False)


def test_play_decomposes_each_llm_variant_before_ranking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_smart_playlist.query.axis_prompts import QueryAxis

    db = tmp_path / "index.db"
    db.touch()
    store = MagicMock()
    store.__enter__.return_value = store
    store.get_meta.return_value = str(tmp_path)
    store.track_count.return_value = 1
    store.vacuum_if_needed.return_value.recommended = False
    track = TrackData(rel_path="result.mp3", title="result", duration=60.0)
    variants = ["sad piano", "dreamy jazz"]
    decompositions = [
        {QueryAxis.MOOD: "sad", QueryAxis.INSTRUMENT: "piano"},
        {QueryAxis.MOOD: "dreamy", QueryAxis.GENRE: "jazz"},
    ]
    decompose = MagicMock(side_effect=decompositions)
    rank = MagicMock(return_value=[(track, 0.95)])

    def make_store(_db: Path) -> MagicMock:
        return store

    monkeypatch.setattr(playlist_cmd, "cap_torch_threads", lambda: None)
    monkeypatch.setattr(playlist_cmd, "Store", make_store)
    monkeypatch.setattr(playlist_cmd, "load_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(playlist_cmd, "vocab_vector_bank_cached", MagicMock(return_value=object()))
    monkeypatch.setattr(playlist_cmd, "axis_vector_banks", MagicMock(return_value={}))
    adapt = MagicMock(return_value=variants)
    monkeypatch.setattr(playlist_cmd, "adapt_mood", adapt)
    monkeypatch.setattr(playlist_cmd, "decompose_query_axes", decompose)
    monkeypatch.setattr(playlist_cmd, "rank_text_components", rank)

    playlist_cmd.PlayArgs.run(
        query="a sad piano tune",
        db=db,
        n=None,
        out=None,
        llm=True,
        context="for reading",
        seed_track=None,
        alpha=playlist_cmd.PLAY_DEFAULT_ALPHA,
        min_score=0.9,
        dry=True,
    )

    adapt.assert_called_once_with("a sad piano tune", context="for reading")
    assert [call.args[0] for call in decompose.call_args_list] == variants
    assert [call.kwargs["components"] for call in rank.call_args_list] == decompositions
    assert rank.call_count == len(variants)
    output = capsys.readouterr().out
    assert "LLM query variants:\n  1. sad piano\n  2. dreamy jazz" in output
    assert "  sad piano -> mood: sad, instruments/sources: piano" in output
    assert "  dreamy jazz -> mood: dreamy, genre/style: jazz" in output


def test_play_allows_min_score_below_neutral(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "index.db"
    db.touch()
    store = MagicMock()
    store.__enter__.return_value = store
    store.get_meta.return_value = str(tmp_path)
    store.track_count.return_value = 1
    store.vacuum_if_needed.return_value.recommended = False
    track = TrackData(rel_path="weak.mp3", title="weak", duration=60.0)
    rank = MagicMock(return_value=[(track, 0.4)])

    def make_store(_db: Path) -> MagicMock:
        return store

    monkeypatch.setattr(playlist_cmd, "cap_torch_threads", lambda: None)
    monkeypatch.setattr(playlist_cmd, "Store", make_store)
    monkeypatch.setattr(playlist_cmd, "load_model", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(playlist_cmd, "vocab_vector_bank_cached", MagicMock(return_value=object()))
    monkeypatch.setattr(playlist_cmd, "axis_vector_banks", MagicMock(return_value={}))
    monkeypatch.setattr(playlist_cmd, "rank_text_components", rank)

    playlist_cmd.PlayArgs.run(
        query="dreamy",
        db=db,
        n=None,
        out=None,
        llm=False,
        seed_track=None,
        alpha=playlist_cmd.PLAY_DEFAULT_ALPHA,
        min_score=0.3,
        dry=True,
    )

    assert rank.call_args.kwargs["min_score"] == 0.3
    assert "0.400" in capsys.readouterr().out


@pytest.mark.parametrize("min_score", [-0.1, 1.0])
def test_play_rejects_min_score_outside_range(min_score: float, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="min score must be in"):
        playlist_cmd.PlayArgs.run(
            query="dreamy",
            db=tmp_path / "missing.db",
            n=None,
            out=None,
            llm=False,
            seed_track=None,
            alpha=playlist_cmd.PLAY_DEFAULT_ALPHA,
            min_score=min_score,
            dry=True,
        )


def test_play_crashes_when_llm_adaptation_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    db.touch()
    store = MagicMock()

    def fail_adapt(_query: str) -> list[str]:
        raise LlmError("failed")

    def make_store(_db: Path) -> MagicMock:
        return store

    monkeypatch.setattr(playlist_cmd, "cap_torch_threads", lambda: None)
    monkeypatch.setattr(playlist_cmd, "Store", make_store)
    monkeypatch.setattr(playlist_cmd, "adapt_mood", fail_adapt)

    with pytest.raises(LlmError, match="failed"):
        playlist_cmd.PlayArgs.run(
            query="sad",
            db=db,
            n=None,
            out=None,
            llm=True,
            seed_track=None,
            alpha=playlist_cmd.PLAY_DEFAULT_ALPHA,
            min_score=0.9,
            dry=True,
        )
