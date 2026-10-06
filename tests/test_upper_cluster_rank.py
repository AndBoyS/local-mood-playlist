"""Production ranking over adaptive upper clusters."""

from math import sqrt
from pathlib import Path

import numpy as np
import pytest
from shape_extensions import IntVar

from local_smart_playlist.index.store import Store
from local_smart_playlist.query import vocab_cal
from local_smart_playlist.query.axis_prompts import QueryAxis
from local_smart_playlist.query.vocab_cal import rank_by_upper_cluster

DIM = 10
ANCHOR_COSINES = [-0.4, -0.3, -0.2, 0.1, 0.2, 0.7, 0.8, 0.9]


def basis(index: int) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = 1.0  # pyrefly: ignore[unsupported-operation]
    return vector


def track_vector(query_similarity: float) -> np.ndarray:
    return query_similarity * basis(0) + sqrt(1.0 - query_similarity**2) * basis(1)


def add_track(store: Store, rel_path: str, query_similarity: float) -> None:
    vector = track_vector(query_similarity)
    windows = np.stack([vector] * 2)
    store.upsert(
        rel_path=rel_path,
        mean_vec=vector,
        p90_vec=vector,
        n_windows=len(windows),
        duration=60.0,
        title=rel_path,
        model="fake",
        indexed_at="now",
        window_vecs=windows,
    )


def anchor_bank(cosines: list[float] = ANCHOR_COSINES) -> np.ndarray:
    return np.stack(
        [cosine * basis(1) + sqrt(1.0 - cosine**2) * basis(index + 2) for index, cosine in enumerate(cosines)]
    )


def test_cluster_rank_uses_eval_order_ties_cutoff_and_exclusions(tmp_path: Path) -> None:
    store = Store(tmp_path / "rank.db", embed_dim=DIM)
    add_track(store, "a-high.mp3", 0.4)
    add_track(store, "b-very-high.mp3", 0.95)
    add_track(store, "c-middle.mp3", 0.4)
    add_track(store, "d-low.mp3", 0.1)

    ranked = rank_by_upper_cluster(
        store,
        query_vecs={QueryAxis.MOOD: basis(0)},
        anchor_vecs={QueryAxis.MOOD: anchor_bank()},
        min_score=0.5,
        k=3,
        exclude={"a-high.mp3"},
    )

    assert [track.rel_path for track, _score in ranked] == ["b-very-high.mp3", "c-middle.mp3"]
    assert ranked[0][1] == 1.0
    assert ranked[1][1] > 0.5


def test_cluster_rank_rejects_query_closer_to_lower_cluster(tmp_path: Path) -> None:
    store = Store(tmp_path / "rank.db", embed_dim=DIM)
    add_track(store, "low-fit.mp3", 0.1)

    ranked = rank_by_upper_cluster(
        store,
        query_vecs={QueryAxis.MOOD: basis(0)},
        anchor_vecs={QueryAxis.MOOD: anchor_bank()},
        min_score=0.5,
        k=1,
    )

    assert ranked == []


def test_cluster_rank_includes_below_neutral_tracks_at_lower_cutoff(tmp_path: Path) -> None:
    store = Store(tmp_path / "rank.db", embed_dim=DIM)
    add_track(store, "low-fit.mp3", 0.1)

    ranked = rank_by_upper_cluster(
        store,
        query_vecs={QueryAxis.MOOD: basis(0)},
        anchor_vecs={QueryAxis.MOOD: anchor_bank()},
        min_score=0.0,
        k=1,
    )

    assert len(ranked) == 1
    assert ranked[0][0].rel_path == "low-fit.mp3"
    assert ranked[0][1] < 0.5


def test_cluster_rank_excludes_neutral_tie_at_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store(tmp_path / "rank.db", embed_dim=DIM)
    add_track(store, "neutral.mp3", 0.4)

    def neutral_fit[W: IntVar, A: IntVar](
        query_sims: np.ndarray[[W]], _anchor_sims: np.ndarray[[W, A]]
    ) -> np.ndarray[[W]]:
        return query_sims * 0.0 + 0.5

    monkeypatch.setattr(vocab_cal, "upper_cluster_fit_per_window", neutral_fit)

    ranked = rank_by_upper_cluster(
        store,
        query_vecs={QueryAxis.MOOD: basis(0)},
        anchor_vecs={QueryAxis.MOOD: anchor_bank()},
        min_score=0.5,
        k=1,
    )

    assert ranked == []


def test_cluster_rank_composite_requires_every_axis_and_orders_by_weakest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path / "rank.db", embed_dim=DIM)
    for path, similarity in (
        ("a-both.mp3", 0.9),
        ("b-both.mp3", 0.8),
        ("c-mood-only.mp3", 0.7),
        ("d-instrument-only.mp3", 0.6),
    ):
        add_track(store, path, similarity)

    axis_call = 0

    def varying_axis_fit[W: IntVar, A: IntVar](
        query_sims: np.ndarray[[W]], _anchor_sims: np.ndarray[[W, A]]
    ) -> np.ndarray[[W]]:
        nonlocal axis_call
        axis_call += 1
        return query_sims if axis_call % 2 == 1 else query_sims * 0.5 + 0.4

    monkeypatch.setattr(vocab_cal, "upper_cluster_fit_per_window", varying_axis_fit)
    bank = anchor_bank()

    ranked = rank_by_upper_cluster(
        store,
        query_vecs={QueryAxis.MOOD: basis(0), QueryAxis.INSTRUMENT: basis(0)},
        anchor_vecs={QueryAxis.MOOD: bank, QueryAxis.INSTRUMENT: bank},
        min_score=0.75,
        k=store.track_count(),
    )

    assert [(track.rel_path, score) for track, score in ranked] == [
        ("a-both.mp3", pytest.approx(0.85)),
        ("b-both.mp3", pytest.approx(0.8)),
    ]


def test_cluster_rank_requires_matching_nonempty_axes(tmp_path: Path) -> None:
    store = Store(tmp_path / "rank.db", embed_dim=DIM)
    with pytest.raises(ValueError, match="identical, non-empty axes"):
        _ = rank_by_upper_cluster(
            store,
            query_vecs={QueryAxis.MOOD: basis(0)},
            anchor_vecs={},
            min_score=0.5,
            k=1,
        )
