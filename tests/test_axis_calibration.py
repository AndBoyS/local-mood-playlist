"""Axis routing and hard all-axes calibration gates."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from local_smart_playlist.index.store import Store
from local_smart_playlist.query.axis_prompts import AXIS_TO_PROMPTS, QueryAxis
from local_smart_playlist.query.prompts import calibration_axis_ranges
from local_smart_playlist.query.vocab_cal import rank_by_axis_calibration

DIM = 32


def basis(index: int) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = 1.0  # pyrefly: ignore[unsupported-operation]  # numpy shape stubs lack __setitem__
    return vector


def upsert_track(store: Store, rel_path: str, window_vecs: np.ndarray) -> None:
    mean_vec = window_vecs.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    mean_vec = (mean_vec / norm).astype(np.float32)
    store.upsert(
        rel_path=rel_path,
        mean_vec=mean_vec,
        p90_vec=mean_vec,
        n_windows=len(window_vecs),
        duration=60.0,
        title=rel_path,
        model="fake",
        indexed_at="now",
        window_vecs=window_vecs,
    )


def test_reference_prompts_are_balanced_and_ranges_match() -> None:
    assert set(AXIS_TO_PROMPTS) == set(QueryAxis)
    assert all(len(prompts) == 32 for prompts in AXIS_TO_PROMPTS.values())

    ranges = calibration_axis_ranges()
    ordered = list(ranges.values())
    assert ordered[0][0] > 0  # readout vocabulary precedes axis references
    assert all(end - start == 32 for start, end in ordered)
    assert all(left[1] == right[0] for left, right in zip(ordered, ordered[1:], strict=False))


def test_axis_rank_requires_every_requested_axis_to_clear_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store(Path(":memory:"), embed_dim=DIM)
    diagonal = (basis(0) + basis(1)) / np.sqrt(2.0)
    upsert_track(store, "both-axes.mp3", np.stack([diagonal, diagonal]))
    upsert_track(store, "mood-only.mp3", np.stack([basis(0), basis(0)]))
    upsert_track(store, "instrument-only.mp3", np.stack([basis(1), basis(1)]))
    load_windows = MagicMock(wraps=store.load_windows_flat)
    monkeypatch.setattr(store, "load_windows_flat", load_windows)

    ranked = rank_by_axis_calibration(
        store,
        query_vecs={QueryAxis.MOOD: basis(0), QueryAxis.INSTRUMENT: basis(1)},
        vocab_vecs={QueryAxis.MOOD: np.stack([basis(2)]), QueryAxis.INSTRUMENT: np.stack([basis(3)])},
        min_score=0.5,
        k=store.track_count(),
    )

    assert [track.rel_path for track, _score in ranked] == ["both-axes.mp3"]
    assert ranked[0][1] == pytest.approx(1.0)
    load_windows.assert_called_once()
