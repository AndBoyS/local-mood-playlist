"""Axis-aware text-query calibration and fallback to the global reference bank."""

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.embed.model import MuLanEmbedder
from local_smart_playlist.index.store import Store, TrackData
from local_smart_playlist.query.axis_prompts import QueryAxis
from local_smart_playlist.query.contrast import query_vector_contrast
from local_smart_playlist.query.prompts import calibration_axis_ranges
from local_smart_playlist.query.vocab_cal import rank_by_axis_calibration, rank_by_vocab_calibration


def axis_vector_banks[D2: IntVar, V2: IntVar](
    vocab_vecs: np.ndarray[[V2, D2]],
) -> dict[QueryAxis, np.ndarray[[V2, D2]]]:
    """Select per-axis reference vectors from the full cached prompt bank."""
    return {
        axis: vocab_vecs[start:end]
        for axis, (start, end) in calibration_axis_ranges().items()
    }


def rank_text_components[D2: IntVar, V2: IntVar](
    store: Store,
    model: MuLanEmbedder[D2],
    *,
    query: str,
    components: dict[QueryAxis, str] | None,
    vocab_vecs: np.ndarray[[V2, D2]],
    axis_vocab_vecs: dict[QueryAxis, np.ndarray[[V2, D2]]],
    min_score: float,
    k: int,
    exclude: set[str] | None = None,
) -> list[tuple[TrackData[D2], float]]:
    """Rank a pre-decomposed query; require every present axis to clear cutoff."""
    if components is None or len(components) == 0:
        ranked = rank_by_vocab_calibration(
            store,
            query_vec=query_vector_contrast(query, model),
            vocab_vecs=vocab_vecs,
            k=k,
            exclude=exclude,
        )
        return [(track, score) for track, score in ranked if score >= min_score]

    query_vecs = {
        axis: query_vector_contrast(component, model)
        for axis, component in components.items()
    }
    banks = {axis: axis_vocab_vecs[axis] for axis in components}
    return rank_by_axis_calibration(
        store,
        query_vecs=query_vecs,
        vocab_vecs=banks,
        min_score=min_score,
        k=k,
        exclude=exclude,
    )
