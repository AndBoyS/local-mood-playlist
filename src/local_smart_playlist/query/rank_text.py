"""Axis-routed text-query ranking using adaptive upper-cluster fits."""

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.embed.model import MuLanEmbedder
from local_smart_playlist.index.store import Store, TrackData
from local_smart_playlist.query.axis_prompts import QueryAxis
from local_smart_playlist.query.contrast import query_vector_contrast
from local_smart_playlist.query.prompts import calibration_axis_ranges
from local_smart_playlist.query.vocab_cal import rank_by_upper_cluster


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
    axis_vocab_vecs: dict[QueryAxis, np.ndarray[[V2, D2]]],
    min_score: float,
    k: int,
    exclude: set[str] | None = None,
) -> list[tuple[TrackData[D2], float]]:
    """Rank query against per-axis upper clusters; all present axes must pass."""
    if components is None or len(components) == 0:
        query_vecs = {QueryAxis.MOOD: query_vector_contrast(query, model)}
    else:
        query_vecs = {
            axis: query_vector_contrast(component, model)
            for axis, component in components.items()
        }
    banks = {axis: axis_vocab_vecs[axis] for axis in query_vecs}
    return rank_by_upper_cluster(
        store,
        query_vecs=query_vecs,
        anchor_vecs=banks,
        min_score=min_score,
        k=k,
        exclude=exclude,
    )
