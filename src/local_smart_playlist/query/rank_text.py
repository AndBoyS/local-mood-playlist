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


def _caption_style_component(axis: QueryAxis, term: str) -> str:
    """Format a query term like reference captions for its axis."""
    component = term.strip().rstrip(" .")
    if axis == QueryAxis.MOOD and not component.casefold().endswith(" mood"):
        component = f"{component} mood"
    return f"{component}."


def _minimum_component_rankings[D3: IntVar](
    rankings: list[list[tuple[TrackData[D3], float]]],
) -> list[tuple[TrackData[D3], float]]:
    """Intersect component rankings and use weakest component as track score."""
    if len(rankings) == 0:
        return []
    if len(rankings) == 1:
        return rankings[0]

    score_maps = [
        {track.rel_path: (track, score) for track, score in ranking}
        for ranking in rankings
    ]
    common_paths = set(score_maps[0])
    for scores in score_maps[1:]:
        common_paths.intersection_update(scores)

    merged: list[tuple[TrackData[D3], float, float]] = []
    for path in common_paths:
        matches = [scores[path] for scores in score_maps]
        component_scores = [score for _track, score in matches]
        merged.append((matches[0][0], min(component_scores), sum(component_scores) / len(component_scores)))

    def sort_key(row: tuple[TrackData[D3], float, float]) -> tuple[float, float, str]:
        track, minimum_score, mean_score = row
        return (-minimum_score, -mean_score, track.rel_path)

    merged.sort(key=sort_key)
    return [(track, minimum_score) for track, minimum_score, _mean_score in merged]


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
    """Rank comma-separated attributes separately; each term must fit."""
    source_components = (
        [(QueryAxis.MOOD, query)]
        if components is None or len(components) == 0
        else list(components.items())
    )
    query_components = [
        (axis, term.strip())
        for axis, component in source_components
        for term in component.split(",")
        if term.strip() != ""
    ]
    if len(query_components) == 0:
        raise ValueError("query components must contain a non-empty term")

    query_vectors = [
        (axis, query_vector_contrast(_caption_style_component(axis, term), model))
        for axis, term in query_components
    ]
    if len({axis for axis, _vector in query_vectors}) == len(query_vectors):
        query_vecs = {axis: vec for axis, vec in query_vectors}
        banks = {axis: axis_vocab_vecs[axis] for axis in query_vecs}
        return rank_by_upper_cluster(
            store,
            query_vecs=query_vecs,
            anchor_vecs=banks,
            min_score=min_score,
            k=k,
            exclude=exclude,
        )

    # Keep complete term rankings; a track outside one term's top-k could win by minimum.
    per_component_rankings = [
        rank_by_upper_cluster(
            store,
            query_vecs={axis: vec},
            anchor_vecs={axis: axis_vocab_vecs[axis]},
            min_score=min_score,
            k=store.track_count(),
            exclude=exclude,
        )
        for axis, vec in query_vectors
    ]
    return _minimum_component_rankings(per_component_rankings)[:k]
