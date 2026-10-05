"""Reference-bank embedding and legacy percentile calibration utilities.

Production text ranking uses the adaptive upper-cluster scorer. Percentile
scoring remains here for historical comparisons and evaluation. All rankers
scan stored window vectors; no re-index needed.
"""

import base64
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.audio.features import batches
from local_smart_playlist.embed.model import MODEL_ID, MuLanEmbedder
from local_smart_playlist.index.store import BatchedVectors, MetaKey, Store, TrackData
from local_smart_playlist.numpy_helpers import l2_normalize, reshape
from local_smart_playlist.query.axis_prompts import QueryAxis
from local_smart_playlist.query.upper_cluster import UPPER_CLUSTER_NEUTRAL, upper_cluster_fit_per_window
from local_smart_playlist.type_utils import NonEmptyTuple

W = IntVar("W")  # window count per track
N = IntVar("N")  # total window count in a flat scoring batch
D = IntVar("D")  # embedding dim
V = IntVar("V")  # vocab size

# Legacy percentile-space neutral floor. Production upper-cluster neutral is
# defined in query/upper_cluster.py.
VOCAB_SCORE_FLOOR = 0.5
CALIBRATION_BATCH_TRACKS = 256
# A window counts toward coverage when it beats more than half of caption prompts.
NEUTRAL = 0.5


def vocab_vector_bank(model: MuLanEmbedder[D], vocab: NonEmptyTuple[str]) -> np.ndarray[[V, D]]:
    """Unit-norm embeddings of the reference prompt bank, one row per prompt."""
    embeds = model.embed_texts(vocab)
    return l2_normalize(embeds)


_CACHE_FORMAT = "v1"
_CACHE_PART_COUNT = 4


def _vocab_digest(vocab: NonEmptyTuple[str]) -> str:
    """Digest vocabulary text for cache validation."""
    return hashlib.sha1("\n".join(vocab).encode("utf-8")).hexdigest()


def _encode_bank(vecs: np.ndarray[[V, D]], *, vocab_digest: str) -> str:
    """Encode model, vocab identity, and float32 vectors into meta TEXT value."""
    encoded_vecs = base64.b64encode(np.ascontiguousarray(vecs, dtype=np.float32).tobytes()).decode("ascii")
    return "\n".join((_CACHE_FORMAT, MODEL_ID, vocab_digest, encoded_vecs))


def _decode_bank(*, blob: str, dim: int, n: int, vocab_digest: str) -> np.ndarray[[V, D]] | None:
    """Decode cache value; return None when identity, encoding, or shape mismatches."""
    parts = blob.split("\n", maxsplit=3)
    if len(parts) != _CACHE_PART_COUNT or parts[0] != _CACHE_FORMAT or parts[1] != MODEL_ID or parts[2] != vocab_digest:
        return None
    try:
        arr = np.frombuffer(base64.b64decode(parts[3], validate=True), dtype=np.float32)
    except ValueError:
        return None
    if arr.size != n * dim:
        return None
    return reshape(arr, (n, dim))


def vocab_vector_bank_cached(store: Store, model: MuLanEmbedder[D], *, vocab: NonEmptyTuple[str]) -> np.ndarray[[V, D]]:
    """Unit-norm reference embeddings, cached under one fixed metadata key.

    Cache value includes format, model, and prompt-bank digest; changes silently
    invalidate the prior value. Decoding also validates row count and dimension.
    """
    digest = _vocab_digest(vocab)
    cached = store.get_meta(MetaKey.VOCAB_VECS)
    if cached is not None:
        vecs = _decode_bank(blob=cached, dim=model.dim, n=len(vocab), vocab_digest=digest)
        if vecs is not None:
            return vecs
    vecs = vocab_vector_bank(model, vocab)
    store.set_meta(MetaKey.VOCAB_VECS, _encode_bank(vecs, vocab_digest=digest))
    return vecs


@dataclass(frozen=True)
class VocabCalScore:
    """Calibrated sustained fit: mean percentile plus coverage."""

    score: float
    coverage: float


def vocab_calibration_score(
    flat_vecs: np.ndarray[[N, D]],
    *,
    track_sizes: NonEmptyTuple[int] | None = None,
    query_vec: np.ndarray[[D]],
    vocab_vecs: np.ndarray[[V, D]],
) -> list[VocabCalScore]:
    """Score a flat batch of tracks; ``track_sizes`` partitions its windows.

    Omit ``track_sizes`` to treat all rows as one track. Otherwise, sizes must
    be positive and sum to the number of rows in ``window_vecs``.
    """
    sizes = [flat_vecs.shape[0]] if track_sizes is None else list(track_sizes)
    if any(size <= 0 for size in sizes) or sum(sizes) != flat_vecs.shape[0]:
        raise ValueError("track_sizes must partition window_vecs into positive track sizes")

    sims_q = flat_vecs @ query_vec
    sims_v = flat_vecs @ vocab_vecs.T
    beaten = np.greater(sims_q[:, None], sims_v)
    percentiles = beaten.mean(axis=1)
    track_ids = np.repeat(np.arange(len(sizes)), sizes)
    track_counts = np.asarray(sizes)
    score_sums = np.bincount(track_ids, weights=percentiles, minlength=len(sizes))
    coverage_sums = np.bincount(
        track_ids,
        weights=np.greater(percentiles, NEUTRAL),
        minlength=len(sizes),
    )
    scores: list[float] = (score_sums / track_counts).tolist()
    coverages: list[float] = (coverage_sums / track_counts).tolist()
    results: list[VocabCalScore] = []
    for score, coverage in zip(scores, coverages, strict=True):
        results.append(VocabCalScore(score=score, coverage=coverage))
    return results


def _iter_calibration_batches[N2: IntVar, D2: IntVar](
    flat_vecs: BatchedVectors[N2, D2],
) -> Iterator[tuple[NonEmptyTuple[str], np.ndarray[[N2, D2]], NonEmptyTuple[int]]]:
    """Yield track paths, windows, and sizes in one ordered pass over flat vectors."""
    if len(flat_vecs.rel_paths) == 0:
        return

    paths = NonEmptyTuple(flat_vecs.rel_paths)
    batch_start = 0
    flat_id_start = 0
    for paths_batch in batches(paths, batch_size=CALIBRATION_BATCH_TRACKS):
        batch_end = batch_start + len(paths_batch)
        sizes_batch = NonEmptyTuple(flat_vecs.window_sizes[batch_start:batch_end])
        flat_id_end = flat_id_start + sum(sizes_batch)
        yield paths_batch, flat_vecs.vec_matrix[flat_id_start:flat_id_end], sizes_batch
        batch_start = batch_end
        flat_id_start = flat_id_end


def rank_by_upper_cluster[D2: IntVar, V2: IntVar](
    store: Store,
    *,
    query_vecs: dict[QueryAxis, np.ndarray[[D2]]],
    anchor_vecs: dict[QueryAxis, np.ndarray[[V2, D2]]],
    min_score: float,
    k: int,
    exclude: set[str] | None = None,
) -> list[tuple[TrackData[D2], float]]:
    """Rank tracks by query fit to each requested axis's upper anchor cluster.

    Per-track query and anchor similarities are averaged over windows, matching
    the production recipe evaluated in ``scripts/eval_softmax.py``. Composite
    queries pass only when every axis clears ``min_score``; their weakest-axis
    fit determines rank, with mean query similarity and path as tie-breaks.
    """
    if len(query_vecs) == 0 or query_vecs.keys() != anchor_vecs.keys():
        raise ValueError("query vectors and anchor banks must have identical, non-empty axes")
    if any(bank.shape[0] == 0 for bank in anchor_vecs.values()):
        raise ValueError("each requested axis needs a non-empty anchor bank")

    excluded = set() if exclude is None else set(exclude)
    flat_vecs = store.load_windows_flat(store.track_rel_paths())
    path_data = {track.rel_path: track for track in store.track_meta()}
    scored: list[tuple[TrackData[D2], float, float, float, str]] = []

    for paths_batch, windows, sizes_batch in _iter_calibration_batches(flat_vecs):
        starts = np.cumsum([0, *sizes_batch[:-1]]).tolist()
        ends = np.cumsum(sizes_batch).tolist()
        fits_by_axis: dict[QueryAxis, list[float]] = {}
        query_means_by_axis: dict[QueryAxis, list[float]] = {}
        for axis, query_vec in query_vecs.items():
            query_sims = np.asarray(windows @ query_vec, dtype=np.float64).ravel()
            anchor_sims = np.asarray(windows @ anchor_vecs[axis].T, dtype=np.float64)
            query_means = [float(query_sims[start:end].mean()) for start, end in zip(starts, ends, strict=True)]
            anchor_means = [
                anchor_sims[start:end].mean(axis=0) for start, end in zip(starts, ends, strict=True)
            ]
            fit_values = upper_cluster_fit_per_window(
                np.asarray(query_means, dtype=np.float64),
                np.stack(anchor_means),
            )
            fits_by_axis[axis] = fit_values.tolist()
            query_means_by_axis[axis] = query_means

        for index, path in enumerate(paths_batch):
            track = path_data.get(path)
            if path in excluded or track is None:
                continue
            axis_scores = [fits[index] for fits in fits_by_axis.values()]
            weakest_score = min(axis_scores)
            passes_cutoff = (
                weakest_score > UPPER_CLUSTER_NEUTRAL
                if min_score == UPPER_CLUSTER_NEUTRAL
                else weakest_score >= min_score
            )
            if passes_cutoff:
                mean_score = sum(axis_scores) / len(axis_scores)
                mean_query = sum(means[index] for means in query_means_by_axis.values()) / len(axis_scores)
                scored.append((track, weakest_score, mean_score, mean_query, path))

    scored.sort(key=lambda entry: (-entry[1], -entry[2], -entry[3], entry[4]))
    return [(track, score) for track, score, _mean_score, _query_mean, _path in scored[:k]]


def rank_by_vocab_calibration(
    store: Store,
    *,
    query_vec: np.ndarray[[D]],
    vocab_vecs: np.ndarray[[V, D]],
    k: int,
    exclude: set[str] | None = None,
) -> list[tuple[TrackData[D], float]]:
    """Rank every indexed track against one global reference bank."""
    excluded = set() if exclude is None else set(exclude)
    flat_vecs = store.load_windows_flat(store.track_rel_paths())
    path_data = {track.rel_path: track for track in store.track_meta()}
    scored: list[tuple[TrackData[D], float, float, str]] = []
    for paths_batch, windows, sizes_batch in _iter_calibration_batches(flat_vecs):
        batch_scores = vocab_calibration_score(
            windows,
            track_sizes=sizes_batch,
            query_vec=query_vec,
            vocab_vecs=vocab_vecs,
        )
        for path, result in zip(paths_batch, batch_scores, strict=True):
            track = path_data.get(path)
            if path not in excluded and track is not None:
                scored.append((track, result.score, result.coverage, path))

    scored.sort(key=lambda entry: (-entry[1], -entry[2], entry[3]))
    return [(track, score) for track, score, _coverage, _rel in scored[:k]]


def rank_by_axis_calibration(
    store: Store,
    *,
    query_vecs: dict[QueryAxis, np.ndarray[[D]]],
    vocab_vecs: dict[QueryAxis, np.ndarray[[V, D]]],
    min_score: float,
    k: int,
    exclude: set[str] | None = None,
) -> list[tuple[TrackData[D], float]]:
    """Score each axis bank per batch in one library traversal; rank by weakest fit."""
    if len(query_vecs) == 0 or query_vecs.keys() != vocab_vecs.keys():
        raise ValueError("query vectors and calibration banks must have identical, non-empty axes")
    if any(bank.shape[0] == 0 for bank in vocab_vecs.values()):
        raise ValueError("each requested axis needs a non-empty calibration bank")

    excluded = set() if exclude is None else set(exclude)
    flat_vecs = store.load_windows_flat(store.track_rel_paths())
    path_data = {track.rel_path: track for track in store.track_meta()}
    scored: list[tuple[TrackData[D], float, float, str]] = []

    for paths_batch, windows, sizes_batch in _iter_calibration_batches(flat_vecs):
        scores_by_axis = {
            axis: vocab_calibration_score(
                windows,
                track_sizes=sizes_batch,
                query_vec=query_vec,
                vocab_vecs=vocab_vecs[axis],
            )
            for axis, query_vec in query_vecs.items()
        }
        for track_index, path in enumerate(paths_batch):
            track = path_data.get(path)
            if path in excluded or track is None:
                continue
            axis_scores = [scores[track_index].score for scores in scores_by_axis.values()]
            weakest_score = min(axis_scores)
            if weakest_score >= min_score:
                scored.append((track, weakest_score, sum(axis_scores) / len(axis_scores), path))

    scored.sort(key=lambda entry: (-entry[1], -entry[2], entry[3]))
    return [(track, weakest_score) for track, weakest_score, _mean_score, _path in scored[:k]]
