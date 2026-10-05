"""Scale-aware query fit within the upper cluster of anchor affinities."""

from math import exp, sqrt

import numpy as np
from shape_extensions import IntVar

W = IntVar("W")  # windows per batch
A = IntVar("A")  # anchor count per window

UPPER_CLUSTER_MIN_REST_SIZE = 3
UPPER_CLUSTER_MIN_TAIL_SIZE = 2
UPPER_CLUSTER_TOP_SIZE = 5
UPPER_CLUSTER_MIN_GAP_RATIO = 2.0
UPPER_CLUSTER_MIN_SCALE = 0.005
UPPER_CLUSTER_NEUTRAL = 0.5


def upper_cluster_fit_per_window[W2: IntVar, A2: IntVar](
    query_sims: np.ndarray[[W2]],
    anchor_sims: np.ndarray[[W2, A2]],
) -> np.ndarray[[W2]]:
    """Score query closeness to a clearly separated upper tail of five anchors.

    Split the top five only when an eligible boundary gap is at least twice
    every other adjacent gap. Otherwise, keep all five in the upper cluster.
    All remaining anchors form the lower cluster. A query above the fitted
    upper center receives exact score 1.0.
    """
    if query_sims.shape[0] != anchor_sims.shape[0]:
        raise ValueError("query and anchor similarities must have the same window count")

    anchor_count = anchor_sims.shape[1]
    lower_size = anchor_count - UPPER_CLUSTER_TOP_SIZE
    if lower_size < UPPER_CLUSTER_MIN_REST_SIZE:
        raise ValueError("anchor bank is too small to split top anchors from the rest")

    fits: np.ndarray[[W2]] = np.empty(query_sims.shape, dtype=np.float64)
    for row in range(anchor_sims.shape[0]):
        ordered = sorted(float(anchor_sims[row, col]) for col in range(anchor_count))
        top = ordered[-UPPER_CLUSTER_TOP_SIZE:]
        gaps = [top[index] - top[index - 1] for index in range(1, UPPER_CLUSTER_TOP_SIZE)]
        best_gap = -1.0
        best_upper_start = UPPER_CLUSTER_TOP_SIZE
        for split in range(UPPER_CLUSTER_MIN_TAIL_SIZE, UPPER_CLUSTER_TOP_SIZE - UPPER_CLUSTER_MIN_TAIL_SIZE + 1):
            gap = gaps[split - 1]
            if gap > best_gap:
                best_gap = gap
                best_upper_start = split

        other_gaps = [gap for index, gap in enumerate(gaps, start=1) if index != best_upper_start]
        next_largest_gap = max(other_gaps)
        gap_ratio = best_gap / max(next_largest_gap, UPPER_CLUSTER_MIN_SCALE)
        upper = top[best_upper_start:] if gap_ratio >= UPPER_CLUSTER_MIN_GAP_RATIO else top
        lower = ordered[: anchor_count - len(upper)]
        low_mean = sum(lower) / len(lower)
        high_mean = sum(upper) / len(upper)
        within_sse = sum((value - low_mean) ** 2 for value in lower) + sum((value - high_mean) ** 2 for value in upper)
        scale = max(sqrt(within_sse / anchor_count), UPPER_CLUSTER_MIN_SCALE)

        query_sim = float(query_sims[row])
        if query_sim > high_mean:
            fits[row] = 1.0  # pyrefly: ignore[unsupported-operation]  # shaped ndarray __setitem__
            continue

        margin = (abs(query_sim - low_mean) - abs(query_sim - high_mean)) / scale
        if margin >= 0.0:
            fit = 1.0 / (1.0 + exp(-margin))
        else:
            exp_margin = exp(margin)
            fit = exp_margin / (1.0 + exp_margin)
        fits[row] = fit  # pyrefly: ignore[unsupported-operation]  # shaped ndarray __setitem__

    return fits
