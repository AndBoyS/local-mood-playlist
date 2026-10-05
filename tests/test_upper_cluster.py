"""Upper-cluster fit scorer tests."""

import numpy as np
import pytest

from local_smart_playlist.query.upper_cluster import upper_cluster_fit_per_window


def test_query_on_lower_edge_of_upper_anchor_cluster_scores_high() -> None:
    anchors = np.asarray([[1, 1.1, 1.2, 10, 10.1, 10.2, 10.3, 10.4]], dtype=np.float64)
    query = np.asarray([10], dtype=np.float64)

    fits = upper_cluster_fit_per_window(query, anchors)

    assert float(fits[0]) > 0.99


def test_query_closer_to_lower_cluster_scores_below_neutral() -> None:
    anchors = np.asarray([[1, 1.1, 1.2, 10, 10.1, 10.2, 10.3, 10.4]], dtype=np.float64)
    query = np.asarray([1.2], dtype=np.float64)

    fits = upper_cluster_fit_per_window(query, anchors)

    assert float(fits[0]) < 0.01


def test_unimodal_anchor_bank_still_splits_top_five_from_rest() -> None:
    anchors = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float64)
    query = np.asarray([8], dtype=np.float64)

    fits = upper_cluster_fit_per_window(query, anchors)

    assert float(fits[0]) > 0.5


def test_clear_gap_inside_top_five_drops_lower_tail_values() -> None:
    anchors = np.asarray([[-0.5, -0.4, -0.3, 0.1, 0.2, 0.7, 0.8, 0.9]], dtype=np.float64)
    query = np.asarray([0.15], dtype=np.float64)

    fits = upper_cluster_fit_per_window(query, anchors)

    assert float(fits[0]) < 0.5


def test_query_above_upper_center_scores_one_even_below_top_anchor() -> None:
    anchors = np.asarray([[-0.5, -0.4, -0.3, 0.1, 0.2, 0.7, 0.8, 1.0]], dtype=np.float64)
    query = np.asarray([0.9], dtype=np.float64)

    fits = upper_cluster_fit_per_window(query, anchors)

    assert float(query[0]) < float(anchors[0, -1])
    assert float(fits[0]) == 1.0


def test_upper_cluster_requires_three_anchors_outside_top_five() -> None:
    anchors = np.asarray([[1, 2, 3, 4, 5, 6, 7]], dtype=np.float64)
    query = np.asarray([7], dtype=np.float64)

    with pytest.raises(ValueError, match="too small"):
        _ = upper_cluster_fit_per_window(query, anchors)


def test_upper_cluster_requires_matching_window_counts() -> None:
    anchors = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float64)
    query = np.asarray([0, 1], dtype=np.float64)

    with pytest.raises(ValueError, match="same window count"):
        _ = upper_cluster_fit_per_window(query, anchors)
