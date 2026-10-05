"""Track → prompt ranking for `sp describe` (diagnostics only).

Playlist ranking never uses this module; it supports the per-track readout by
finding prompt phrases nearest to the track's mean vector.
"""

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.type_utils import NonEmptyTuple

V = IntVar("V")
D = IntVar("D")

DOCS_PER_TRACK = 10  # phrases shown per track


def track_documents(
    mean_vec: np.ndarray[[D]], vocab_vecs: np.ndarray[[V, D]], *, top_n: int = DOCS_PER_TRACK
) -> NonEmptyTuple[tuple[int, float]]:
    """(vocab_index, sim) for the *top_n* vocab phrases nearest to a unit track-mean vector."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    sims = (vocab_vecs @ mean_vec).ravel()
    count = min(top_n, vocab_vecs.shape[0])
    order = np.argsort(-sims)[:count]
    return NonEmptyTuple((int(i), float(sims[int(i)])) for i in order)
