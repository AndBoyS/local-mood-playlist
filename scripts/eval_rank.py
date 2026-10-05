"""Offline ranking eval: probe queries x exemplar tracks over the live index.

Measures exemplar ranks under the production upper-cluster scorer, plus a
top-k preview. Read-only: never writes the DB or playlists.

Usage:
    uv run python scripts/eval_rank.py                  # default probes
    uv run python scripts/eval_rank.py "calm piano"     # ad-hoc probe
"""

import argparse
from pathlib import Path

from shape_extensions import IntVar

from local_smart_playlist.embed.model import MODEL_ID, MuLanEmbedder, cap_torch_threads, load_model
from local_smart_playlist.index.store import Store, TrackData
from local_smart_playlist.query.prompts import calibration_prompt_bank
from local_smart_playlist.query.rank_text import axis_vector_banks, rank_text_components
from local_smart_playlist.query.upper_cluster import UPPER_CLUSTER_NEUTRAL
from local_smart_playlist.query.vocab_cal import vocab_vector_bank

# Probe query -> case-insensitive substrings of exemplar tracks (rel_path or
# title). Soft/dense exemplars from difficulties.md §1 measurements.
PROBES: dict[str, list[str]] = {
    "dreamy, melancholic": ["To Far Shores", "The Terminal Show"],
    "calm piano": ["Celeste"],
    "dark, aggressive": [],
}
PREVIEW_COUNT = 5
PERCENTILE_SCALE = 100.0
DEFAULT_MIN_SCORE = 0.9  # sp play --min-score default; kept = score >= max(UPPER_CLUSTER_NEUTRAL, this)


def rank_full[D: IntVar](
    store: Store, model: MuLanEmbedder[D], *, query: str
) -> list[tuple[TrackData[D], float]]:
    """Full-library ranking with the production upper-cluster scorer."""
    vocab_vecs = vocab_vector_bank(model, calibration_prompt_bank())
    ranked = rank_text_components(
        store,
        model,
        query=query,
        components=None,
        axis_vocab_vecs=axis_vector_banks(vocab_vecs),
        min_score=0.0,
        k=store.track_count(),
        exclude=set(),
    )
    return [(track, float(score)) for track, score in ranked]


def report[D: IntVar](ranked: list[tuple[TrackData[D], float]], *, query: str, exemplars: list[str]) -> None:
    total = len(ranked)
    print(f"== {query!r} — {total} ranked tracks")
    for needle in exemplars:
        hit = next(
            ((idx, score) for idx, (track, score) in enumerate(ranked) if _matches(track, needle)),
            None,
        )
        if hit is None:
            print(f"   exemplar {needle!r}: NOT IN INDEX (or no stored windows)")
            continue
        rank0, score = hit
        pct = rank0 / total * PERCENTILE_SCALE if total > 0 else 0.0
        kept = score >= max(UPPER_CLUSTER_NEUTRAL, DEFAULT_MIN_SCORE)
        print(
            f"   exemplar {needle!r}: rank {rank0 + 1}/{total}"
            f" ({pct:.1f}% from top)  score {score:.3f}  kept={kept}"
        )
    print("   top preview:")
    for track, score in ranked[:PREVIEW_COUNT]:
        print(f"     {score:.3f}  {track.title}")


def _matches[D: IntVar](track: TrackData[D], needle: str) -> bool:
    needle_l = needle.lower()
    return needle_l in track.rel_path.lower() or needle_l in track.title.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("queries", nargs="*", help="probe queries (default: built-in PROBES)")
    _ = parser.add_argument("--db", type=Path, default=None, help="index DB path")
    args = parser.parse_args()

    cap_torch_threads()
    db_path = args.db if args.db is not None else Path(".sp-index/index.db")
    if not db_path.is_file():
        raise SystemExit(f"no index at {db_path}; run `sp index` first")

    probes: dict[str, list[str]] = {}
    for query in args.queries:
        exemplars: list[str] = []
        probes[query] = exemplars
    if len(probes) == 0:
        probes = PROBES
    with Store(db_path) as store:
        store.require_model(MODEL_ID)
        model = load_model()
        print(f"# db={db_path}")
        for query, exemplars in probes.items():
            ranked = rank_full(store, model, query=query)
            report(ranked, query=query, exemplars=exemplars)


if __name__ == "__main__":
    main()
