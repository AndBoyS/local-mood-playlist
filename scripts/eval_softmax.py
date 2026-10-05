"""Compare upper-cluster and softmax fits with global-bank percentile ranking.

Read-only live-index evaluation. By default, global baseline uses current
caption and all-axis prompts. Pass historical pre-axis references to recreate
that bank; upper-cluster and softmax comparisons use current mood anchors.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.const import CAPTION_VOCAB_PATH
from local_smart_playlist.embed.model import MODEL_ID, cap_torch_threads, load_model
from local_smart_playlist.index.store import Store, TrackData
from local_smart_playlist.query.axis_prompts import ALL_AXIS_PROMPTS, AXIS_TO_PROMPTS, QueryAxis
from local_smart_playlist.query.contrast import query_vector_contrast
from local_smart_playlist.query.prompts import read_prompt_bank
from local_smart_playlist.query.upper_cluster import (
    UPPER_CLUSTER_MIN_GAP_RATIO,
    UPPER_CLUSTER_MIN_TAIL_SIZE,
    UPPER_CLUSTER_TOP_SIZE,
    upper_cluster_fit_per_window,
)
from local_smart_playlist.type_utils import NonEmptyTuple

D = IntVar("D")  # embedding dim

DEFAULT_QUERY = "dreamy, ethereal"
DEFAULT_TEMPERATURES = (0.02, 0.05, 0.1)
NEUTRAL_FIT = 0.5
LEGACY_PASS_THRESHOLD = 0.9
COMPARISON_SOFTMAX_CUTOFF = 0.3
EXEMPLARS = ("To Far Shores", "The Terminal Show")
PREVIEW_COUNT = 5


def softmax_fit_per_window[W: IntVar, A: IntVar](
    query_sims: np.ndarray[[W]], anchor_sims: np.ndarray[[W, A]], *, temperature: float
) -> np.ndarray[[W]]:
    """Query softmax mass against anchors, computed in float64 log space."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if anchor_sims.shape[1] == 0:
        raise ValueError("softmax anchor bank must be non-empty")

    logits = (anchor_sims - query_sims[:, None]) / temperature
    row_max = logits.max(axis=1)
    exp_logits: np.ndarray[[W, A]] = np.exp(logits - row_max[:, None])
    log_anchor_mass = row_max + np.log(exp_logits.sum(axis=1))
    fits: np.ndarray[[W]] = np.exp(-np.logaddexp(0.0, log_anchor_mass))
    return fits


def percentile_fit_per_window[W: IntVar, A: IntVar](
    query_sims: np.ndarray[[W]], anchor_sims: np.ndarray[[W, A]]
) -> np.ndarray[[W]]:
    """Legacy per-window query percentile among anchor similarities."""
    return np.greater(query_sims[:, None], anchor_sims).mean(axis=1)


def _matches[D2: IntVar](track: TrackData[D2], needle: str) -> bool:
    needle_l = needle.casefold()
    return needle_l in track.rel_path.casefold() or needle_l in track.title.casefold()


def _nearest_anchor_summary[N: IntVar](
    label: str, *, prompts: NonEmptyTuple[str], similarities: np.ndarray[[N]]
) -> str:
    nearest = sorted(range(len(prompts)), key=lambda i: float(similarities[i]), reverse=True)[:3]
    return f"# nearest {label} anchors: " + " | ".join(
        f"{prompts[i]}={float(similarities[i]):.3f}" for i in nearest
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    _ = parser.add_argument("--db", type=Path, default=Path(".sp-index/index.db"))
    _ = parser.add_argument(
        "--save-upper-cluster",
        type=Path,
        help="write full ranking CSV ordered by raw upper-cluster score",
    )
    _ = parser.add_argument(
        "--pre-axis-references",
        type=Path,
        help="historical calibration_reference_prompts.txt; combines with current caption vocabulary",
    )
    args = parser.parse_args()

    cap_torch_threads()
    if not args.db.is_file():
        raise SystemExit(f"no index at {args.db}; run `sp index` first")
    if args.pre_axis_references is not None and not args.pre_axis_references.is_file():
        raise SystemExit(f"pre-axis reference file not found: {args.pre_axis_references}")

    caption_prompts = read_prompt_bank(CAPTION_VOCAB_PATH)
    if args.pre_axis_references is None:
        global_prompts = NonEmptyTuple((*caption_prompts, *ALL_AXIS_PROMPTS))
        global_label = "unrestricted current all-axis bank"
    else:
        old_references = read_prompt_bank(args.pre_axis_references)
        global_prompts = NonEmptyTuple((*caption_prompts, *old_references))
        global_label = "pre-axis global bank"

    mood_prompts = AXIS_TO_PROMPTS[QueryAxis.MOOD]
    cap_text = len(caption_prompts)
    with Store(args.db) as store:
        store.require_model(MODEL_ID)
        model = load_model(device="cpu", text_only=True)
        query_vec = query_vector_contrast(args.query, model)
        global_vecs = model.embed_texts(global_prompts)
        mood_vecs = model.embed_texts(mood_prompts)
        global_text_sims = np.asarray(global_vecs @ query_vec, dtype=np.float64)
        mood_text_sims = np.asarray(mood_vecs @ query_vec, dtype=np.float64)

        track_paths = store.track_rel_paths()
        windows_by_path = store.load_windows(track_paths)
        tracks: list[tuple[TrackData[Any], np.ndarray]] = []
        for rel_path in track_paths:
            track = store.get_track(rel_path)
            windows = windows_by_path.get(rel_path)
            if track is not None and windows is not None and windows.shape[0] > 0:
                tracks.append((track, windows))
        if len(tracks) == 0:
            raise SystemExit(f"no indexed tracks with stored windows in {args.db}")

        print(f"# db={args.db}  tracks={len(tracks)}  query={args.query!r}")
        print(
            f"# anchors: {global_label}={len(global_prompts)} "
            f"(captions={cap_text}); mood-axis={len(mood_prompts)}"
        )
        print(_nearest_anchor_summary("global", prompts=global_prompts, similarities=global_text_sims))
        print(_nearest_anchor_summary("mood", prompts=mood_prompts, similarities=mood_text_sims))

        scores: dict[str, list[float]] = {}
        global_legacy_scores: list[float] = []
        global_legacy_coverages: list[float] = []
        upper_cluster_query_means: list[float] = []
        for _track, windows in tracks:
            q_sims = np.asarray(windows @ query_vec, dtype=np.float64).ravel()
            global_sims = np.asarray(windows @ global_vecs.T, dtype=np.float64)
            mood_sims = np.asarray(windows @ mood_vecs.T, dtype=np.float64)

            legacy_windows = percentile_fit_per_window(q_sims, global_sims)
            global_legacy_scores.append(float(legacy_windows.mean()))
            global_legacy_coverages.append(float(np.greater(legacy_windows, NEUTRAL_FIT).mean()))
            track_query_sim = np.asarray([q_sims.mean()], dtype=np.float64)
            track_mood_sims = mood_sims.mean(axis=0)[None, :]
            upper_cluster_query_means.append(float(track_query_sim[0]))
            upper_fit = upper_cluster_fit_per_window(track_query_sim, track_mood_sims)
            upper_name = "mood upper-cluster"
            scores.setdefault(upper_name, []).append(float(upper_fit[0]))
            for bank_name, bank_sims in (
                ("global", global_sims),
                ("mood", mood_sims),
            ):
                for temperature in DEFAULT_TEMPERATURES:
                    name = f"{bank_name} softmax T={temperature:.2f}"
                    values = softmax_fit_per_window(q_sims, bank_sims, temperature=temperature)
                    scores.setdefault(name, []).append(float(values.mean()))

        labels = [global_label + " percentile", *scores]
        score_sets = {labels[0]: global_legacy_scores, **scores}
        orderings: dict[str, list[int]] = {}
        for name, values in score_sets.items():
            if name == labels[0]:
                order = sorted(
                    range(len(values)),
                    key=lambda i: (-values[i], -global_legacy_coverages[i], tracks[i][0].rel_path),
                )
            elif name == "mood upper-cluster":
                order = sorted(
                    range(len(values)),
                    key=lambda i: (-values[i], -upper_cluster_query_means[i], tracks[i][0].rel_path),
                )
            else:
                order = sorted(range(len(values)), key=lambda i: (-values[i], tracks[i][0].rel_path))
            orderings[name] = order
            arr = np.asarray(values, dtype=np.float64)
            quantiles = np.percentile(arr, (50, 90, 95)).tolist()
            if name == labels[0]:
                pass_rate = float(np.greater_equal(arr, LEGACY_PASS_THRESHOLD).mean())
                threshold_text = f"pass@0.90={pass_rate:.1%}"
            elif name.startswith("mood upper-cluster"):
                pass_half = float(np.greater(arr, NEUTRAL_FIT).mean())
                threshold_text = f"above neutral >0.50={pass_half:.1%}"
            else:
                pass_half = float(np.greater_equal(arr, NEUTRAL_FIT).mean())
                pass_nine = float(np.greater_equal(arr, LEGACY_PASS_THRESHOLD).mean())
                threshold_text = f"pass@0.50={pass_half:.1%}, @0.90={pass_nine:.1%}"
            print(
                f"\n[{name}] median/p90/p95={quantiles[0]:.4f}/{quantiles[1]:.4f}/{quantiles[2]:.4f}; "
                f"{threshold_text}"
            )
            for needle in EXEMPLARS:
                hits = [idx for idx, (track, _windows) in enumerate(tracks) if _matches(track, needle)]
                if len(hits) > 0:
                    print(
                        f"  {needle}: "
                        + ", ".join(
                            f"rank {order.index(hit) + 1} score={values[hit]:.4f}"
                            for hit in hits[:3]
                        )
                    )
            print("  top: " + " | ".join(tracks[i][0].rel_path for i in order[:PREVIEW_COUNT]))

        print(
            f"\nUpper-cluster fit uses track-mean similarities, then adaptively trims the top {UPPER_CLUSTER_TOP_SIZE} "
            f"anchors (gap ratio>={UPPER_CLUSTER_MIN_GAP_RATIO:.1f}, "
            f"at least {UPPER_CLUSTER_MIN_TAIL_SIZE} anchors per tail group); score-1 ties use query mean."
        )
        baseline_order = orderings[labels[0]]
        baseline_top = set(baseline_order[:20])
        print(f"\nTop-20 overlap with {labels[0]} baseline:")
        for name, order in orderings.items():
            if name == labels[0]:
                continue
            overlap = len(baseline_top.intersection(order[:20]))
            print(f"  {name}: {overlap}/20")

        baseline_members = {
            i for i, score in enumerate(global_legacy_scores) if score >= LEGACY_PASS_THRESHOLD
        }
        mood_variant = f"mood softmax T={min(DEFAULT_TEMPERATURES):.2f}"
        mood_values = scores[mood_variant]
        mood_members = {i for i, score in enumerate(mood_values) if score >= COMPARISON_SOFTMAX_CUTOFF}
        shared = baseline_members.intersection(mood_members)
        print(
            "\n# Playlist diff compares complete recipes; score cutoffs are not equivalent "
            "probabilities across banks/scorers."
        )
        print(
            f"Playlist diff ({labels[0]} >= {LEGACY_PASS_THRESHOLD:.2f} vs "
            f"{mood_variant} >= {COMPARISON_SOFTMAX_CUTOFF:.2f}): "
            f"old={len(baseline_members)} new={len(mood_members)} shared={len(shared)} "
            f"removed={len(baseline_members) - len(shared)} added={len(mood_members) - len(shared)}"
        )
        old_only_order = [i for i in baseline_order if i in baseline_members - shared]
        mood_order = orderings[mood_variant]
        new_only_order = [i for i in mood_order if i in mood_members - shared]
        print("Removed from playlist:")
        for i in old_only_order:
            print(f"- {tracks[i][0].rel_path}  old={global_legacy_scores[i]:.3f} new={mood_values[i]:.3f}")
        print("Added to playlist:")
        for i in new_only_order:
            print(f"+ {tracks[i][0].rel_path}  old={global_legacy_scores[i]:.3f} new={mood_values[i]:.3f}")

        upper_variant = "mood upper-cluster"
        upper_values = scores[upper_variant]
        upper_members = {i for i, score in enumerate(upper_values) if score > NEUTRAL_FIT}
        upper_shared = baseline_members.intersection(upper_members)
        print(
            f"\nUpper-cluster recipe diff ({labels[0]} >= {LEGACY_PASS_THRESHOLD:.2f} vs "
            f"{upper_variant} > {NEUTRAL_FIT:.2f}): old={len(baseline_members)} "
            f"new={len(upper_members)} shared={len(upper_shared)} "
            f"removed={len(baseline_members) - len(upper_shared)} "
            f"added={len(upper_members) - len(upper_shared)}"
        )
        upper_removed = [i for i in baseline_order if i in baseline_members - upper_shared]
        upper_order = orderings[upper_variant]
        upper_added = [i for i in upper_order if i in upper_members - upper_shared]
        print("  removed examples: " + " | ".join(tracks[i][0].rel_path for i in upper_removed[:5]))
        print("  added examples: " + " | ".join(tracks[i][0].rel_path for i in upper_added[:5]))
        if args.save_upper_cluster is not None:
            args.save_upper_cluster.parent.mkdir(parents=True, exist_ok=True)
            output: TextIO
            with args.save_upper_cluster.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                _ = writer.writerow(("rank", "rel_path", "title", "upper_cluster_score", "query_mean"))
                for rank, index in enumerate(upper_order, start=1):
                    track = tracks[index][0]
                    _ = writer.writerow(
                        (rank, track.rel_path, track.title, upper_values[index], upper_cluster_query_means[index])
                    )
            print(f"\nSaved upper-cluster ranking: {args.save_upper_cluster} ({len(upper_order)} tracks).")


if __name__ == "__main__":
    main()
