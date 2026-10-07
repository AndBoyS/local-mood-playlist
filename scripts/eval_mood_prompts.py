"""Read-only mood-prompt evaluation over indexed tracks.

Usage:
    uv run python scripts/eval_mood_prompts.py \
        --track "Robot Waiting Room #1" --out-prefix temp/robot-mood
    uv run python scripts/eval_mood_prompts.py \
        --track "Robot Waiting Room #1" --track "Another track" \
        --labels mood-labels.csv --out-prefix temp/mood-pilot

Optional labels CSV columns: rel_path,valence,arousal. Ratings must be in [1, 5];
CSV-quote rel_path values containing commas.
The script computes prompt vectors in memory and writes window-level and
track-level CSVs; label correlations are written when labels are supplied.
It does not populate the vector cache or change production ranking code.
"""

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.audio.windowing import WINDOW_SECONDS
from local_smart_playlist.embed.model import MODEL_ID, MuLanEmbedder, cap_torch_threads, load_model
from local_smart_playlist.index.store import Store, TrackData
from local_smart_playlist.query.axis_prompts import AXIS_TO_PROMPTS, QueryAxis
from local_smart_playlist.query.contrast import query_vector_contrast
from local_smart_playlist.query.prompt_eval import (
    MoodPromptVariant,
    build_mood_prompt_variants,
    pearson_correlation,
)
from local_smart_playlist.query.prompts import calibration_prompt_bank
from local_smart_playlist.query.rank_text import axis_vector_banks
from local_smart_playlist.query.upper_cluster import upper_cluster_fit_per_window
from local_smart_playlist.query.vocab_cal import vocab_vector_bank

W = IntVar("W")
A = IntVar("A")
D = IntVar("D")
MIN_RATING = 1.0
MAX_RATING = 5.0


@dataclass(frozen=True)
class MoodRatings:
    """Human ratings for one track, each on a five-point scale."""

    valence: float
    arousal: float


@dataclass(frozen=True)
class TrackPromptScores:
    """Track-level measurements for one prompt variant."""

    rel_path: str
    title: str
    duration: float
    window_count: int
    term: str
    template_id: str
    prompt: str
    window_mean_query_cosine: float
    track_mean_vector_cosine: float
    mean_anchor_cosine: float
    top_five_anchor_mean_cosine: float
    top_anchor_prompt: str
    top_anchor_cosine: float
    upper_cluster_fit: float
    same_text_anchor_embedding_cosine: float | None


def _rating(value: str | None, *, column: str, rel_path: str) -> float:
    if value is None or value.strip() == "":
        raise ValueError(f"missing {column} rating for {rel_path!r}")
    try:
        rating = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid {column} rating for {rel_path!r}: {value!r}") from exc
    if not isfinite(rating) or not MIN_RATING <= rating <= MAX_RATING:
        raise ValueError(f"{column} rating for {rel_path!r} must be finite and in [1, 5]")
    return rating


def _read_labels(path: Path) -> dict[str, MoodRatings]:
    required_columns = {"rel_path", "valence", "arousal"}
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        if fieldnames is None or not required_columns.issubset(fieldnames):
            raise ValueError("labels CSV must have columns: rel_path,valence,arousal")
        labels: dict[str, MoodRatings] = {}
        for row in reader:
            assert isinstance(row, dict)
            raw_path = row.get("rel_path")
            assert raw_path is None or isinstance(raw_path, str)
            if raw_path is None or raw_path.strip() == "":
                raise ValueError("labels CSV contains an empty rel_path")
            rel_path = raw_path.strip()
            if rel_path in labels:
                raise ValueError(f"duplicate labels for {rel_path!r}")
            raw_valence = row.get("valence")
            raw_arousal = row.get("arousal")
            assert raw_valence is None or isinstance(raw_valence, str)
            assert raw_arousal is None or isinstance(raw_arousal, str)
            labels[rel_path] = MoodRatings(
                valence=_rating(raw_valence, column="valence", rel_path=rel_path),
                arousal=_rating(raw_arousal, column="arousal", rel_path=rel_path),
            )
    if len(labels) == 0:
        raise ValueError("labels CSV contains no track ratings")
    return labels


def _resolve_tracks[D2: IntVar](
    track_rows: list[TrackData[D2]],
    selectors: Sequence[str],
) -> list[TrackData[D2]]:
    selected: list[TrackData[D2]] = []
    selected_paths: set[str] = set()
    for selector in selectors:
        needle = selector.casefold().strip()
        matches = [
            track
            for track in track_rows
            if needle in track.rel_path.casefold() or needle in track.title.casefold()
        ]
        if len(matches) == 0:
            raise ValueError(f"no indexed track matches {selector!r}")
        if len(matches) > 1:
            candidates = ", ".join(track.rel_path for track in matches[:5])
            raise ValueError(f"ambiguous track selector {selector!r}; matches: {candidates}")
        track = matches[0]
        if track.rel_path not in selected_paths:
            selected.append(track)
            selected_paths.add(track.rel_path)
    return selected


def _evaluate_track[W2: IntVar, D2: IntVar, A2: IntVar](
    track: TrackData[D2],
    *,
    windows: np.ndarray[[W2, D2]],
    query_vec: np.ndarray[[D2]],
    anchor_vecs: np.ndarray[[A2, D2]],
    anchor_prompts: Sequence[str],
    anchor_indices: dict[str, int],
    variant: MoodPromptVariant,
) -> tuple[TrackPromptScores, list[float]]:
    if track.vectors is None:
        raise ValueError(f"track vectors missing for {track.rel_path!r}")
    if windows.shape[0] == 0:
        raise ValueError(f"track has no stored windows: {track.rel_path!r}")

    query_window_sims = (windows @ query_vec).astype(np.float64)
    anchor_window_sims = (windows @ anchor_vecs.T).astype(np.float64)
    query_mean = float(query_window_sims.mean())
    anchor_means = anchor_window_sims.mean(axis=0)
    anchor_matrix = np.stack([anchor_means])
    query_row: np.ndarray[[1]] = np.asarray([query_mean], dtype=np.float64)
    fit = float(upper_cluster_fit_per_window(query_row, anchor_matrix)[0])

    top_anchor_index = int(anchor_means.argmax())
    top_five = sorted(float(sim) for sim in anchor_means)[-5:]
    anchor_index = anchor_indices.get(variant.prompt)
    same_text_cosine = None if anchor_index is None else float(query_vec @ anchor_vecs[anchor_index])
    result = TrackPromptScores(
        rel_path=track.rel_path,
        title=track.title,
        duration=track.duration,
        window_count=windows.shape[0],
        term=variant.term,
        template_id=variant.template_id,
        prompt=variant.prompt,
        window_mean_query_cosine=query_mean,
        track_mean_vector_cosine=float(track.vectors.mean_vec @ query_vec),
        mean_anchor_cosine=float(anchor_means.mean()),
        top_five_anchor_mean_cosine=sum(top_five) / len(top_five),
        top_anchor_prompt=anchor_prompts[top_anchor_index],
        top_anchor_cosine=float(anchor_means[top_anchor_index]),
        upper_cluster_fit=fit,
        same_text_anchor_embedding_cosine=same_text_cosine,
    )
    return result, [float(sim) for sim in query_window_sims]


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        _ = writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    result: TrackPromptScores,
    ratings: MoodRatings | None,
) -> dict[str, str]:
    return {
        "rel_path": result.rel_path,
        "title": result.title,
        "duration_seconds": f"{result.duration:.6f}",
        "window_count": str(result.window_count),
        "term": result.term,
        "template_id": result.template_id,
        "prompt": result.prompt,
        "window_mean_query_cosine": f"{result.window_mean_query_cosine:.9f}",
        "track_mean_vector_cosine": f"{result.track_mean_vector_cosine:.9f}",
        "mean_mood_anchor_cosine": f"{result.mean_anchor_cosine:.9f}",
        "top_five_anchor_mean_cosine": f"{result.top_five_anchor_mean_cosine:.9f}",
        "top_anchor_prompt": result.top_anchor_prompt,
        "top_anchor_cosine": f"{result.top_anchor_cosine:.9f}",
        "current_upper_cluster_fit": f"{result.upper_cluster_fit:.9f}",
        "same_text_anchor_embedding_cosine": (
            "" if result.same_text_anchor_embedding_cosine is None else f"{result.same_text_anchor_embedding_cosine:.9f}"
        ),
        "valence_rating": "" if ratings is None else f"{ratings.valence:.3f}",
        "arousal_rating": "" if ratings is None else f"{ratings.arousal:.3f}",
    }


def _label_metric_rows(
    results: Sequence[TrackPromptScores],
    labels: dict[str, MoodRatings],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    dimensions = ("valence", "arousal")
    metrics = ("window_mean_query_cosine", "current_upper_cluster_fit")
    variants = list(dict.fromkeys((result.term, result.template_id, result.prompt) for result in results))
    for term, template_id, prompt in variants:
        variant_results = [
            result for result in results if result.term == term and result.template_id == template_id
        ]
        for metric_name in metrics:
            for dimension in dimensions:
                pairs: list[tuple[float, float]] = []
                for result in variant_results:
                    ratings = labels.get(result.rel_path)
                    if ratings is None:
                        continue
                    score = (
                        result.window_mean_query_cosine
                        if metric_name == "window_mean_query_cosine"
                        else result.upper_cluster_fit
                    )
                    rating = ratings.valence if dimension == "valence" else ratings.arousal
                    pairs.append((score, rating))
                correlation = pearson_correlation(
                    [score for score, _rating_value in pairs],
                    [rating_value for _score, rating_value in pairs],
                )
                rows.append(
                    {
                        "term": term,
                        "template_id": template_id,
                        "prompt": prompt,
                        "score_metric": metric_name,
                        "label_dimension": dimension,
                        "n_tracks": str(len(pairs)),
                        "pearson_r": "" if correlation is None else f"{correlation:.9f}",
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--track", action="append", required=True, help="unique indexed path/title substring; repeatable")
    _ = parser.add_argument("--db", type=Path, default=Path(".sp-index/index.db"), help="index DB (read-only evaluation)")
    _ = parser.add_argument("--labels", type=Path, default=None, help="optional CSV: rel_path,valence,arousal, ratings 1-5")
    _ = parser.add_argument("--out-prefix", type=Path, required=True, help="output prefix for CSV files")
    _ = parser.add_argument("--term", action="append", default=None, help="mood term to include; repeatable; default: all anchors + extras")
    _ = parser.add_argument(
        "--template",
        action="append",
        choices=("mood", "paper", "natural"),
        default=None,
        help="prompt template to test; repeatable; default: all three",
    )
    args = parser.parse_args()
    assert isinstance(args.db, Path)
    assert isinstance(args.out_prefix, Path)
    assert args.labels is None or isinstance(args.labels, Path)
    assert isinstance(args.track, list)
    selectors: list[str] = []
    for selector in args.track:
        assert isinstance(selector, str)
        selectors.append(selector)
    terms: list[str] | None = None
    if args.term is not None:
        assert isinstance(args.term, list)
        terms = []
        for term in args.term:
            assert isinstance(term, str)
            terms.append(term)
    template_ids: list[str] | None = None
    if args.template is not None:
        assert isinstance(args.template, list)
        template_ids = []
        for template_id in args.template:
            assert isinstance(template_id, str)
            template_ids.append(template_id)

    db_path: Path = args.db
    labels_path: Path | None = args.labels
    out_prefix: Path = args.out_prefix
    if not db_path.is_file():
        raise SystemExit(f"no index at {db_path}; run `sp index` first")
    labels: dict[str, MoodRatings] = {} if labels_path is None else _read_labels(labels_path)

    cap_torch_threads()
    with Store(db_path) as store:
        store.require_model(MODEL_ID)
        selected_tracks = _resolve_tracks(store.track_meta(), selectors)
        selected_paths = [track.rel_path for track in selected_tracks]
        tracks_with_vectors = store.get_tracks(selected_paths)
        windows_by_path = store.load_windows(selected_paths)
        missing_labels = [path for path in selected_paths if labels_path is not None and path not in labels]
        if len(missing_labels) > 0:
            raise ValueError(f"labels CSV missing selected track(s): {', '.join(missing_labels)}")

        model: MuLanEmbedder[512] = load_model(device="cpu", text_only=True)
        prompt_bank = calibration_prompt_bank()
        full_anchor_vecs = vocab_vector_bank(model, prompt_bank)
        mood_anchor_vecs = axis_vector_banks(full_anchor_vecs)[QueryAxis.MOOD]
        mood_prompts = AXIS_TO_PROMPTS[QueryAxis.MOOD]
        anchor_indices = {prompt: index for index, prompt in enumerate(mood_prompts)}
        variants = build_mood_prompt_variants(
            mood_prompts,
            terms=terms,
            template_ids=template_ids,
        )
        query_vectors = {
            (variant.template_id, variant.term): query_vector_contrast(variant.prompt, model)
            for variant in variants
        }

        summary_results: list[TrackPromptScores] = []
        summary_rows: list[dict[str, str]] = []
        window_rows: list[dict[str, str]] = []
        for track_stub in selected_tracks:
            track = tracks_with_vectors.get(track_stub.rel_path)
            windows = windows_by_path.get(track_stub.rel_path)
            if track is None or windows is None:
                raise ValueError(f"track has no stored vectors/windows: {track_stub.rel_path!r}")
            if track.vectors is None:
                raise ValueError(f"track mean vector missing: {track_stub.rel_path!r}")
            for variant in variants:
                query_vec = query_vectors[(variant.template_id, variant.term)]
                result, window_sims = _evaluate_track(
                    track,
                    windows=windows,
                    query_vec=query_vec,
                    anchor_vecs=mood_anchor_vecs,
                    anchor_prompts=mood_prompts,
                    anchor_indices=anchor_indices,
                    variant=variant,
                )
                summary_results.append(result)
                summary_rows.append(_summary_row(result, labels.get(track.rel_path)))
                for window_index, cosine in enumerate(window_sims, start=1):
                    window_rows.append(
                        {
                            "rel_path": track.rel_path,
                            "title": track.title,
                            "window_index": str(window_index),
                            "window_start_seconds_post_trim": f"{(window_index - 1) * WINDOW_SECONDS:.3f}",
                            "window_end_seconds_post_trim": f"{window_index * WINDOW_SECONDS:.3f}",
                            "term": variant.term,
                            "template_id": variant.template_id,
                            "prompt": variant.prompt,
                            "audio_window_query_cosine": f"{cosine:.9f}",
                            "track_window_mean_query_cosine": f"{result.window_mean_query_cosine:.9f}",
                            "current_upper_cluster_fit": f"{result.upper_cluster_fit:.9f}",
                        }
                    )

    prefix = out_prefix
    track_output = prefix.with_name(f"{prefix.name}.tracks.csv")
    window_output = prefix.with_name(f"{prefix.name}.windows.csv")
    _write_csv(
        track_output,
        fieldnames=(
            "rel_path",
            "title",
            "duration_seconds",
            "window_count",
            "term",
            "template_id",
            "prompt",
            "window_mean_query_cosine",
            "track_mean_vector_cosine",
            "mean_mood_anchor_cosine",
            "top_five_anchor_mean_cosine",
            "top_anchor_prompt",
            "top_anchor_cosine",
            "current_upper_cluster_fit",
            "same_text_anchor_embedding_cosine",
            "valence_rating",
            "arousal_rating",
        ),
        rows=summary_rows,
    )
    _write_csv(
        window_output,
        fieldnames=(
            "rel_path",
            "title",
            "window_index",
            "window_start_seconds_post_trim",
            "window_end_seconds_post_trim",
            "term",
            "template_id",
            "prompt",
            "audio_window_query_cosine",
            "track_window_mean_query_cosine",
            "current_upper_cluster_fit",
        ),
        rows=window_rows,
    )
    print(f"tracks: {track_output}")
    print(f"windows: {window_output}")
    if labels_path is not None:
        metrics_output = prefix.with_name(f"{prefix.name}.label_metrics.csv")
        _write_csv(
            metrics_output,
            fieldnames=("term", "template_id", "prompt", "score_metric", "label_dimension", "n_tracks", "pearson_r"),
            rows=_label_metric_rows(summary_results, labels),
        )
        print(f"label metrics: {metrics_output}")
    print(f"scored {len(selected_tracks)} track(s), {len(variants)} prompt variant(s), {len(window_rows)} windows")


if __name__ == "__main__":
    main()
