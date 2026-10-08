"""Rank cached track predictions and render extended M3U playlists."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from local_smart_playlist.predictions import PredictionCache


@dataclass(frozen=True)
class RankedTrack:
    audio_path: Path
    score: float


def rank_cached_tracks(
    cache: PredictionCache,
    mood_tags: tuple[str, ...],
    *,
    limit: int | None,
    min_score: float,
) -> list[RankedTrack]:
    """Rank fresh cache entries by mean probability across requested mood tags."""
    if len(mood_tags) == 0:
        raise ValueError("at least one mood tag is required")
    if limit is not None and limit < 1:
        raise ValueError("playlist size must be at least 1")
    if not 0 < min_score <= 1:
        raise ValueError("minimum score must be in (0, 1]")

    predictions = list(cache.iter_predictions())
    if len(predictions) == 0:
        raise ValueError("prediction cache contains no fresh indexed tracks")

    ranked: list[RankedTrack] = []
    for audio_path, prediction in predictions:
        missing_tags = [tag for tag in mood_tags if tag not in prediction.mood_scores]
        if len(missing_tags) > 0:
            raise ValueError(
                f"cached prediction for {audio_path} is missing mood tag(s): {', '.join(missing_tags)}"
            )
        score = math.fsum(prediction.mood_scores[tag] for tag in mood_tags) / len(mood_tags)
        ranked.append(RankedTrack(audio_path, score))

    ranked.sort(key=lambda track: (-track.score, track.audio_path.as_posix()))
    ranked = [track for track in ranked if track.score >= min_score]
    if limit is None:
        return ranked
    return ranked[:limit]


def render_m3u8(tracks: list[RankedTrack]) -> str:
    """Render ranked track paths as UTF-8-compatible extended M3U text."""
    lines = ["#EXTM3U"]
    for track in tracks:
        title = f"{track.audio_path.parent.name} - {track.audio_path.stem}"
        title = title.replace("\r", " ").replace("\n", " ")
        lines.extend((f"#EXTINF:-1,{title}", track.audio_path.as_posix()))
    return "\n".join(lines) + "\n"
