"""Rank cached track predictions and render extended M3U playlists."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from local_mood_playlist.moods import MoodName
from local_mood_playlist.predictions import PredictionCache


@dataclass(frozen=True)
class RankedTrack:
    audio_path: Path
    score: float


def rank_cached_tracks(
    cache: PredictionCache,
    *,
    moods: Sequence[MoodName],
    limit: int | None,
    min_score: float,
) -> list[RankedTrack]:
    """Rank fresh cache entries by mean probability across requested moods."""
    if len(moods) == 0:
        raise ValueError("at least one mood is required")
    if limit is not None and limit < 1:
        raise ValueError("playlist size must be at least 1")
    if not 0 < min_score <= 1:
        raise ValueError("minimum score must be in (0, 1]")

    predictions = list(cache.iter_predictions())
    if len(predictions) == 0:
        raise ValueError("prediction cache contains no fresh indexed tracks")

    ranked: list[RankedTrack] = []
    for audio_path, prediction in predictions:
        missing_moods = [mood for mood in moods if mood not in prediction.mood_scores]
        if len(missing_moods) > 0:
            raise ValueError(f"cached prediction for {audio_path} is missing mood(s): {', '.join(missing_moods)}")
        score = math.fsum(prediction.mood_scores[mood] for mood in moods) / len(moods)
        ranked.append(RankedTrack(audio_path, score))

    ranked.sort(key=lambda track: (-track.score, track.audio_path.as_posix()))
    ranked = [track for track in ranked if track.score >= min_score]
    if limit is None:
        return ranked
    return ranked[:limit]


def render_m3u8(tracks: list[RankedTrack], *, output_path: Path | None = None) -> str:
    """Render extended M3U text, relative to output playlist when known."""
    lines = ["#EXTM3U"]
    for track in tracks:
        title = f"{track.audio_path.parent.name} - {track.audio_path.stem}"
        title = title.replace("\r", " ").replace("\n", " ")
        audio_path = track.audio_path.expanduser().resolve()
        if output_path is None:
            playlist_entry = audio_path.as_posix()
        else:
            try:
                playlist_entry = audio_path.relative_to(
                    output_path.expanduser().resolve().parent,
                    walk_up=True,
                ).as_posix()
            except ValueError:
                playlist_entry = audio_path.as_posix()
        lines.extend((f"#EXTINF:-1,{title}", playlist_entry))
    return "\n".join(lines) + "\n"
