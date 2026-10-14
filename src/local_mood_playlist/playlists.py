"""Rank cached track predictions and render extended M3U playlists."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from local_mood_playlist.moods import BlendMode, MoodName
from local_mood_playlist.paths import RelativePath, ResolvedPath
from local_mood_playlist.predictions import PredictionCache


@dataclass(frozen=True)
class RankedTrack:
    audio_path: ResolvedPath
    score: float


def blend_scores(scores: Sequence[float], mode: BlendMode) -> float:
    """Combine nonempty probability list into one score per blend mode."""
    match mode:
        case "mean":
            return math.fsum(scores) / len(scores)
        case "and":
            return min(scores)
        case "or":
            return max(scores)


def rank_cached_tracks(
    cache: PredictionCache,
    *,
    moods: Sequence[MoodName],
    mode: BlendMode,
    limit: int | None,
    min_score: float,
) -> list[RankedTrack]:
    """Rank fresh cache entries by blended probability across requested moods."""
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
        score = blend_scores([prediction.mood_scores[mood] for mood in moods], mode)
        ranked.append(RankedTrack(audio_path, score))

    ranked.sort(key=lambda track: (-track.score, track.audio_path.path.as_posix()))
    ranked = [track for track in ranked if track.score >= min_score]
    if limit is None:
        return ranked
    return ranked[:limit]


def render_m3u8(tracks: list[RankedTrack], *, output_path: ResolvedPath | None = None) -> str:
    """Render extended M3U text with track entries relative to the playlist's own directory.

    Relative entries keep the playlist playing on another device where the
    playlist and the library keep the same layout; the playlist can live
    anywhere inside or beside the library (at the library root itself the
    entries are simply ``Album/song.mp3``). Without an output path (rendering
    to stdout) the entries cannot be positional and stay absolute.
    """
    lines = ["#EXTM3U"]
    for track in tracks:
        audio_path = track.audio_path.path
        title = f"{audio_path.parent.name} - {audio_path.stem}"
        title = title.replace("\r", " ").replace("\n", " ")
        lines.extend((f"#EXTINF:-1,{title}", _playlist_entry(audio_path, output_path).as_posix()))
    return "\n".join(lines) + "\n"


def _playlist_entry(audio_path: Path, output_path: ResolvedPath | None) -> RelativePath | Path:
    """Return one track path relative to the playlist directory, or absolute.

    Only rendering to stdout (no output path) and Windows-cross-drive audio
    paths produce absolute entries; ``walk_up`` entries stay relative for any
    shared layout between playlist and library.
    """
    if output_path is None:
        return audio_path
    with contextlib.suppress(ValueError):
        return RelativePath(audio_path.relative_to(output_path.path.parent, walk_up=True))
    return audio_path
