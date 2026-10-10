"""CLI shell for fixed-vocabulary mood playlists."""

import sys
from pathlib import Path
from time import monotonic
from typing import NamedTuple

import questionary
import typed_argparse as tap

from local_mood_playlist.indexing import ProgressUpdate, default_cache_path, index_library
from local_mood_playlist.moods import MoodName
from local_mood_playlist.music2emotion import MODEL_VERSION
from local_mood_playlist.playlists import rank_cached_tracks, render_m3u8
from local_mood_playlist.predictions import PredictionCache


class MoodEntry(NamedTuple):
    mood: MoodName
    color: str
    # Is in cli by default?
    is_core: bool


MOOD_ENTRIES: list[MoodEntry] = [
    MoodEntry(mood="calm", color="#8db9d8", is_core=True),
    MoodEntry(mood="relaxing", color="#8db9d8", is_core=True),
    MoodEntry(mood="meditative", color="#8db9d8", is_core=True),
    MoodEntry(mood="soft", color="#8db9d8", is_core=True),
    MoodEntry(mood="dreamy", color="#8db9d8", is_core=True),
    MoodEntry(mood="ballad", color="#8db9d8", is_core=True),
    MoodEntry(mood="melodic", color="#8db9d8", is_core=True),
    MoodEntry(mood="slow", color="#8db9d8", is_core=True),
    MoodEntry(mood="soundscape", color="#8db9d8", is_core=True),
    MoodEntry(mood="space", color="#8db9d8", is_core=True),
    MoodEntry(mood="deep", color="#7893ab", is_core=True),
    MoodEntry(mood="emotional", color="#7893ab", is_core=True),
    MoodEntry(mood="melancholic", color="#7893ab", is_core=True),
    MoodEntry(mood="sad", color="#7893ab", is_core=True),
    MoodEntry(mood="dark", color="#7893ab", is_core=True),
    MoodEntry(mood="dramatic", color="#7893ab", is_core=True),
    MoodEntry(mood="happy", color="#d8c26a", is_core=True),
    MoodEntry(mood="positive", color="#d8c26a", is_core=True),
    MoodEntry(mood="hopeful", color="#d8c26a", is_core=True),
    MoodEntry(mood="fun", color="#d8c26a", is_core=True),
    MoodEntry(mood="funny", color="#d8c26a", is_core=True),
    MoodEntry(mood="cool", color="#d8c26a", is_core=True),
    MoodEntry(mood="inspiring", color="#d8c26a", is_core=True),
    MoodEntry(mood="uplifting", color="#d8c26a", is_core=True),
    MoodEntry(mood="motivational", color="#d8c26a", is_core=True),
    MoodEntry(mood="energetic", color="#d99a6c", is_core=True),
    MoodEntry(mood="upbeat", color="#d99a6c", is_core=True),
    MoodEntry(mood="groovy", color="#d99a6c", is_core=True),
    MoodEntry(mood="party", color="#d99a6c", is_core=True),
    MoodEntry(mood="fast", color="#d99a6c", is_core=True),
    MoodEntry(mood="action", color="#d99a6c", is_core=True),
    MoodEntry(mood="powerful", color="#cf8585", is_core=True),
    MoodEntry(mood="heavy", color="#cf8585", is_core=True),
    MoodEntry(mood="epic", color="#cf8585", is_core=True),
    MoodEntry(mood="nature", color="#9fc48f", is_core=True),
    MoodEntry(mood="summer", color="#9fc48f", is_core=True),
    MoodEntry(mood="travel", color="#9fc48f", is_core=True),
    MoodEntry(mood="adventure", color="#9fc48f", is_core=True),
    MoodEntry(mood="retro", color="#93a0ac", is_core=True),
    MoodEntry(mood="background", color="#9d8fd9", is_core=True),
    MoodEntry(mood="romantic", color="#d6a0c2", is_core=True),
    MoodEntry(mood="sexy", color="#d6a0c2", is_core=True),
    MoodEntry(mood="love", color="#d6a0c2", is_core=True),
    MoodEntry(mood="children", color="#d8c26a", is_core=False),
    MoodEntry(mood="christmas", color="#d8c26a", is_core=False),
    MoodEntry(mood="holiday", color="#d8c26a", is_core=False),
    MoodEntry(mood="sport", color="#d99a6c", is_core=False),
    MoodEntry(mood="drama", color="#7893ab", is_core=False),
    MoodEntry(mood="documentary", color="#7893ab", is_core=False),
    MoodEntry(mood="movie", color="#9d8fd9", is_core=False),
    MoodEntry(mood="film", color="#9d8fd9", is_core=False),
    MoodEntry(mood="game", color="#9d8fd9", is_core=False),
    MoodEntry(mood="advertising", color="#9d8fd9", is_core=False),
    MoodEntry(mood="commercial", color="#9d8fd9", is_core=False),
    MoodEntry(mood="corporate", color="#9d8fd9", is_core=False),
    MoodEntry(mood="trailer", color="#9d8fd9", is_core=False),
]

_MAX_FAILURES_TO_PRINT = 20

PICKER_STYLE = questionary.Style(
    [
        ("question", "bold"),
        ("pointer", "fg:ansicyan bold"),
        ("highlighted", "fg:ansicyan bold"),
        ("selected", "fg:ansigreen"),
        ("instruction", "fg:ansibrightblack italic"),
    ]
)


class CreateArgs(tap.TypedArgs):
    moods: list[MoodName] = tap.arg(positional=True, nargs="*", help="Mood labels to blend")
    all: bool = tap.arg("-a", "--all", default=False, help="Offer extended model-only mood tags in the dialog")
    count: int | None = tap.arg("-n", default=None, help="Playlist size (default: all ranked tracks)")
    threshold: float = tap.arg(
        "-t", "--threshold", default=0.5, help="Minimum mean mood probability (0-1]; default: 0.5"
    )
    output: str | None = tap.arg(
        "-o", default=None, help="Playlist destination (.m3u8); use {moods} for the selected moods"
    )
    library: str | None = tap.arg(default=None, help="Library root for the default prediction cache")
    cache: str | None = tap.arg(default=None, help="Prediction cache file path (overrides --library)")
    dry_run: bool = tap.arg(default=False, help="Preview ranked tracks when cache exists; never write a playlist")


class IndexArgs(tap.TypedArgs):
    library: str = tap.arg(positional=True, help="Music library directory to analyze")
    cache: str | None = tap.arg(default=None, help="Prediction cache file path")
    force: bool = tap.arg(default=False, help="Recompute predictions for every supported audio file")


def moods_list_cmd(_: tap.TypedArgs) -> None:
    for entry in MOOD_ENTRIES:
        if entry.is_core:
            print(f"  {entry.mood}")
    print()
    print("  Extended model tags (offerable in the dialog via `moodpl create -a`):")
    for entry in MOOD_ENTRIES:
        if not entry.is_core:
            print(f"  {entry.mood}")


def index_cmd(args: IndexArgs) -> None:
    progress_started = monotonic()
    progress_active = False

    def report_progress(update: ProgressUpdate) -> None:
        nonlocal progress_active
        completed, total, audio_path = update
        if total == 0:
            return
        width = 30
        filled = width * completed // total
        bar = "#" * filled + "-" * (width - filled)
        percent = completed * 100 // total
        if completed == 0:
            eta = "ETA calculating"
        else:
            remaining = int((monotonic() - progress_started) / completed * (total - completed))
            eta = f"ETA {remaining // 60:02d}m{remaining % 60:02d}s"
        track_name = "starting" if audio_path is None else audio_path.name[:48]
        progress_line = f"Indexing [{bar}] {percent:3d}% {completed}/{total} {eta} {track_name}"
        if sys.stderr.isatty():
            print(f"\r{progress_line}", end="", file=sys.stderr, flush=True)
            progress_active = True
        else:
            print(progress_line, file=sys.stderr, flush=True)

    try:
        summary = index_library(
            Path(args.library),
            cache_path=Path(args.cache) if args.cache is not None else None,
            force=args.force,
            on_progress=report_progress,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"moodpl index: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    finally:
        if progress_active:
            print(file=sys.stderr, flush=True)

    print(f"Indexed {summary.tracks_found} audio files")
    print(f"  Reused: {summary.predictions_reused}")
    print(f"  Predicted: {summary.predictions_computed}")
    print(f"  Failed: {summary.predictions_failed}")
    for failure in summary.failures[:_MAX_FAILURES_TO_PRINT]:
        print(f"  {failure.audio_path}: {failure.error}", file=sys.stderr)
    if summary.predictions_failed > _MAX_FAILURES_TO_PRINT:
        remaining_failures = summary.predictions_failed - _MAX_FAILURES_TO_PRINT
        print(f"  ... and {remaining_failures} more failures", file=sys.stderr)
    print(f"  Cache: {summary.cache_path}")


def run_mood_picker(show_all: bool = False) -> list[MoodName] | None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("moodpl create: pass mood labels when running non-interactively", file=sys.stderr)
        raise SystemExit(2)

    shown = MOOD_ENTRIES if show_all else [entry for entry in MOOD_ENTRIES if entry.is_core]
    choices = [questionary.Choice(title=[(f"fg:{entry.color}", entry.mood)], value=entry.mood) for entry in shown]
    try:
        selected: list[MoodName] | None = questionary.checkbox(
            "Choose moods to blend",
            choices=choices,
            instruction="(Use arrows/jk to move, Space to toggle, Enter to confirm)",
            style=PICKER_STYLE,
        ).ask()

    except KeyboardInterrupt:
        return None

    if selected is None:
        return None

    return selected


def _validate_playlist_output_path(output_path: Path, cache_path: Path | None) -> None:
    if output_path.suffix.lower() != ".m3u8":
        raise ValueError("playlist output path must end in .m3u8")
    if cache_path is not None and output_path.resolve() == cache_path.resolve():
        raise ValueError("playlist output path must not overwrite prediction cache")


def create_cmd(args: CreateArgs) -> None:
    if args.count is not None and args.count < 1:
        print("moodpl create: --count must be at least 1", file=sys.stderr)
        raise SystemExit(2)
    if args.threshold <= 0 or args.threshold > 1:
        print("moodpl create: --threshold must be in (0, 1]", file=sys.stderr)
        raise SystemExit(2)

    moods = args.moods
    if len(moods) == 0:
        moods = run_mood_picker(show_all=args.all)
        if moods is None or len(moods) == 0:
            print("No moods selected; cancelled.")
            return
    if len(set(moods)) != len(moods):
        print("moodpl create: each mood label can be selected only once", file=sys.stderr)
        raise SystemExit(2)

    output_arg = args.output
    if output_arg is not None and "{moods}" in output_arg:
        output_arg = output_arg.replace("{moods}", "-".join(moods))

    if args.cache is not None:
        cache_path = Path(args.cache).expanduser()
    elif args.library is not None:
        cache_path = default_cache_path(Path(args.library))
    else:
        cache_path = None

    if cache_path is None and not args.dry_run:
        print("moodpl create: pass --cache or --library to use indexed predictions", file=sys.stderr)
        raise SystemExit(2)

    if args.dry_run and cache_path is None:
        print("Playlist request preview (not ranked; no cache selected)")
        print(f"  Moods: {', '.join(moods)}")
        print("  Matching: mean of selected mood probabilities")
        print(f"  Threshold: mean mood probability >= {args.threshold:g}")
        if args.count is None:
            print("  Tracks requested: all ranked tracks")
        else:
            print(f"  Tracks requested: {args.count}")
        if output_arg is not None:
            print(f"  Would write: {output_arg}")
        return

    assert cache_path is not None
    try:
        cache = PredictionCache(cache_path, model_version=MODEL_VERSION)
        ranked = rank_cached_tracks(cache, moods=moods, limit=args.count, min_score=args.threshold)
        output_path = Path(output_arg).expanduser() if output_arg is not None else None
        if output_path is not None:
            _validate_playlist_output_path(output_path, cache.path)

        if args.dry_run:
            print("Playlist preview (not written)")
            print(f"  Moods: {', '.join(moods)}")
            requested = "all" if args.count is None else str(args.count)
            print(f"  Ranked tracks: {len(ranked)} of {requested} requested")
            print(f"  Threshold: mean mood probability >= {args.threshold:g}")
            for index, track in enumerate(ranked, start=1):
                print(f"  {index:2d}. {track.score:.4f} {track.audio_path}")
            if output_path is not None:
                print(f"  Would write: {output_path}")
            return

        playlist = render_m3u8(ranked, output_path=output_path)
        if output_path is None:
            _ = sys.stdout.write(playlist)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _ = output_path.write_text(playlist, encoding="utf-8")
            print(f"Created {len(ranked)}-track playlist: {output_path}", file=sys.stderr)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"moodpl create: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def main() -> None:
    tap.Parser(
        tap.SubParserGroup(
            tap.SubParser("moods", tap.TypedArgs, help="List selectable mood labels"),
            tap.SubParser("create", CreateArgs, help="Create a playlist from selected mood tags"),
            tap.SubParser("index", IndexArgs, help="Analyze library tracks and cache mood predictions"),
            description="Create local playlists from a fixed set of mood tags.",
        ),
        prog="moodpl",
    ).bind(moods_list_cmd, create_cmd, index_cmd).run()


if __name__ == "__main__":
    main()
