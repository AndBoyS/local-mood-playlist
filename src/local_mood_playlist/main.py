"""CLI shell for fixed-vocabulary mood playlists."""

import sys
from pathlib import Path
from time import monotonic
from typing import NamedTuple, NoReturn

import questionary
import typed_argparse as tap

from local_mood_playlist.args import CreateArgs, IndexArgs, SetupArgs
from local_mood_playlist.config import (
    Config,
    config_from_args,
    config_from_user_file,
    config_path,
    dump_raw_config,
    resolve_configs,
)
from local_mood_playlist.indexing import (
    AUDIO_SUFFIXES,
    ProgressUpdate,
    index_library,
)
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


def _fail(cmd: str, msg: str) -> NoReturn:
    print(f"moodpl {cmd}: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _confirm(question: str) -> bool:
    answer: bool | None = questionary.confirm(question, style=PICKER_STYLE).ask()
    if answer is None:
        print("Cancelled.")
        raise SystemExit(0)
    return answer


def _prompt_path(question: str, *, default: str | None = None, only_directories: bool = True) -> str | None:
    prompt_default = default if default is not None else ""
    try:
        answer: str | None = questionary.path(
            question, default=prompt_default, only_directories=only_directories, style=PICKER_STYLE
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    return answer


def require_library_dir(cfg: Config, *, command: str) -> Path:
    if cfg.library_dir is None:
        _fail(command, "no library directory: pass one or run `moodpl setup`")
    return cfg.library_dir


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
        cfg = resolve_configs(config_from_args(args), config_from_user_file())
        library = require_library_dir(cfg, command="index")
        cache_path = cfg.pred_cache_path
        if cache_path is None:
            _fail("index", "configured library has no prediction cache path")
        summary = index_library(
            library,
            cache_path=cache_path,
            cache_root=cfg.cache_dir,
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


def _prompt_library_dir(cmd: str, *, fallback: Path | None = None) -> Path:
    """Prompt for a library directory until it exists; empty answer re-prompts, None cancels."""
    while True:
        answer = _prompt_path("Music library directory", default=str(fallback) if fallback is not None else None)
        if answer is None or len(answer.strip()) == 0:
            if answer is not None:
                print(f"moodpl {cmd}: empty directory", file=sys.stderr)
                continue
            print("Cancelled.")
            raise SystemExit(0)
        candidate = Path(answer).expanduser()
        if candidate.is_dir():
            return candidate
        print(f"moodpl {cmd}: directory does not exist: {candidate}", file=sys.stderr)


def _setup_library_dir(cfg: Config, *, non_interactive: bool) -> Path:
    """Return the library directory for setup; interactive setup may prompt, non-interactive must fail."""
    library = cfg.library_dir
    if library is not None and library.is_dir():
        return library
    if not non_interactive:
        return _prompt_library_dir("setup", fallback=library)
    if library is None:
        _fail("setup", "no library directory: pass --library (terminal not detected)")
    _fail("setup", f"directory does not exist: {library}")


def setup_cmd(args: SetupArgs) -> None:
    non_interactive = not (sys.stdin.isatty() and sys.stdout.isatty())
    try:
        user_cfg = config_from_user_file()
        pre_interactive_cfg = resolve_configs(config_from_args(args), user_cfg)
        library = _setup_library_dir(pre_interactive_cfg, non_interactive=non_interactive)

        prompt_cache_dir: str | None = None
        if args.cache_dir is None and not non_interactive:
            answer = _prompt_path("Cache directory", default=str(pre_interactive_cfg.cache_dir))
            prompt_cache_dir = answer if answer is not None and len(answer.strip()) > 0 else None

        audio_count = sum(path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES for path in library.rglob("*"))
        if audio_count == 0:
            print(f"warning: no supported audio files found yet under {library}", file=sys.stderr)

        cfg_path = config_path()
        if cfg_path.is_file():
            if non_interactive:
                _fail(
                    "setup",
                    f"refusing to overwrite {cfg_path} without a terminal; delete it first or run interactively",
                )
            if not _confirm(f"Overwrite {cfg_path}?"):
                return

        final_args = SetupArgs(
            library=str(library), cache_dir=prompt_cache_dir if prompt_cache_dir is not None else args.cache_dir
        )
        cfg = resolve_configs(config_from_args(final_args), user_cfg)
        payload = dump_raw_config(cfg, cfg_path)
    except (OSError, TypeError, ValueError) as error:
        print(f"moodpl setup: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    print(f"Config written: {cfg_path}")
    for name, value in payload.items():
        print(f"  {name} = {value}")
    print("Next: moodpl index")


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


def _print_request_preview(
    moods: list[MoodName], *, threshold: float, count: int | None, output_arg: str | None
) -> None:
    print("Playlist request preview (not ranked; no usable prediction cache)")
    print(f"  Moods: {', '.join(moods)}")
    print("  Matching: mean of selected mood probabilities")
    print(f"  Threshold: mean mood probability >= {threshold:g}")
    print("  Tracks requested: all ranked tracks" if count is None else f"  Tracks requested: {count}")
    if output_arg is not None:
        print(f"  Would write: {output_arg}")


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

    try:
        cache_path = resolve_configs(config_from_args(args), config_from_user_file()).pred_cache_path
    except OSError as error:
        _fail("create", f"cannot locate library directory: {error}")

    if cache_path is None and not args.dry_run:
        _fail("create", "pass --cache or --library (or run `moodpl setup`) to use indexed predictions")

    if args.dry_run and (cache_path is None or not cache_path.is_file()):
        _print_request_preview(moods, threshold=args.threshold, count=args.count, output_arg=output_arg)
        return

    if cache_path is None:
        _fail("create", "no prediction cache available")

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
    try:
        tap.Parser(
            tap.SubParserGroup(
                tap.SubParser("moods", tap.TypedArgs, help="List selectable mood labels"),
                tap.SubParser("create", CreateArgs, help="Create a playlist from selected mood tags"),
                tap.SubParser("index", IndexArgs, help="Analyze library tracks and cache mood predictions"),
                tap.SubParser("setup", SetupArgs, help="Configure the library directory (and optional cache dir)"),
                description="Create local playlists from a fixed set of mood tags.",
            ),
            prog="moodpl",
        ).bind(moods_list_cmd, create_cmd, index_cmd, setup_cmd).run()
    except ValueError as error:
        print(f"moodpl: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
