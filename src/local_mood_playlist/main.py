"""CLI shell for fixed-vocabulary mood playlists."""

import sys
from pathlib import Path
from time import monotonic
from typing import NamedTuple, NoReturn

import typed_argparse as tap
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

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
from local_mood_playlist.mood_picker import PICKER_STYLE, MoodSelection, ask_moods_with_mode
from local_mood_playlist.moods import BLEND_MODE_DETAIL, BLEND_MODES, BlendMode, MoodName
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


def _fail(cmd: str, msg: str) -> NoReturn:
    print(f"moodpl {cmd}: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _confirm(question: str) -> bool:
    try:
        answer = inquirer.confirm(question, style=PICKER_STYLE).execute()
    except (KeyboardInterrupt, EOFError):
        answer = None
    if not isinstance(answer, bool):
        print("Cancelled.")
        raise SystemExit(0)
    return answer


def _prompt_path(question: str, *, default: str | None = None, only_directories: bool = True) -> str | None:
    try:
        answer = inquirer.filepath(
            question,
            default="" if default is None else default,
            only_directories=only_directories,
            style=PICKER_STYLE,
        ).execute()
    except (KeyboardInterrupt, EOFError):
        return None
    if not isinstance(answer, str):
        raise TypeError(f"InquirerPy returned an unexpected path result: {answer!r}")
    return answer


def require_library_dir(cfg: Config, *, command: str) -> Path:
    """Return the configured library directory, failing when missing or nonexistent."""
    if cfg.library_dir is None:
        _fail(command, "no library directory: pass one or run `moodpl setup`")
    if not cfg.library_dir.is_dir():
        _fail(command, f"library directory does not exist: {cfg.library_dir}")
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
        force = args.force
        del args  # remaining fields resolve through cfg
        library = require_library_dir(cfg, command="index")
        cache_path = cfg.pred_cache_path
        if cache_path is None:
            _fail("index", "configured library has no prediction cache path")
        summary = index_library(
            library,
            cache_path=cache_path,
            cache_root=cfg.cache_dir,
            force=force,
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


def _prompt_blend_mode(current: BlendMode) -> BlendMode:
    """Prompt for the blend mode, defaulting to ``current``; cancel exits."""
    choices = [Choice(value=mode, name=f"{mode} — {BLEND_MODE_DETAIL[mode].matching}") for mode in BLEND_MODES]
    try:
        answer = inquirer.select(
            "Blend mode",
            choices=choices,
            default=current,
            style=PICKER_STYLE,
        ).execute()
    except (KeyboardInterrupt, EOFError):
        answer = None
    if answer not in BLEND_MODES:
        print("Cancelled.")
        raise SystemExit(0)
    return answer


def _setup_library_dir(cfg: Config, *, non_interactive: bool) -> Path:
    """Return the library directory for setup; interactive setup prompts with the current value prefilled."""
    library = cfg.library_dir
    if not non_interactive:
        return _prompt_library_dir("setup", fallback=library)
    if library is not None and library.is_dir():
        return library
    if library is None:
        _fail("setup", "no library directory: pass --library (terminal not detected)")
    _fail("setup", f"directory does not exist: {library}")


def setup_cmd(args: SetupArgs) -> None:
    non_interactive = not (sys.stdin.isatty() and sys.stdout.isatty())
    arg_config = config_from_args(args)
    cache_dir = args.cache_dir
    mode = args.mode
    del args  # remaining fields resolve through config/prompts
    try:
        user_cfg = config_from_user_file()
        pre_interactive_cfg = resolve_configs(arg_config, user_cfg)
        library = _setup_library_dir(pre_interactive_cfg, non_interactive=non_interactive)

        if not non_interactive:
            answer = _prompt_path("Cache directory", default=str(pre_interactive_cfg.cache_dir))
            if answer is None:
                print("Cancelled.")
                raise SystemExit(0)
            if len(answer.strip()) > 0:
                cache_dir = answer

        audio_count = sum(path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES for path in library.rglob("*"))
        if audio_count == 0:
            print(f"warning: no supported audio files found yet under {library}", file=sys.stderr)

        if not non_interactive:
            mode = _prompt_blend_mode(pre_interactive_cfg.blend_mode)

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
            library=str(library),
            cache_dir=cache_dir,
            mode=mode,
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


def run_mood_picker(*, show_all: bool, default_mode: BlendMode) -> MoodSelection | None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("moodpl create: pass mood labels when running non-interactively", file=sys.stderr)
        raise SystemExit(2)

    shown = MOOD_ENTRIES if show_all else [entry for entry in MOOD_ENTRIES if entry.is_core]
    entries = [(entry.mood, entry.color) for entry in shown]
    return ask_moods_with_mode(entries, message="Choose moods to blend", default_mode=default_mode)


def _print_preview_details(moods: list[MoodName], *, mode: BlendMode, threshold: float) -> None:
    detail = BLEND_MODE_DETAIL[mode]
    print(f"  Moods: {', '.join(moods)}")
    print(f"  Matching: {detail.matching}")
    print(f"  Threshold: {detail.threshold_label} >= {threshold:g}")


def _print_request_preview(
    moods: list[MoodName], *, mode: BlendMode, threshold: float, count: int | None, output_arg: str | None
) -> None:
    print("Playlist request preview (not ranked; no usable prediction cache)")
    _print_preview_details(moods, mode=mode, threshold=threshold)
    print("  Tracks requested: all ranked tracks" if count is None else f"  Tracks requested: {count}")
    if output_arg is not None:
        print(f"  Would write: {output_arg}")


def _validate_playlist_output_path(output_path: Path, cache_path: Path | None) -> None:
    if output_path.suffix.lower() != ".m3u8":
        raise ValueError("playlist output path must end in .m3u8")
    if cache_path is not None and output_path.resolve() == cache_path.resolve():
        raise ValueError("playlist output path must not overwrite prediction cache")


def create_cmd(args: CreateArgs) -> None:
    cfg = resolve_configs(config_from_args(args), config_from_user_file())
    moods = args.moods
    dry_run = args.dry_run
    del args  # remaining fields resolve through cfg
    if cfg.count is not None and cfg.count < 1:
        print("moodpl create: playlist size must be at least 1 (check --count or config `count`)", file=sys.stderr)
        raise SystemExit(2)
    if not 0 < cfg.threshold <= 1:
        print("moodpl create: threshold must be in (0, 1] (check --threshold or config `threshold`)", file=sys.stderr)
        raise SystemExit(2)

    mode = cfg.blend_mode
    if len(moods) == 0:
        picked = run_mood_picker(show_all=cfg.all_moods, default_mode=mode)
        if picked is None:
            print("No moods selected; cancelled.")
            return
        moods = picked.moods
        mode = picked.mode
    if len(set(moods)) != len(moods):
        print("moodpl create: each mood label can be selected only once", file=sys.stderr)
        raise SystemExit(2)

    output_arg = cfg.output
    if output_arg is not None and "{moods}" in output_arg:
        output_arg = output_arg.replace("{moods}", "-".join(moods))

    if cfg.cache_path is None and cfg.library_dir is not None:
        _ = require_library_dir(cfg, command="create")
    cache_path = cfg.pred_cache_path

    if cache_path is None and not dry_run:
        _fail("create", "pass --cache or --library (or run `moodpl setup`) to use indexed predictions")

    if dry_run and (cache_path is None or not cache_path.is_file()):
        _print_request_preview(moods, mode=mode, threshold=cfg.threshold, count=cfg.count, output_arg=output_arg)
        return

    if cache_path is None:
        _fail("create", "no prediction cache available")

    try:
        cache = PredictionCache(cache_path, model_version=MODEL_VERSION)
        ranked = rank_cached_tracks(cache, moods=moods, mode=mode, limit=cfg.count, min_score=cfg.threshold)
        output_path = Path(output_arg).expanduser() if output_arg is not None else None
        if output_path is not None:
            _validate_playlist_output_path(output_path, cache.path)

        if dry_run:
            print("Playlist preview (not written)")
            _print_preview_details(moods, mode=mode, threshold=cfg.threshold)
            requested = "all" if cfg.count is None else str(cfg.count)
            print(f"  Ranked tracks: {len(ranked)} of {requested} requested")
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
                tap.SubParser("setup", SetupArgs, help="Configure the library directory, cache dir, and blend mode"),
                description="Create local playlists from a fixed set of mood tags.",
            ),
            prog="moodpl",
        ).bind(moods_list_cmd, create_cmd, index_cmd, setup_cmd).run()
    except ValueError as error:
        print(f"moodpl: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
