"""CLI shell for fixed-vocabulary mood playlists."""

import re
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import NamedTuple, NoReturn

import typed_argparse as tap
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from local_mood_playlist.args import CreateArgs, IndexArgs, SetupArgs
from local_mood_playlist.assets import AssetDownloadProgress, asset_download
from local_mood_playlist.config import (
    DEFAULT_THRESHOLD,
    Config,
    config_from_args,
    config_from_user_file,
    config_path,
    dump_raw_config,
    resolve_configs,
)
from local_mood_playlist.indexing import (
    AUDIO_SUFFIXES,
    IndexSummary,
    index_library,
)
from local_mood_playlist.mood_picker import (
    MOVE_KEYBINDINGS,
    PICKER_STYLE,
    MoodSelection,
    ask_moods_with_mode,
)
from local_mood_playlist.moods import BLEND_MODE_DETAIL, BLEND_MODES, BlendMode, MoodName
from local_mood_playlist.music2emotion import MODEL_VERSION
from local_mood_playlist.paths import ResolvedPath
from local_mood_playlist.playlists import rank_cached_tracks, render_m3u8
from local_mood_playlist.predictions import PredictionCache


class MoodEntry(NamedTuple):
    mood: MoodName
    color: str
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


def _cancel() -> NoReturn:
    print("Cancelled.")
    raise SystemExit(0)


def _run_prompt[T](run: Callable[[], T]) -> T | None:
    """Execute an InquirerPy prompt, mapping interrupt or EOF to ``None``."""
    try:
        return run()
    except (KeyboardInterrupt, EOFError):
        return None


def _confirm(question: str) -> bool:
    answer = _run_prompt(lambda: inquirer.confirm(question, style=PICKER_STYLE).execute())
    if not isinstance(answer, bool):
        _cancel()
    return answer


def require_library_dir(cfg: Config, *, command: str) -> ResolvedPath:
    """Return library dir / informational errors for different kinds of library problems."""
    if cfg.library_dir is None:
        _fail(command, "no library directory: pass one or run `moodpl setup`")
    if not cfg.library_dir.path.exists():
        _fail(command, f"library directory does not exist: {cfg.library_dir}")
    if not cfg.library_dir.path.is_dir():
        _fail(command, f"not a directory: {cfg.library_dir}")
    return cfg.library_dir


def moods_list_cmd(_: tap.TypedArgs) -> None:
    for entry in MOOD_ENTRIES:
        if entry.is_core:
            print(f"  {entry.mood}")
    print()
    print("  Extended model tags (offerable in the dialog via `moodpl create -e` or `moodpl setup`):")
    for entry in MOOD_ENTRIES:
        if not entry.is_core:
            print(f"  {entry.mood}")


def run_mood_picker(*, show_extended: bool, default_mode: BlendMode) -> MoodSelection | None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("moodpl create: pass mood labels when running non-interactively", file=sys.stderr)
        raise SystemExit(2)

    shown = MOOD_ENTRIES if show_extended else [entry for entry in MOOD_ENTRIES if entry.is_core]
    entries = [(entry.mood, entry.color) for entry in shown]
    return ask_moods_with_mode(entries, message="Choose moods to blend", default_mode=default_mode)


def _moods_str(moods: list[MoodName], *, mode: BlendMode, threshold: float) -> str:
    """What {moods} in filename expands to."""
    parts: list[str] = ["-".join(moods)]
    if mode != "and":
        parts.append(mode)
    if threshold != DEFAULT_THRESHOLD:
        parts.append(f"t{threshold:.2f}")
    return "-".join(parts)


def _resolve_playlist_output(
    output: str | None,
    moods: list[MoodName],
    *,
    mode: BlendMode = "and",
    threshold: float = DEFAULT_THRESHOLD,
) -> ResolvedPath | None:
    if output is None:
        return None
    if sys.platform == "win32" and re.fullmatch(r"[A-Za-z]:", output) is not None:
        print(
            f"moodpl create: warning: output {output!r} is a bare drive letter which is not the root"
            " but the relative path to current folder",
            file=sys.stderr,
        )
    moods_str = _moods_str(moods, mode=mode, threshold=threshold)
    if "{moods}" in output:
        candidate = Path(output.replace("{moods}", moods_str))
        if candidate.suffix.lower() != ".m3u8":
            candidate = Path(f"{candidate}.m3u8")
    elif output.lower().endswith(".m3u8"):
        candidate = Path(output)
    else:
        candidate = Path(output) / f"{moods_str}.m3u8"
    return ResolvedPath(candidate)


def _validate_playlist_output_path(output_path: ResolvedPath, cache_path: ResolvedPath) -> None:
    if output_path.path.suffix.lower() != ".m3u8":
        raise ValueError("playlist output path must end in .m3u8")
    if output_path.path.parent.is_file():
        raise ValueError(f"playlist output parent directory is an existing file: {output_path.path.parent}")
    if output_path == cache_path:
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
        picked = run_mood_picker(show_extended=cfg.show_extended, default_mode=mode)
        if picked is None or len(picked.moods) == 0:
            print("No moods selected; cancelled.")
            return
        moods = picked.moods
        mode = picked.mode
    if len(set(moods)) != len(moods):
        print("moodpl create: each mood label can be selected only once", file=sys.stderr)
        raise SystemExit(2)

    output_path = _resolve_playlist_output(cfg.output, moods, mode=mode, threshold=cfg.threshold)

    if cfg.music_cache is None and cfg.library_dir is not None:
        _ = require_library_dir(cfg, command="create")
    cache_path = cfg.pred_cache_path

    if cache_path is None and not dry_run:
        _fail("create", "pass --music-cache or --library (or run `moodpl setup`) to use indexed predictions")

    if dry_run and (cache_path is None or not cache_path.path.is_file()):
        _print_request_preview(moods, mode=mode, threshold=cfg.threshold, count=cfg.count, output_path=output_path)
        return

    assert cache_path is not None, "Already processed this case"
    try:
        cache = PredictionCache(cache_path, model_version=MODEL_VERSION)
        ranked = rank_cached_tracks(cache, moods=moods, mode=mode, limit=cfg.count, min_score=cfg.threshold)
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
            output_path.path.parent.mkdir(parents=True, exist_ok=True)
            _ = output_path.path.write_text(playlist, encoding="utf-8")
            print(f"Created {len(ranked)}-track playlist: {output_path}", file=sys.stderr)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"moodpl create: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def _print_preview_details(moods: list[MoodName], *, mode: BlendMode, threshold: float) -> None:
    detail = BLEND_MODE_DETAIL[mode]
    print(f"  Moods: {', '.join(moods)}")
    print(f"  Matching: {detail.matching}")
    print(f"  Threshold: {detail.threshold_label} >= {threshold:g}")


_WORK_BAR_WIDTH = 30


def _bar(done: int, total: int, *, width: int = _WORK_BAR_WIDTH) -> str:
    filled = done * width // total
    return "#" * filled + "-" * (width - filled)


MB = 1 << 20


def _print_download_pbar(progress: AssetDownloadProgress) -> str:
    which = progress.files_processed + 1
    if progress.file_complete:
        return f"Downloaded {which}/{progress.files_total} {progress.current_file_name} ({progress.done_bytes / MB:.1f} MB)"
    return f"Downloading {which}/{progress.files_total} {progress.current_file_name}..."


def _print_request_preview(
    moods: list[MoodName], *, mode: BlendMode, threshold: float, count: int | None, output_path: ResolvedPath | None
) -> None:
    print("Playlist request preview (not ranked; no usable prediction cache)")
    _print_preview_details(moods, mode=mode, threshold=threshold)
    print("  Tracks requested: all ranked tracks" if count is None else f"  Tracks requested: {count}")
    if output_path is not None:
        print(f"  Would write: {output_path}")


MINUTES = 60


def index_cmd(args: IndexArgs) -> None:
    progress_active = False
    progress_line_len = 0

    try:
        cfg = resolve_configs(config_from_args(args), config_from_user_file())
        force = args.force
        del args  # remaining fields resolve through cfg
        library = require_library_dir(cfg, command="index")
        cache_path = cfg.pred_cache_path
        if cache_path is None:
            _fail("index", "configured library has no prediction cache path")
        print(f"Music cache: {cache_path}")

        # Validates the library, scans it, and partitions against the cache
        # eagerly — a bad or empty library fails here, before asset work.
        indexing = index_library(
            library,
            cache_path=cache_path,
            cache_root=cfg.model_cache,
            device=cfg.device,
            force=force,
        )

        print("Verifying vendored assets...", file=sys.stderr, flush=True)

        for progress in asset_download(cfg.model_cache):
            print(_print_download_pbar(progress), file=sys.stderr, flush=True)

        progress_started = monotonic()
        summary: IndexSummary | None = None

        for frame in indexing:
            if isinstance(frame, IndexSummary):
                summary = frame
                break

            percent = frame.processed * 100 // frame.to_process
            if frame.processed == 0:
                eta = "ETA calculating"
            else:
                remaining = int(
                    (monotonic() - progress_started) / frame.processed * (frame.to_process - frame.processed)
                )
                eta = f"ETA {remaining // MINUTES:02d}m{remaining % MINUTES:02d}s"

            track_name = frame.path_being_predicted.path.name[:48]
            progress_line = (
                f"Indexing [{_bar(frame.processed, frame.to_process)}]"
                f" {percent:3d}% predicting {frame.processed}/{frame.to_process}"
                f" {eta} {track_name}"
            )
            if sys.stderr.isatty():
                # Pad to the previous redraw's width so a shorter next line
                # erases its stale characters instead of leaving them behind.
                print(f"\r{progress_line.ljust(progress_line_len)}", end="", file=sys.stderr, flush=True)
                progress_line_len = len(progress_line)
                progress_active = True
            else:
                print(progress_line, file=sys.stderr, flush=True)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        # Close any dangling bar line.
        if progress_active:
            progress_active = False
            print(file=sys.stderr, flush=True)
        print(f"moodpl index: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    finally:
        if progress_active:
            print(file=sys.stderr, flush=True)

    assert summary is not None, "index_library always ends with a summary unless it raised above."
    print(f"Indexed {summary.tracks_found} audio files")
    print(f"  Reused: {summary.predictions_reused}")
    print(f"  Predicted: {summary.predictions_computed}")
    print(f"  Failed: {len(summary.failures)}")
    for failure in summary.failures[:_MAX_FAILURES_TO_PRINT]:
        print(f"  {failure.audio_path}: {failure.error}", file=sys.stderr)
    if len(summary.failures) > _MAX_FAILURES_TO_PRINT:
        remaining_failures = len(summary.failures) - _MAX_FAILURES_TO_PRINT
        print(f"  ... and {remaining_failures} more failures", file=sys.stderr)
    print(f"  Music cache: {cache_path}")


def setup_cmd(args: SetupArgs) -> None:
    non_interactive = not (sys.stdin.isatty() and sys.stdout.isatty())
    arg_config = config_from_args(args)
    model_cache = args.model_cache
    mode = args.mode
    show_extended = args.show_extended
    del args  # remaining fields resolve through config/prompts
    try:
        user_cfg = config_from_user_file()
        pre_interactive_cfg = resolve_configs(arg_config, user_cfg)
        library = _setup_library_dir(pre_interactive_cfg, non_interactive=non_interactive)

        if not non_interactive:
            answer = _prompt_path("Model cache directory", default=str(pre_interactive_cfg.model_cache))
            if answer is None:
                _cancel()
            if len(answer.strip()) > 0:
                model_cache = answer

        audio_count = sum(path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES for path in library.path.rglob("*"))
        if audio_count == 0:
            print(f"warning: no supported audio files found yet under {library}", file=sys.stderr)

        if not non_interactive:
            mode = _prompt_blend_mode(pre_interactive_cfg.blend_mode)
            show_extended = _prompt_show_extended(show_extended or pre_interactive_cfg.show_extended)

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
            model_cache=model_cache,
            mode=mode,
            show_extended=show_extended,
        )
        cfg = resolve_configs(config_from_args(final_args), user_cfg)
        if not non_interactive:
            cfg = replace(cfg, show_extended=show_extended)
        payload = dump_raw_config(cfg, cfg_path)
    except (OSError, TypeError, ValueError) as error:
        print(f"moodpl setup: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    print(f"Config written: {cfg_path}")
    for name, value in payload.items():
        print(f"  {name} = {value}")
    print("Next: moodpl index")


def _prompt_path(question: str, *, default: str | None = None) -> str | None:
    answer = _run_prompt(
        lambda: inquirer.filepath(
            question,
            default="" if default is None else default,
            only_directories=True,
            style=PICKER_STYLE,
        ).execute()
    )
    if answer is None:
        return None
    if not isinstance(answer, str):
        raise TypeError(f"InquirerPy returned an unexpected path result: {answer!r}")
    return answer


def _prompt_library_dir(cmd: str, *, fallback: ResolvedPath | None = None) -> ResolvedPath:
    """Prompt for a library directory until it exists; empty answer re-prompts, cancel exits."""
    while True:
        answer = _prompt_path("Music library directory", default=str(fallback) if fallback is not None else None)
        if answer is None:
            _cancel()
        if len(answer.strip()) == 0:
            print(f"moodpl {cmd}: empty directory", file=sys.stderr)
            continue
        expanded = Path(answer).expanduser()
        candidate = ResolvedPath(expanded)
        if candidate.path.is_dir():
            return candidate
        print(f"moodpl {cmd}: directory does not exist: {expanded}", file=sys.stderr)


def _prompt_blend_mode(current: BlendMode) -> BlendMode:
    """Prompt for the blend mode, defaulting to ``current``; cancel exits."""
    choices = [Choice(value=mode, name=f"{mode} — {BLEND_MODE_DETAIL[mode].matching}") for mode in BLEND_MODES]
    answer = _run_prompt(
        lambda: inquirer.select(
            "Blend mode",
            choices=choices,
            default=current,
            style=PICKER_STYLE,
            keybindings=MOVE_KEYBINDINGS,
        ).execute()
    )
    if answer not in BLEND_MODES:
        _cancel()
    return answer


def _setup_library_dir(cfg: Config, *, non_interactive: bool) -> ResolvedPath:
    """Return the library directory for setup; interactive setup prompts with the current value prefilled."""
    library = cfg.library_dir
    if not non_interactive:
        return _prompt_library_dir("setup", fallback=library)
    if library is not None and library.path.is_dir():
        return library
    if library is None:
        _fail("setup", "no library directory: pass --library (terminal not detected)")
    _fail("setup", f"directory does not exist: {library}")


def _prompt_show_extended(current: bool) -> bool:
    """Prompt whether the picker offers extended mood tags, defaulting to ``current``; cancel exits."""
    examples = [entry.mood for entry in MOOD_ENTRIES if not entry.is_core]
    examples = ", ".join(examples[:4])
    answer = _run_prompt(
        lambda: inquirer.confirm(
            f"Offer extended mood tags (e.g. {examples}, ...)",
            default=current,
            style=PICKER_STYLE,
        ).execute()
    )
    if not isinstance(answer, bool):
        _cancel()
    return answer


def main() -> None:
    try:
        tap.Parser(
            tap.SubParserGroup(
                tap.SubParser("moods", tap.TypedArgs, help="List selectable mood labels"),
                tap.SubParser("create", CreateArgs, help="Create a playlist from selected mood tags"),
                tap.SubParser("index", IndexArgs, help="Analyze library tracks and cache mood predictions"),
                tap.SubParser("setup", SetupArgs, help="Configure the library directory, model cache, and blend mode"),
                description="Create local playlists from a fixed set of mood tags.",
            ),
            prog="moodpl",
        ).bind(moods_list_cmd, create_cmd, index_cmd, setup_cmd).run()
    except ValueError as error:
        print(f"moodpl: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
