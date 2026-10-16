"""Typed CLI argument definitions.

Shared by the parser binding (``main``) and config resolution (``config``), so
both see exactly the same typed args.
"""

import typed_argparse as tap

from local_mood_playlist.moods import BlendMode, MoodName
from local_mood_playlist.music2emotion import Device


class IndexArgs(tap.TypedArgs):
    library: str | None = tap.arg(positional=True, help="Music library directory (default: configured library_dir)")
    cache: str | None = tap.arg(default=None, help="Prediction cache file path")
    force: bool = tap.arg(
        default=False,
        help="Recompute predictions for every supported audio file",
        auto_default_help=False,
    )
    device: Device | None = tap.arg(
        default=None,
        help="Inference device (default: first available of CUDA, MPS, then CPU); pass cpu, mps, or cuda to override",
        auto_default_help=False,
    )


class CreateArgs(tap.TypedArgs):
    moods: list[MoodName] = tap.arg(positional=True, nargs="*", help="Mood labels to blend")
    show_extended: bool = tap.arg(
        "-e",
        "--extended",
        default=False,
        help="Offer extended model-only mood tags in the dialog, e.g. sport, film, game, trailer "
        "(default: config `show_extended`)",
        auto_default_help=False,
    )
    mode: BlendMode | None = tap.arg(
        "-m",
        "--mode",
        default=None,
        help="Blend selected mood probabilities: and (every mood), or (any mood), mean (average); "
        "default: config `mode`",
    )
    count: int | None = tap.arg("-n", default=None, help="Playlist size (default: config `count` or all ranked tracks)")
    threshold: float | None = tap.arg(
        "-t",
        "--threshold",
        default=None,
        help="Minimum blend score in (0, 1]; default: config `threshold` or 0.5",
    )
    output: str | None = tap.arg(
        "-o",
        default=None,
        help=(
            "Playlist destination; {moods} names the playlist file and expands to the selected labels"
            " joined by -, plus the blend mode and threshold (t<score>, e.g. t0.30, always 2 decimals)"
            " whenever they differ from the defaults (and / 0.5) (.m3u8 appended when missing), a value"
            " ending in .m3u8 is used as-is, otherwise it is a directory receiving <moods>.m3u8 following"
            " the same expansion rule (default: config `output` or stdout)"
        ),
    )
    library: str | None = tap.arg(default=None, help="Library root for the default prediction cache")
    cache: str | None = tap.arg(default=None, help="Prediction cache file path (overrides --library)")
    dry_run: bool = tap.arg(
        default=False,
        help="Preview ranked tracks when cache exists; never write a playlist",
        auto_default_help=False,
    )


class SetupArgs(tap.TypedArgs):
    library: str | None = tap.arg(default=None, help="Music library directory to configure")
    cache_dir: str | None = tap.arg(default=None, help="Cache root directory to configure")
    mode: BlendMode | None = tap.arg(
        "-m",
        "--mode",
        default=None,
        help="Blend mode to configure (and, or, mean); prompted interactively when omitted",
    )
    show_extended: bool = tap.arg(
        "-e",
        "--extended",
        default=False,
        help="Offer extended mood tags in the dialog, e.g. sport, film, game, trailer",
    )
