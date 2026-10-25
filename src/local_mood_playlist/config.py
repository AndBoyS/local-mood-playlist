"""Layered config resolution.

Configuration is a chain of typed layers, each a :class:`LayerConfig`; the
first layer that sets a field wins:

1. CLI args (:func:`config_from_args`)
2. User TOML file (:func:`config_from_user_file`, at :func:`config_path`)
3. Built-in defaults (:data:`DEFAULT_CONFIG`)

This module is also the single owner of cache-path resolution.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final, Self, assert_never

import msgspec

from local_mood_playlist.args import CreateArgs, IndexArgs, SetupArgs
from local_mood_playlist.moods import BlendMode
from local_mood_playlist.music2emotion import Device
from local_mood_playlist.paths import ResolvedPath


def _xdg_home(var: str, fallback: str) -> Path:
    """Return the absolute base directory for an XDG variable; relative or unset falls back to ``fallback`` under home."""
    home = Path(os.environ.get(var, "").strip())
    return home if home.is_absolute() else Path.home() / fallback


def default_model_cache_dir() -> ResolvedPath:
    """Return the user-level default model cache root: ``$XDG_CACHE_HOME/moodpl`` or ``~/.cache/moodpl``."""
    return ResolvedPath(_xdg_home("XDG_CACHE_HOME", ".cache") / "moodpl")


def default_music_dir() -> ResolvedPath:
    """Return the user-level default predictions home: ``$XDG_DATA_HOME/moodpl`` or ``~/.local/share/moodpl``."""
    return ResolvedPath(_xdg_home("XDG_DATA_HOME", ".local/share") / "moodpl")


class _Unset:
    """Sentinel for "this layer does not set this field"."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


@dataclass(frozen=True)
class LayerConfig:
    """Values set by one source."""

    model_cache: ResolvedPath | _Unset = UNSET
    library_dir: ResolvedPath | None | _Unset = UNSET
    music_cache: ResolvedPath | None | _Unset = UNSET
    blend_mode: BlendMode | _Unset = UNSET
    threshold: float | _Unset = UNSET
    count: int | None | _Unset = UNSET
    output: str | None | _Unset = UNSET
    show_extended: bool | _Unset = UNSET
    device: Device | _Unset = UNSET


@dataclass(frozen=True)
class Config:
    """Fully resolved config layers merged by :func:`resolve_configs`."""

    model_cache: ResolvedPath
    library_dir: ResolvedPath | None
    music_cache: ResolvedPath | None
    blend_mode: BlendMode
    threshold: float
    count: int | None
    output: str | None
    show_extended: bool
    device: Device

    @property
    def pred_cache_path(self) -> ResolvedPath | None:
        """Explicit music cache path; otherwise a predictions path keyed by the configured library path."""
        if self.music_cache is not None:
            return self.music_cache
        if self.library_dir is None:
            return None
        library_id = hashlib.sha256(str(self.library_dir.path).encode("utf-8")).hexdigest()[:20]
        return ResolvedPath(default_music_dir().path / "predictions" / f"predictions-{library_id}.pkl")

    def to_raw(self) -> RawConfig:
        return RawConfig(
            model_cache=str(self.model_cache),
            library_dir=str(self.library_dir) if self.library_dir is not None else msgspec.UNSET,
            mode=self.blend_mode,
            threshold=self.threshold,
            count=self.count if self.count is not None else msgspec.UNSET,
            output=self.output if self.output is not None else msgspec.UNSET,
            show_extended=self.show_extended,
            device=self.device if self.device != "auto" else msgspec.UNSET,
        )


DEFAULT_THRESHOLD: Final = 0.5
DEFAULT_CONFIG = LayerConfig(
    model_cache=default_model_cache_dir(),
    library_dir=None,
    music_cache=None,
    blend_mode="and",
    threshold=DEFAULT_THRESHOLD,
    count=None,
    output=None,
    show_extended=False,
    device="auto",
)


def config_path() -> Path:
    """Return ``$XDG_CONFIG_HOME/moodpl/config.toml`` or ``~/.config/moodpl/config.toml``."""
    return _xdg_home("XDG_CONFIG_HOME", ".config") / "moodpl" / "config.toml"


class RawConfig(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Wire shape of the TOML table.

    Fields stay ``UNSET`` when absent from the table (TOML cannot express
    null); ``omit_defaults`` makes ``msgspec.toml.encode``/``to_builtins`` skip
    unset fields, so the same struct drives reading and writing the file.
    """

    model_cache: str | msgspec.UnsetType = msgspec.UNSET
    library_dir: str | msgspec.UnsetType = msgspec.UNSET
    mode: BlendMode | msgspec.UnsetType = msgspec.UNSET
    threshold: float | msgspec.UnsetType = msgspec.UNSET
    count: int | msgspec.UnsetType = msgspec.UNSET
    output: str | msgspec.UnsetType = msgspec.UNSET
    show_extended: bool | msgspec.UnsetType = msgspec.UNSET
    device: Device | msgspec.UnsetType = msgspec.UNSET

    @classmethod
    def from_user_config(cls) -> Self:
        """Read the user config file; missing file means an empty table."""
        actual_path = config_path()
        if not actual_path.is_file():
            return cls()
        try:
            data = tomllib.loads(actual_path.read_text(encoding="utf-8"))
            return msgspec.convert(data, cls)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"invalid TOML in {actual_path}: {error}") from error
        except msgspec.ValidationError as error:
            raise ValueError(f"{actual_path}: {error}") from error


def _non_empty_or_unset[T](value: T | None | msgspec.UnsetType | _Unset) -> T | _Unset:
    if value is None or isinstance(value, (_Unset, msgspec.UnsetType)):
        return UNSET
    if isinstance(value, str) and len(value.strip()) == 0:
        return UNSET
    return value


def _path_or_unset(value: str | Path | None | msgspec.UnsetType | _Unset) -> ResolvedPath | _Unset:
    """Resolve a path config value; absent or empty becomes UNSET."""
    kept = _non_empty_or_unset(value)
    if isinstance(kept, _Unset):
        return UNSET
    return ResolvedPath(Path(kept))


def config_from_args(args: IndexArgs | CreateArgs | SetupArgs) -> LayerConfig:
    """Return the CLI-args layer; unset flags for the given command stay UNSET."""
    if isinstance(args, CreateArgs):
        return LayerConfig(
            library_dir=_path_or_unset(args.library),
            music_cache=_path_or_unset(args.music_cache),
            blend_mode=_non_empty_or_unset(args.mode),
            threshold=_non_empty_or_unset(args.threshold),
            count=_non_empty_or_unset(args.count),
            output=_non_empty_or_unset(args.output),
            show_extended=UNSET if not args.show_extended else True,
        )
    if isinstance(args, IndexArgs):
        return LayerConfig(
            model_cache=_path_or_unset(args.model_cache),
            library_dir=_path_or_unset(args.library),
            music_cache=_path_or_unset(args.music_cache),
            device=_non_empty_or_unset(args.device),
        )
    if isinstance(args, SetupArgs):
        return LayerConfig(
            model_cache=_path_or_unset(args.model_cache),
            library_dir=_path_or_unset(args.library),
            blend_mode=_non_empty_or_unset(args.mode),
            show_extended=UNSET if not args.show_extended else True,
        )
    assert_never(args)


def config_from_user_file() -> LayerConfig:
    """Return the user config-file layer; absent keys stay UNSET."""
    raw = RawConfig.from_user_config()
    return LayerConfig(
        model_cache=_path_or_unset(raw.model_cache),
        library_dir=_path_or_unset(raw.library_dir),
        blend_mode=_non_empty_or_unset(raw.mode),
        threshold=_non_empty_or_unset(raw.threshold),
        count=_non_empty_or_unset(raw.count),
        output=_non_empty_or_unset(raw.output),
        show_extended=_non_empty_or_unset(raw.show_extended),
        device=_non_empty_or_unset(raw.device),
    )


def _pick_field[T](*values: T | _Unset) -> T:
    """Return the first value that is not ``UNSET``; DEFAULT_CONFIG last must cover every field."""
    for value in values:
        if not isinstance(value, _Unset):
            return value
    msg = "config field set in no layer and missing from DEFAULT_CONFIG"
    raise ValueError(msg)


def resolve_configs(*layers: LayerConfig, default: LayerConfig = DEFAULT_CONFIG) -> Config:
    """Merge config layers: for each field the first non-UNSET value wins."""
    chain = (*layers, default)

    kwargs: dict[str, object] = {}
    for field in fields(LayerConfig):
        kwargs[field.name] = _pick_field(*(getattr(chain_layer, field.name) for chain_layer in chain))
    return Config(**kwargs)  # pyrefly: ignore[bad-argument-type]


def dump_raw_config(cfg: Config, path: Path | None = None) -> dict[str, object]:
    """Write the resolved config to the user config file (or ``path``); return the TOML payload."""
    raw = cfg.to_raw()

    target = path if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_bytes(msgspec.toml.encode(raw))

    payload: dict[str, object] = msgspec.to_builtins(raw)
    return payload
