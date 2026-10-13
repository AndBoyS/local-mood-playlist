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
from typing import Self, assert_never

import msgspec

from local_mood_playlist.args import CreateArgs, IndexArgs, SetupArgs


def _xdg_home(var: str, fallback: str) -> Path:
    """Return the absolute base directory for an XDG variable; relative or unset falls back to ``fallback`` under home."""
    home = Path(os.environ.get(var, "").strip())
    return home if home.is_absolute() else Path.home() / fallback


def default_cache_dir() -> Path:
    """Return the user-level default cache root: ``$XDG_CACHE_HOME/moodpl`` or ``~/.cache/moodpl``."""
    return _xdg_home("XDG_CACHE_HOME", ".cache") / "moodpl"


class _Unset:
    """Sentinel for "this layer does not set this field"."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


@dataclass(frozen=True)
class LayerConfig:
    """Values set by one source."""

    cache_dir: Path | _Unset = UNSET
    library_dir: Path | None | _Unset = UNSET
    cache_path: Path | None | _Unset = UNSET


@dataclass(frozen=True)
class Config:
    """Fully resolved config layers merged by :func:`resolve_configs`."""

    cache_dir: Path
    library_dir: Path | None
    cache_path: Path | None

    @property
    def pred_cache_path(self) -> Path | None:
        if self.cache_path is not None:
            return self.cache_path
        if self.library_dir is None:
            return None
        resolved_library = self.library_dir.resolve(strict=True)
        library_id = hashlib.sha256(str(resolved_library).encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / "predictions" / f"predictions-{library_id}.pkl"

    def to_raw(self) -> RawConfig:
        return RawConfig(
            cache_dir=str(self.cache_dir),
            library_dir=str(self.library_dir) if self.library_dir is not None else msgspec.UNSET,
        )


DEFAULT_CONFIG = LayerConfig(
    cache_dir=default_cache_dir(),
    library_dir=None,
    cache_path=None,
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

    cache_dir: str | msgspec.UnsetType = msgspec.UNSET
    library_dir: str | msgspec.UnsetType = msgspec.UNSET

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


def _path_or_unset(value: str | Path | msgspec.UnsetType | _Unset | None) -> Path | _Unset:
    """Validate one candidate value at its source boundary; blank or absent becomes UNSET.

    Expanded and made absolute; a path is never silently turned into cwd by an
    empty string.
    """
    if value is None or isinstance(value, (_Unset, msgspec.UnsetType)):
        return UNSET
    value = str(value)
    if len(value.strip()) == 0:
        return UNSET
    return Path(value).expanduser().resolve()


def config_from_args(args: IndexArgs | CreateArgs | SetupArgs) -> LayerConfig:
    """Return the CLI-args layer; unset flags for the given command stay UNSET."""
    if isinstance(args, (IndexArgs, CreateArgs)):
        return LayerConfig(
            library_dir=_path_or_unset(args.library),
            cache_path=_path_or_unset(args.cache),
        )
    if isinstance(args, SetupArgs):
        return LayerConfig(
            cache_dir=_path_or_unset(args.cache_dir),
            library_dir=_path_or_unset(args.library),
        )
    assert_never(args)


def config_from_user_file() -> LayerConfig:
    """Return the user config-file layer; absent keys stay UNSET."""
    raw = RawConfig.from_user_config()
    return LayerConfig(
        cache_dir=_path_or_unset(raw.cache_dir),
        library_dir=_path_or_unset(raw.library_dir),
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
