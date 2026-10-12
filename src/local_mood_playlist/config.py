"""User config file.

Optional TOML file: it only supplies defaults. CLI flags keep precedence over
any value here; unknown keys and wrong value types are rejected automatically
via msgspec (see :class:`RawConfig`).

The validated file contents (:class:`RawConfig`) and the typed CLI args
(:mod:`.args`) merge into a :class:`ResolvedConfig` in exactly one place,
:meth:`ResolvedConfig.from_raw_and_cli`. Fields have no defaults, so callers
must pass absent values explicitly as ``None``; nothing is silently dropped.

This module is also the single owner of cache-path resolution.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ResolvedConfig:
    """Config resolved from the file and the typed CLI args; CLI args win.

    Created only by :meth:`from_raw_and_cli` — consumers never re-derive values
    from the file or the environment.
    """

    cache_dir: Path
    library_dir: Path | None
    cache_path: Path | None

    @classmethod
    def from_raw_and_cli(cls, raw: RawConfig, args: IndexArgs | CreateArgs | SetupArgs) -> Self:
        """Merge the validated config file with the typed CLI args; CLI args win."""

        args_cache: str | None = None
        args_cache_dir: str | None = None
        if isinstance(args, (IndexArgs, CreateArgs)):
            args_cache = args.cache
        elif isinstance(args, SetupArgs):
            args_cache_dir = args.cache_dir
        else:
            assert_never(args)

        cache_dir = _to_path(
            _pick_first_non_empty(
                args_cache_dir,
                raw.cache_dir,
            )
        )
        if cache_dir is None:
            cache_dir = default_cache_dir()

        return cls(
            cache_dir=cache_dir,
            library_dir=_to_path(
                _pick_first_non_empty(
                    args.library,
                    raw.library_dir,
                )
            ),
            cache_path=_to_path(
                _pick_first_non_empty(
                    args_cache,
                )
            ),
        )

    def to_raw(self) -> RawConfig:
        return RawConfig(
            cache_dir=str(self.cache_dir),
            library_dir=str(self.library_dir) if self.library_dir is not None else msgspec.UNSET,
        )

    @property
    def pred_cache_path(self) -> Path | None:
        if self.cache_path is not None:
            return self.cache_path
        if self.library_dir is None:
            return None
        resolved_library = self.library_dir.resolve(strict=True)
        library_id = hashlib.sha256(str(resolved_library).encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / "predictions" / f"predictions-{library_id}.pkl"


def _pick_first_non_empty(*candidates: str | Path | msgspec.UnsetType | None) -> str | Path | None:
    """Select the first candidate that is set and non-empty; no path conversion here."""
    for candidate in candidates:
        if candidate is None or isinstance(candidate, msgspec.UnsetType):
            continue
        if isinstance(candidate, str) and len(candidate.strip()) == 0:
            continue
        return candidate
    return None


def _to_path(val: str | Path | None) -> Path | None:
    """Convert a chosen value to an expanded, absolute Path; None stays None (never becomes cwd)."""
    if val is None:
        return None
    return Path(val).expanduser().resolve()


def dump_raw_config(cfg: ResolvedConfig, path: Path | None = None) -> dict[str, object]:
    raw = cfg.to_raw()

    target = path if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_bytes(msgspec.toml.encode(raw))

    payload: dict[str, object] = msgspec.to_builtins(raw)
    return payload
