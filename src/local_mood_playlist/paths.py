"""Path value types resolved and validated once at input boundaries.

CLI args, config values, and prompts construct a :class:`ResolvedPath`, which
canonicalizes the value once (``~`` expanded, absolute, symlinks resolved). A
:class:`ResolvedPath` in a signature marks a path whose canonical form is
part of the contract (config values, library roots, the prediction cache's
audio keys); :class:`RelativePath` holds the library-root-relative form
embedded in playlists. Existence is not part of either type: functions that
need it check with ``.path`` (e.g. ``is_dir``) at their own boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class PathError(ValueError):
    """A user-supplied path does not satisfy its semantic contract."""


@dataclass(frozen=True, slots=True, order=True)
class ResolvedPath:
    """Expanded, absolute, symlink-resolved path; no existence requirement.

    ``__post_init__`` canonicalizes on construction (``~`` expanded,
    absolute, symlinks resolved), so no construction path produces a
    non-canonical instance that would hash distinctly from its cache-key
    form. A relative input resolves against the current working directory.
    """

    path: Path

    def __str__(self) -> str:
        return str(self.path)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.expanduser().resolve())


@dataclass(frozen=True, slots=True)
class RelativePath:
    """Relative path, as embedded in playlist entries to stay valid across devices.

    This is a different namespace from :class:`ResolvedPath`: it is never
    absolute, unchecked on the filesystem, and rendered in POSIX form.
    Construction normalizes lexically (``a/./b`` → ``a/b``).
    """

    path: Path

    def __str__(self) -> str:
        return self.as_posix()

    def as_posix(self) -> str:
        return self.path.as_posix()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(os.path.normpath(self.path)))
        if self.path.is_absolute():
            raise PathError(f"path must be relative, got {self.path}")
