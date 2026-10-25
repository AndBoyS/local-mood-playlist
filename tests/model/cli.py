import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from const import REPO_DIR


@dataclass(frozen=True)
class CliResult:
    exit_code: int
    stdout: str
    stderr: str


class CliContext:
    def __init__(self, temp_dir: Path, moodpl_binary: Path | None = None) -> None:
        self.temp_dir = temp_dir
        self.config_dir = temp_dir / "config"
        self.data_dir = temp_dir / "data"
        self.model_cache = temp_dir / "cache"
        self.library_dir = temp_dir / "library"
        self.playlists_dir = temp_dir / "playlists"
        self.moodpl_binary = moodpl_binary
        for directory in (self.config_dir, self.data_dir, self.model_cache, self.library_dir, self.playlists_dir):
            directory.mkdir(parents=True)


    @property
    def _env(self) -> dict[str, str]:
        passthrough = {
            key: value
            for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")
            if (value := os.environ.get(key)) is not None
        }
        return {
            **passthrough,
            "XDG_CONFIG_HOME": str(self.config_dir),  # empty config, not the user's
            "XDG_CACHE_HOME": str(self.model_cache),  # default_model_cache_dir routes here
            "XDG_DATA_HOME": str(self.data_dir),  # default_music_dir routes here
            "NO_COLOR": "1",
        }

    def run(self, *args: str) -> CliResult:
        command = (
            [str(self.moodpl_binary), *args]
            if self.moodpl_binary is not None
            else ["uv", "run", "--project", str(REPO_DIR), "moodpl", *args]
        )
        completed = subprocess.run(
            command,
            cwd=self.temp_dir,
            env=self._env,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=600,
        )
        return CliResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@dataclass(frozen=True)
class IndexedLibrary:
    context: CliContext
    music_cache: Path
    tracks: list[Path]
