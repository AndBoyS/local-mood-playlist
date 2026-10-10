"""Project-wide filesystem constants."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CACHE_DIR: Path = REPO_ROOT / ".cache"
