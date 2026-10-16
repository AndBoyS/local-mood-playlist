"""Shared constants for repo-local scripts, derived from this file's location."""

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]

assert (REPO_DIR / "src" / "local_mood_playlist").is_dir(), (
    f"expected the package at {REPO_DIR / 'src' / 'local_mood_playlist'}; const.py may have been moved"
)
