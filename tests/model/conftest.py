import hashlib
import os
import subprocess
from pathlib import Path

import manifest
import pytest
from cli import CliContext, IndexedLibrary
from const import REPO_DIR

from local_mood_playlist.indexing import AUDIO_SUFFIXES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def pinned_library_tracks() -> list[Path]:
    library = manifest.LIBRARY_DIR
    if not library.is_dir():
        pytest.skip(f"test library not present on this machine: {library}")
    by_sha = {
        _sha256(path): path
        for path in library.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    }
    missing = [track.sha256 for track in manifest.TRACKS if track.sha256 not in by_sha]
    if len(missing) == 0:
        return [by_sha[track.sha256] for track in manifest.TRACKS]
    pytest.fail(f"pinned tracks missing from {library}: {missing}; update manifest.py if the tracks moved")


@pytest.fixture(scope="session")
def indexed_library(pinned_library_tracks: list[Path], tmp_path_factory: pytest.TempPathFactory) -> IndexedLibrary:
    artifact_venv = tmp_path_factory.mktemp("artifact") / "venv"
    build = subprocess.run(
        ["uv", "sync", "--no-editable", "--no-dev"],
        cwd=REPO_DIR,
        env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(artifact_venv)},
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        pytest.fail(f"built-wheel venv failed: {build.stderr.strip()} {build.stdout.strip()}")

    context = CliContext(
        tmp_path_factory.mktemp("cli-model"),
        moodpl_binary=artifact_venv / "bin" / "moodpl",
    )
    for track in pinned_library_tracks:
        (context.library_dir / track.name).symlink_to(track)

    cache = context.cache_dir / "predictions.pkl"
    result = context.run("index", str(context.library_dir), "--cache", str(cache))
    assert result.exit_code == 0, result.stderr
    assert cache.is_file(), "index wrote no prediction cache"
    return IndexedLibrary(context=context, cache=cache, tracks=sorted(context.library_dir.glob("*")))
