"""Download and verify vendored Music2Emotion binary assets on first use.

The large model checkpoints are not stored in git. They are fetched from the
pinned upstream Hugging Face repository revision and cached under the project
`.cache` directory, keyed by revision.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MUSIC2EMO_HF_REVISION = "b036e59471583c3d5b30c69e63e8c7323cc36c4a"
MUSIC2EMO_HF_REPO = "amaai-lab/music2emo"


@dataclass(frozen=True)
class RemoteAsset:
    """A binary asset fetched from the pinned Hugging Face revision."""

    repo_path: str
    sha256: str


MUSIC2EMO_ASSETS: dict[str, RemoteAsset] = {
    "saved_models/J_all.ckpt": RemoteAsset(
        repo_path="saved_models/J_all.ckpt",
        sha256="deaceb291f7974deb688167d3639b7f6eb66eb0715824668618393777d1e07a5",
    ),
    "inference/data/btc_model_large_voca.pt": RemoteAsset(
        repo_path="inference/data/btc_model_large_voca.pt",
        sha256="1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51",
    ),
}


def music2emo_asset_cache_root() -> Path:
    """Project-local cache directory for pinned assets, keyed by revision."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".cache" / "music2emo" / MUSIC2EMO_HF_REVISION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk: bytes = handle.read(1 << 20)
            if chunk == b"":
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as partial:
        partial_path = Path(partial.name)
        with urllib.request.urlopen(url) as response:
            while True:
                chunk: bytes = response.read(1 << 20)
                if chunk == b"":
                    break
                _ = partial.write(chunk)
    _ = partial_path.replace(destination)


def ensure_music2emo_assets() -> dict[str, Path]:
    """Return cache paths for all pinned assets, downloading any missing ones.

    Raises RuntimeError when a downloaded file does not match the pinned
    SHA-256 digest.
    """
    cache_root = music2emo_asset_cache_root()
    paths: dict[str, Path] = {}
    for key, asset in MUSIC2EMO_ASSETS.items():
        cached = cache_root / key
        if cached.is_file():
            if _sha256(cached) != asset.sha256:
                cached.unlink()
            else:
                paths[key] = cached
                continue
        url = (
            f"https://huggingface.co/{MUSIC2EMO_HF_REPO}/resolve/"
            f"{MUSIC2EMO_HF_REVISION}/{asset.repo_path}"
        )
        print(f"Downloading Music2Emotion asset: {asset.repo_path}")
        _download(url, cached)
        if _sha256(cached) != asset.sha256:
            cached.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded asset failed SHA-256 check: {asset.repo_path}")
        paths[key] = cached
    return paths
