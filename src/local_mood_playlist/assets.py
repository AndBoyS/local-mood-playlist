"""Download and verify vendored Music2Emotion assets on first use.

Each cache file is downloaded once from the pinned vendored Hugging Face revision
(:data:`VENDORED_REPO` / :data:`VENDORED_REVISION`) and cached under
``<cache root>/moodpl/assets/<revision>/{music2emo,mert}`` (the caller supplies
the cache root, resolved from its :class:`~local_mood_playlist.config.Config`),
mirroring the same ``{music2emo,mert}`` layout.
The payload bytes are pre-transformed per upstream revision by
``scripts/publish_assets.py`` (Lightning checkpoints → safetensors, scalar
metadata extraction, MERT trimmed to 7 layers + zstd, config patch) and
uploaded to that repo; the upstream sources are the pinned Music2Emotion and
MERT HF revisions listed in ``vendor/music2emotion/UPSTREAM.md`` and the
generated HF README.

Cache integrity is verified as one whole-directory digest per bundle
(:data:`BUNDLE_SHA256S`); a mismatch raises and the cache directory under
the pinned revision must be cleared to recover.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import zstandard as zstd

from local_mood_playlist.filesystem_utils import open_atomic

BundleName = Literal["music2emo", "mert"]

VENDORED_REPO = "mementoor/music2emo-inference-assets"
# SHA of the vendored repo upload commit; published via `publish_assets.py --upload`.
VENDORED_REVISION = "50b68ba9b28c03c9acbc0b18c6fefe3c4ccdda59"

BUNDLE_SHA256S: dict[BundleName, str] = {
    "music2emo": "0882b7b6f851d6faf7aa887b347a5ea3916bb7eb5d2d656a48e13867ea6efd62",
    "mert": "da22b50a76a12b3ee1aeb01755fa6f9b0993d24c5edb94a562d355340a631687",
}

BUNDLE_FILES: dict[BundleName, list[str]] = {
    "music2emo": [
        "music2emo/J_all.safetensors",
        "music2emo/btc_model_large_voca.safetensors",
    ],
    "mert": [
        "mert/model.safetensors.zst",
        "mert/config.json",
        "mert/configuration_MERT.py",
        "mert/modeling_MERT.py",
        "mert/preprocessor_config.json",
    ],
}


def asset_cache_dir(cache_root: Path) -> Path:
    """Asset bundle root for the pinned revision, under the given cache root."""
    return cache_root / "assets" / VENDORED_REVISION


def mert_cache_dir(cache_root: Path) -> Path:
    """MERT bundle directory, under :func:`asset_cache_dir`."""
    return asset_cache_dir(cache_root) / "mert"


def _hf_url(repo: str, *, revision: str, repo_path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{repo_path}"


def _download(url: str, dest: Path) -> None:
    with open_atomic(dest) as out, urllib.request.urlopen(url) as response:
        while True:
            chunk: bytes = response.read(1 << 20)
            if chunk == b"":
                break
            _ = out.write(chunk)


def dir_digest(root: Path, rel_paths: Iterable[str]) -> str:
    """Digest files at one root in one stream: length-framed rel path, content, size."""
    digest = hashlib.sha256()
    for rel in sorted(rel_paths):
        digest.update(f"{len(rel)}\x00{rel}\x00".encode())
        total = 0
        with (root / rel).open("rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if chunk == b"":
                    break
                total += len(chunk)
                digest.update(chunk)
        digest.update(f"{total}\x00".encode())
    return digest.hexdigest()


def _is_bundle_intact(root: Path, *, rel_paths: Iterable[str], pin: str) -> bool:
    if not all((root / rel).is_file() for rel in rel_paths):
        return False
    return dir_digest(root, rel_paths) == pin


def _ensure_bundle(name: BundleName, cache_root: Path) -> dict[str, Path]:
    """Verify one bundle; fetch any missing file; validate the whole dir digest."""
    rel_paths = BUNDLE_FILES[name]
    pin = BUNDLE_SHA256S[name]
    root = asset_cache_dir(cache_root)
    paths = {rel: root / rel for rel in rel_paths}
    if _is_bundle_intact(root, rel_paths=rel_paths, pin=pin):
        return paths
    for rel in rel_paths:
        cached = root / rel
        if not cached.is_file():
            _download(_hf_url(VENDORED_REPO, revision=VENDORED_REVISION, repo_path=rel), cached)
    if not _is_bundle_intact(root, rel_paths=rel_paths, pin=pin):
        raise RuntimeError(f"{name} asset bundle failed its directory digest; delete {root} to re-download")
    return paths


def ensure_bundles(cache_root: Path) -> dict[str, Path]:
    """Verify all bundles; fetch any missing file; return one cache path per file."""
    paths: dict[str, Path] = {}
    for name in BUNDLE_FILES:
        paths.update(_ensure_bundle(name, cache_root))
    return paths


@contextmanager
def decompress_mert_weights(snapshot: Path) -> Generator[Path]:
    target_path = snapshot / "model.safetensors"
    source_path = snapshot / "model.safetensors.zst"
    with (
        open_atomic(target_path) as target,
        source_path.open("rb") as source,
        zstd.ZstdDecompressor().stream_reader(source) as reader,
    ):
        shutil.copyfileobj(reader, target, 1 << 20)
    yield snapshot
    target_path.unlink(missing_ok=True)
