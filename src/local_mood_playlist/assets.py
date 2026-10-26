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

Cache integrity is verified per file (:data:`FILE_SHA256S`); an incorrect
cached file is replaced by a fresh download, and a re-downloaded file that
still mismatches its pin is removed and raises.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import zstandard as zstd

from local_mood_playlist.filesystem_utils import open_atomic
from local_mood_playlist.paths import ResolvedPath

_DOWNLOAD_TIMEOUT_S = 30

VENDORED_REPO = "mementoor/music2emo-inference-assets"
# SHA of the vendored repo upload commit; published via `publish_assets.py --upload`.
VENDORED_REVISION = "50b68ba9b28c03c9acbc0b18c6fefe3c4ccdda59"

FILE_SHA256S: dict[str, str] = {
    "music2emo/J_all.safetensors": "275139403465341709cbeb5dcd5d8c274d2788e4048b8f600d3eb98b7a8f8458",
    "music2emo/btc_model_large_voca.safetensors": "46280982b9a13ab2a30fcc4e5728f24c82fac1accb417665a464710c0dc0685f",
    "mert/model.safetensors.zst": "6f8e05b485d31606b451e5897892db2e5550732b81516e7fda28a2eaf68dd22b",
    "mert/config.json": "73619bc3ad5c4288a1c509c1f73133a1212c42f92a1f1b1f370bef34999fdb31",
    "mert/configuration_MERT.py": "ae0ec2bab8f59c724ba9878a7c20b67210189536ea62d34a56775968e9decb03",
    "mert/modeling_MERT.py": "6c3ee73cef6f0c30ef494f88d96f891fa6925ffe663fa391b512f4b57abecc6c",
    "mert/preprocessor_config.json": "cc5a5e4a5d3b1a758a5ed984b2eaa15bb0522d811d44a9eed82bfca4baa0dc8f",
}

PINNED_FILES = tuple(FILE_SHA256S)


def asset_cache_dir(cache_root: ResolvedPath) -> Path:
    """Asset bundle root for the pinned revision, under the given cache root."""
    return cache_root.path / "assets" / VENDORED_REVISION


def mert_cache_dir(cache_root: ResolvedPath) -> Path:
    """MERT bundle directory, under :func:`asset_cache_dir`."""
    return asset_cache_dir(cache_root) / "mert"


def _hf_url(repo: str, *, revision: str, repo_path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{repo_path}"


@dataclass(frozen=True, slots=True)
class AssetDownloadProgress:
    files_processed: int
    files_total: int
    current_file_name: str
    done_bytes: int
    file_complete: bool


def _download(
    url: str,
    dest: Path,
    *,
    files_processed: int,
    files_total: int,
) -> Iterator[AssetDownloadProgress]:
    with open_atomic(dest) as out, urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        progress = AssetDownloadProgress(
            files_processed=files_processed,
            files_total=files_total,
            current_file_name=dest.name,
            done_bytes=0,
            file_complete=False,
        )
        yield progress
        done_bytes = 0
        while True:
            chunk: bytes = response.read(1 << 20)
            if chunk == b"":
                break
            _ = out.write(chunk)
            done_bytes += len(chunk)
        yield replace(progress, done_bytes=done_bytes, file_complete=True)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if chunk == b"":
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_asset_paths(cache_root: ResolvedPath) -> dict[str, Path]:
    root = asset_cache_dir(cache_root)
    return {rel: root / rel for rel in PINNED_FILES}


def asset_download(cache_root: ResolvedPath) -> Iterator[AssetDownloadProgress]:
    root = asset_cache_dir(cache_root)
    to_fetch: list[str] = []
    for rel in PINNED_FILES:
        dest = root / rel
        if not dest.is_file() or file_digest(dest) != FILE_SHA256S[rel]:
            to_fetch.append(rel)

    for files_processed, rel in enumerate(to_fetch):
        dest = root / rel
        yield from _download(
            _hf_url(VENDORED_REPO, revision=VENDORED_REVISION, repo_path=rel),
            dest,
            files_processed=files_processed,
            files_total=len(to_fetch),
        )
        if file_digest(dest) != FILE_SHA256S[rel]:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"{rel} still fails its pinned digest after download; server bytes may differ from the pin"
            )


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
