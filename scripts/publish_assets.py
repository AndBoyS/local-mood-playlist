"""Build and optionally publish the vendored Music2Emo asset bundle.

Reads the pinned upstream files, applies the one-time transforms (Lightning
checkpoint → safetensors, scalar metadata extraction, MERT 7-layer trim +
zstd compression, config patch), and stages a directory ready for upload to
the Hugging Face repo. ``--upload`` pushes the staging directory
(``HF_TOKEN`` token required).

uv run scripts/publish_assets.py --upload

The printed upload commit SHA becomes ``VENDORED_REVISION`` in
``src/local_mood_playlist/assets.py``.
"""

import argparse
import io
import json
import shutil
import urllib.request
from pathlib import Path

import torch
import zstandard as zstd
from huggingface_hub import HfApi
from safetensors.torch import save as save_safetensors

from local_mood_playlist.assets import BUNDLE_FILES, VENDORED_REPO, dir_digest
from local_mood_playlist.const import CACHE_DIR, REPO_ROOT

# Upstream provenance (see README generated below for the full chain).
MUSIC2EMO_UPSTREAM_REPO = "amaai-lab/music2emo"
MUSIC2EMO_UPSTREAM_REVISION = "b036e59471583c3d5b30c69e63e8c7323cc36c4a"
MERT_UPSTREAM_REPO = "m-a-p/MERT-v1-95M"
MERT_UPSTREAM_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"

MERT_LAYERS_TO_KEEP = 7
MERT_ZSTD_LEVEL = 12


def _url(repo: str, *, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def _fetch_music2emo_file(path: str) -> bytes:
    return _fetch(_url(MUSIC2EMO_UPSTREAM_REPO, revision=MUSIC2EMO_UPSTREAM_REVISION, path=path))


def _fetch_mert_file(path: str) -> bytes:
    return _fetch(_url(MERT_UPSTREAM_REPO, revision=MERT_UPSTREAM_REVISION, path=path))


def _j_all_into_safetensors(upstream: bytes) -> bytes:
    checkpoint = torch.load(io.BytesIO(upstream), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("unexpected J_all.ckpt layout: not a dict")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("unexpected J_all.ckpt layout: state_dict is not a dict")
    tensors: dict[str, torch.Tensor] = {
        key: value for key, value in state_dict.items() if isinstance(value, torch.Tensor)
    }
    return save_safetensors(tensors, metadata={"format": "pt"})


def _btc_into_safetensors(upstream: bytes) -> bytes:
    checkpoint = torch.load(io.BytesIO(upstream), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("unexpected chord checkpoint layout: not a dict")
    model = checkpoint["model"]
    if not isinstance(model, dict):
        raise TypeError("unexpected chord checkpoint layout: model is not a dict")
    tensors: dict[str, torch.Tensor] = {key: value for key, value in model.items() if isinstance(value, torch.Tensor)}
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    if not isinstance(mean, int | float) or not isinstance(std, int | float):
        raise TypeError("unexpected chord checkpoint layout: mean/std not numeric")
    metadata = {"mean": repr(float(mean)), "std": repr(float(std)), "format": "pt"}
    return save_safetensors(tensors, metadata=metadata)


def _mert_into_zstd_safetensors(upstream: bytes) -> bytes:
    """Trim MERT weights to the layers inference consumes, compress."""
    state_dict = torch.load(io.BytesIO(upstream), map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError("unexpected MERT checkpoint layout: not a dict")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("unexpected MERT checkpoint layout: non-string key")
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("encoder.layers.") and int(key.split(".")[2]) >= MERT_LAYERS_TO_KEEP:
            continue
        tensors[key] = value
    payload = save_safetensors(tensors, metadata={"format": "pt"})
    return zstd.ZstdCompressor(level=MERT_ZSTD_LEVEL).compress(payload)


def _patch_mert_config(config: bytes) -> bytes:
    parsed = json.loads(config.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("unexpected MERT config.json layout: not an object")
    parsed["num_hidden_layers"] = MERT_LAYERS_TO_KEEP
    return json.dumps(parsed, indent=2).encode() + b"\n"


APACHE2_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
CC_BY_NC4_URL = "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (asset-publisher)"})
    with urllib.request.urlopen(request) as response:
        payload: bytes = response.read()
    return payload


def _fetch_text(url: str) -> str:
    return _fetch(url).decode("utf-8")


def _readme() -> str:
    return f"""\
---
license: cc-by-nc-4.0
tags:
- music
- music-information-retrieval
- audio
library_name: custom
---
# Music2Emo Inference Assets (vendored)

Pinned, pre-transformed asset bundle for the local-mood-playlist Music2Emotion
inference pipeline. Not an original model.

## Transformations (one-time, verified upstream pre-transform)

- `J_all.ckpt`: Lightning checkpoint → `state_dict` safetensors;
  optimizer state and Lightning training metadata dropped.
- `btc_model_large_voca.pt`: scalar mean/std moved into safetensors metadata.
- `pytorch_model.bin`: MERT weights trimmed to the first {MERT_LAYERS_TO_KEEP}
  transformer layers (inference consumes only hidden states 6 and 7) and
  stored zstd level {MERT_ZSTD_LEVEL} compressed; `num_hidden_layers` patched
  into `mert/config.json`.

## Provenance

| Asset | Upstream | Upstream revision |
|---|---|---|
| `music2emo/J_all.safetensors` | [{MUSIC2EMO_UPSTREAM_REPO}](https://huggingface.co/{MUSIC2EMO_UPSTREAM_REPO}) `saved_models/J_all.ckpt` | `{MUSIC2EMO_UPSTREAM_REVISION}` |
| `music2emo/btc_model_large_voca.safetensors` | [{MUSIC2EMO_UPSTREAM_REPO}](https://huggingface.co/{MUSIC2EMO_UPSTREAM_REPO}) `inference/data/btc_model_large_voca.pt` | `{MUSIC2EMO_UPSTREAM_REVISION}` |
| `mert/model.safetensors.zst` | [{MERT_UPSTREAM_REPO}](https://huggingface.co/{MERT_UPSTREAM_REPO}) `pytorch_model.bin` | `{MERT_UPSTREAM_REVISION}` |
| `mert/config.json` | [{MERT_UPSTREAM_REPO}](https://huggingface.co/{MERT_UPSTREAM_REPO}) `config.json` | `{MERT_UPSTREAM_REVISION}` |
| `mert/configuration_MERT.py` | [{MERT_UPSTREAM_REPO}](https://huggingface.co/{MERT_UPSTREAM_REPO}) | `{MERT_UPSTREAM_REVISION}` |
| `mert/modeling_MERT.py` | [{MERT_UPSTREAM_REPO}](https://huggingface.co/{MERT_UPSTREAM_REPO}) | `{MERT_UPSTREAM_REVISION}` |
| `mert/preprocessor_config.json` | [{MERT_UPSTREAM_REPO}](https://huggingface.co/{MERT_UPSTREAM_REPO}) | `{MERT_UPSTREAM_REVISION}` |

## Licensing

- Music2Emo checkpoints: Apache-2.0, see `LICENSE-MUSIC2EMO.txt`
  (attribution: AMAAI Lab, [amaai-lab/music2emo](https://huggingface.co/{MUSIC2EMO_UPSTREAM_REPO})).
- MERT files under `mert/`: CC-BY-NC-4.0, see `LICENSE-MERT.txt`
  (attribution: m-a-p, [MERT-v1-95M](https://huggingface.co/{MERT_UPSTREAM_REPO})).
  **Non-commercial use only.** The restriction applies to the pruned weights
  as derivative work.
"""


def build_staging(staging: Path) -> dict[str, str]:
    """Render every vendored file into ``staging``; return digest map."""
    staging_files: dict[str, bytes] = {
        "music2emo/J_all.safetensors": _j_all_into_safetensors(_fetch_music2emo_file("saved_models/J_all.ckpt")),
        "music2emo/btc_model_large_voca.safetensors": _btc_into_safetensors(
            _fetch_music2emo_file("inference/data/btc_model_large_voca.pt")
        ),
        "mert/model.safetensors.zst": _mert_into_zstd_safetensors(_fetch_mert_file("pytorch_model.bin")),
        "mert/config.json": _patch_mert_config(_fetch_mert_file("config.json")),
        "mert/configuration_MERT.py": _fetch_mert_file("configuration_MERT.py"),
        "mert/modeling_MERT.py": _fetch_mert_file("modeling_MERT.py"),
        "mert/preprocessor_config.json": _fetch_mert_file("preprocessor_config.json"),
        "LICENSE-MUSIC2EMO.txt": _fetch_text(APACHE2_URL).encode(),
        "LICENSE-MERT.txt": _fetch_text(CC_BY_NC4_URL).encode(),
        "README.md": _readme().encode(),
    }
    for path, payload in staging_files.items():
        target = staging / path
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_bytes(payload)

    # Whole-directory digests: these are the constants pinned in assets.py.
    return {name: dir_digest(staging, rels) for name, rels in BUNDLE_FILES.items()}


def upload(staging: Path) -> str:
    """Upload the staging directory; return the new ``main`` commit SHA."""
    api = HfApi()
    _ = api.create_repo(repo_id=VENDORED_REPO, repo_type="model", private=False, exist_ok=True)
    _ = api.upload_folder(repo_id=VENDORED_REPO, repo_type="model", folder_path=staging, path_in_repo=".")
    info = api.model_info(repo_id=VENDORED_REPO, revision="main")
    revision = info.sha
    if revision is None:
        raise RuntimeError("upload commit sha unavailable")
    return revision


def update_assets_pins(digests: dict[str, str]) -> None:
    """Write the bundle-dir digest pins into assets.py."""
    path = REPO_ROOT / "src" / "local_mood_playlist" / "assets.py"
    text = path.read_text(encoding="utf-8")
    for name in BUNDLE_FILES:
        marker = f'"{name}": "'
        start = text.index(marker)
        end = text.index('"', start + len(marker))
        text = text[: start + len(marker)] + digests[name] + text[end:]
    _ = path.write_text(text, encoding="utf-8")
    print("assets.py updated: bundle pins")


def main() -> None:
    description = __doc__.splitlines()[0] if __doc__ is not None else ""
    parser = argparse.ArgumentParser(description=description)
    _ = parser.add_argument("--upload", action="store_true", help="publish staging directory to HF")
    args = parser.parse_args()

    staging = CACHE_DIR / "music2emo-publish"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    digests = build_staging(staging)
    print(f"staging: {staging}")
    for path, digest in sorted(digests.items()):
        print(f"  {digest}  {path}")

    if args.upload:
        revision = upload(staging)
        print()
        print(f"published as revision {revision}")
        update_assets_pins(digests)
        print("Snapshot-dir digests now pinned in assets.py (matches this upload's bytes).")
        print('Set VENDORED_REVISION = "' + revision + '" in src/local_mood_playlist/assets.py.')


if __name__ == "__main__":
    main()
