# Vendored Music2Emotion

Source: [AMAAI-Lab/Music2Emotion](https://github.com/AMAAI-Lab/Music2Emotion)

Pinned upstream commit: `c5f7984f78d9928b841f25bd05a47377c12c58ca` (2025-08-24).

This is a sparse copy of upstream inference code, inference data, and the default
`J_all.ckpt` model. Unused upstream training and preprocessing modules
(`preprocess/`, demo helpers under `utils/`, and unused `model/` variants) are
pruned. Upstream MIT license is retained in `LICENSE`.

The large model payloads are not stored in git. They are pre-transformed
(`state_dict` safetensors conversion, scalar metadata extraction, MERT trimmed
to 7 transformer layers + zstd, config patch) by
`scripts/publish_assets.py` once per upstream revision and published to the
vendored Hugging Face repo `music2emo-inference-assets`. On first use
`src/local_smart_playlist/assets.py` downloads the pinned vendored revision,
verifies whole-directory digests, and caches under the project `.cache`
directory. The pickle checkpoints from upstream are never executed.

Local patch in `music2emo.py`:

- Resolve model/config/tag assets relative to this source tree, not process cwd.
- Put scratch output under caller-owned temporary work directory.
- Decode M4A through librosa/FFmpeg where torchaudio's SoundFile backend cannot.
- Select Apple MPS between CUDA and CPU; load checkpoints on CPU before moving models.
- Stop MERT after transformer layer 7; classifier only consumes hidden states 6 and 7.
- Batch equal-length 30-second MERT segments in groups of four; process shorter tail singly.
- Keep segment embeddings on the selected device and average before device transfer; avoid per-segment NPY I/O.
- Transfer BTC chord predictions to CPU once per 10-second window, not once per frame.
- Drop unused chord class-index tensor; retain model-required chord-root, chord-attribute, and key features.
- Return raw canonical mood-tag probabilities alongside upstream thresholded tags.
- Load checkpoints from safe `safetensors` inputs: whole-state-dict safetensors
  are never executed as pickles.
- Load MERT from a project-cache snapshot (``assets.py``): vendored zstd
  archive of the 7 transformer layers inference consumes, decompressed to disk
  only during model construction; layer count already patched in the vendored
  `config.json`; identical outputs to the untruncated hub checkpoint
  (bit-identical hidden states).
- Remove unused demo/training imports from inference module.
- Require the project-cache MERT snapshot at construction; no hub fallback
  (a missing staged snapshot fails loudly instead of fetching an un-pinned
  revision from Hugging Face).

The Music2Emotion code is MIT-licensed, but the MERT checkpoint downloaded from
Hugging Face is licensed CC BY-NC 4.0. This dependency restricts use to
non-commercial settings unless separate permission is obtained.

Runtime dependencies are managed by the main project environment in
`pyproject.toml`/`uv.lock`. This project uses Python 3.13-compatible PyTorch
and Transformers versions; MERT weights are managed by the project asset cache
(see `src/local_smart_playlist/assets.py`).
