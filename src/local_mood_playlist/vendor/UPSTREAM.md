# Vendored Music2Emotion

Source: [AMAAI-Lab/Music2Emotion](https://github.com/AMAAI-Lab/Music2Emotion)

Pinned upstream commit: `c5f7984f78d9928b841f25bd05a47377c12c58ca` (2025-08-24).

This is a sparse copy of upstream inference code, inference data, and the default
`J_all.ckpt` model. Unused upstream training and preprocessing modules
(`preprocess/`, demo helpers under `utils/`, and unused `model/` variants) are
pruned, as are the unused per-file eval helpers in `utils/mir_eval_modules.py`
and `utils/logger.py` plus the unused
`inference/data/chord.json` and `inference/data/chord_inv.json` lookup tables.
Upstream MIT license is retained in `LICENSE`.

The large model payloads are not stored in git. They are pre-transformed
(`state_dict` safetensors conversion, scalar metadata extraction, MERT trimmed
to 7 transformer layers + zstd, config patch) by
`scripts/publish_assets.py` once per upstream revision and published to the
vendored Hugging Face repo `music2emo-inference-assets`. On first use
`src/local_mood_playlist/assets.py` downloads the pinned vendored revision,
verifies per-file digests, and caches under the project `.cache`
directory. The pickle checkpoints from upstream are never executed.

Local patch in `music2emo.py`:

- Resolve model/config/tag assets relative to this source tree, not process cwd.
- Put scratch output under caller-owned temporary work directory.
- Decode M4A through librosa/FFmpeg where torchaudio's SoundFile backend cannot.
- Select Apple MPS between CUDA and CPU; load checkpoints on CPU before moving models.
- Stop MERT after transformer layer 7; classifier only consumes hidden states 6 and 7.
- Batch equal-length 30-second MERT segments in groups of four; process shorter tail singly.
- Keep segment embeddings on the selected device and average before device transfer; avoid per-segment NPY I/O.
- Split per-track work into CPU-only staging (`prepare`: decode, resample,
  segment split, CQT chord features) and GPU inference (`infer`), so callers
  can stage one track ahead on a worker thread while the GPU predicts the
  current one; `predict` remains the fused single-track path.
- Transfer BTC chord predictions to CPU once per track, not once per 10-second window batch.
- Drop unused chord class-index tensor; retain model-required chord-root, chord-attribute, and key features.
- Return raw canonical mood-tag probabilities alongside upstream thresholded tags.
- Load checkpoints from safe `safetensors` inputs: whole-state-dict safetensors
  are never executed as pickles.
- Load MERT from a project-cache snapshot (`assets.py`): vendored zstd
  archive of the 7 transformer layers inference consumes, decompressed to disk
  only during model construction; layer count already patched in the vendored
  `config.json`; identical outputs to the untruncated hub checkpoint
  (bit-identical hidden states).
- Run MERT through the pure-torch reimplementation in
  `local_mood_playlist.mert_model` (numeric port of the HuBERT feature
  encoder, encoder, and encoder-layer the checkpoint weights target, with
  `Wav2Vec2FeatureExtractor` normalization reproduced inline): fp32 CPU and
  MPS fp16 outputs are bit-identical to the transformers build of the same
  checkpoint.
  This removes the transformers/huggingface_hub runtime entirely.
- Remove unused demo/training imports from inference module.
- Require the project-cache MERT snapshot at construction; no hub fallback
  (a missing staged snapshot fails loudly instead of fetching an un-pinned
  revision from Hugging Face).

The Music2Emotion local patch also prunes dead upstream training paths from
the vendored inference modules: `BTC_model.forward` (and its `loss`/
`probs_out` chain through `OutputLayer`/`SoftmaxOutputLayer`), the unused
scoring helpers, and the unused per-tensor flags. Only the layer modules the
inference entrypoint calls (`self_attn_layers`, `output_layer`) remain.

The Music2Emotion code is MIT-licensed, but the MERT checkpoint downloaded from
Hugging Face is licensed CC BY-NC 4.0. This dependency restricts use to
non-commercial settings unless separate permission is obtained.

Runtime dependencies are managed by the main project environment in
`pyproject.toml`/`uv.lock`. This project uses a Python 3.13-compatible PyTorch
version; MERT weights are managed by the project asset cache (see
`src/local_mood_playlist/assets.py`) and run through the local pure-torch
backbone (`local_mood_playlist.mert_model`), so transformers and
huggingface_hub are not runtime dependencies.
