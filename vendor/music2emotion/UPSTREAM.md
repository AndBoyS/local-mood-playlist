# Vendored Music2Emotion

Source: [AMAAI-Lab/Music2Emotion](https://github.com/AMAAI-Lab/Music2Emotion)

Pinned upstream commit: `c5f7984f78d9928b841f25bd05a47377c12c58ca` (2025-08-24).

This is a sparse copy of upstream inference code, inference data, and the default
`J_all.ckpt` model. Unused upstream training and preprocessing modules
(`preprocess/`, demo helpers under `utils/`, and unused `model/` variants) are
pruned. Upstream MIT license is retained in `LICENSE`.

The large binary checkpoints (`saved_models/J_all.ckpt`,
`inference/data/btc_model_large_voca.pt`) are not stored in git. They are
downloaded on first use from the pinned Hugging Face revision
`amaai-lab/music2emo@b036e59471583c3d5b30c69e63e8c7323cc36c4a` and cached under
the project `.cache` directory; see `src/local_smart_playlist/assets.py` for
the pinned SHA-256 digests.

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
- Remove unused demo/training imports from inference module.
- Pin MERT repository revision to `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.

The Music2Emotion code is MIT-licensed, but the MERT checkpoint downloaded from
Hugging Face is licensed CC BY-NC 4.0. This dependency restricts use to
non-commercial settings unless separate permission is obtained.

Runtime dependencies are managed by the main project environment in
`pyproject.toml`/`uv.lock`. This project uses Python 3.13-compatible PyTorch
and Transformers versions; MERT weights download from Hugging Face on first
initialization.
