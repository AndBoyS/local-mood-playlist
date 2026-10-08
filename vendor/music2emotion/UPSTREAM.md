# Vendored Music2Emotion

Source: [AMAAI-Lab/Music2Emotion](https://github.com/AMAAI-Lab/Music2Emotion)

Pinned upstream commit: `c5f7984f78d9928b841f25bd05a47377c12c58ca` (2025-08-24).

This is a sparse copy of upstream inference code, inference data, and the default
`J_all.ckpt` model. Upstream MIT license is retained in `LICENSE`.

Local patch in `music2emo.py`:

- Resolve model/config/tag assets relative to this source tree, not process cwd.
- Put scratch output under caller-owned temporary work directory.
- Decode M4A through librosa/FFmpeg where torchaudio's SoundFile backend cannot.
- Select Apple MPS between CUDA and CPU; load checkpoints on CPU before moving models.
- Stop MERT after transformer layer 7; classifier only consumes hidden states 6 and 7.
- Keep segment embeddings on the selected device and average before device transfer; avoid per-segment NPY I/O.
- Return raw canonical mood-tag probabilities alongside upstream thresholded tags.
- Remove unused demo/training imports from inference module.
- Pin MERT repository revision to `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.

The Music2Emotion code is MIT-licensed, but the MERT checkpoint downloaded from
Hugging Face is licensed CC BY-NC 4.0. This dependency restricts use to
non-commercial settings unless separate permission is obtained.

Dependency subset lives in [`../../inference/requirements.txt`](../../inference/requirements.txt).
It targets Python 3.10 because upstream pins PyTorch 2.3.1. Keep this runtime
separate from app's Python 3.13 environment. MERT weights download from Hugging
Face on first initialization.
