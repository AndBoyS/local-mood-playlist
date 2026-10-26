"""MERT feature extraction on top of the pure-torch backbone in ``mert_model``.

Replaces the upstream transformers-based loader: the model and preprocessing
behaviour (zero-mean unit-variance normalization, matching
``Wav2Vec2FeatureExtractor`` with ``do_normalize=True``) are reproduced
locally, so no Hugging Face runtime is needed.
"""

from pathlib import Path
from typing import Final

import numpy as np
import torch
from local_mood_playlist.vendor.mert_model import load_mert_backbone, mert_sample_rate_check

_HIDDEN_STATES: Final = (6, 7)

class FeatureExtractorMERT:
    def __init__(self, local_path: Path, device: torch.device, sr: int) -> None:
        mert_sample_rate_check(sr, source="FeatureExtractorMERT")
        self.sr = sr
        self.device = torch.device(device)
        mert_dir = local_path.resolve()
        if not mert_dir.is_dir():
            raise FileNotFoundError(f"MERT snapshot directory missing: {mert_dir}")

        # fp16 - good speedup, negligible feature change on MPS
        # not tested on CUDA
        self.inference_dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.model = load_mert_backbone(mert_dir, device=self.device, dtype=self.inference_dtype)

    def extract_features_from_segments(self, segments, sample_rate):
        mert_sample_rate_check(sample_rate, source="FeatureExtractorMERT.extract_features_from_segments")
        input_audio = [segment.float().detach().cpu().numpy() for segment in segments]
        normalized = [_zero_mean_unit_variance(vector) for vector in input_audio]
        # from_numpy zero copy, then convert to device
        input_values = torch.from_numpy(np.stack(normalized)).to(self.device, dtype=self.inference_dtype)

        with torch.no_grad():
            _, hidden_states = self.model.forward(input_values)

        selected_hidden_states = torch.stack(tuple(hidden_states[i] for i in _HIDDEN_STATES))
        all_layer_hidden_states = selected_hidden_states.mean(dim=2).permute(1, 0, 2)
        return all_layer_hidden_states.reshape(all_layer_hidden_states.shape[0], -1).float()

    def extract_features_from_segment(self, segment, sample_rate):
        return self.extract_features_from_segments([segment], sample_rate)


def _zero_mean_unit_variance(vector: np.ndarray) -> np.ndarray:
    # Wav2Vec2FeatureExtractor normalization with do_normalize=True.
    return (vector - vector.mean()) / np.sqrt(vector.var() + 1e-7)
