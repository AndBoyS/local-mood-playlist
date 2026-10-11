import torch
from pathlib import Path
import numpy as np

from transformers import Wav2Vec2FeatureExtractor, AutoModel


class FeatureExtractorMERT:
    def __init__(self, local_path: Path, device: torch.device, sr: int) -> None:
        self.sr = sr
        self.device = torch.device(device)
        mert_dir = local_path.resolve()
        if not mert_dir.is_dir():
            raise FileNotFoundError(f"MERT snapshot directory missing: {mert_dir}")
        self.model = AutoModel.from_pretrained(
            mert_dir,
            trust_remote_code=True,
        ).to(self.device)

        # Downstream classifier consumes only hidden states 6 and 7.
        # They are outputs of encoder layers 5 and 6; omit unused layers 7-11.
        self.model.encoder.layers = torch.nn.ModuleList(list(self.model.encoder.layers[:7]))
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            mert_dir,
            trust_remote_code=True,
        )

        # fp16 - good speedup, negligible feature change on MPS
        # not tested on CUDA
        self.inference_dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.model = self.model.to(self.inference_dtype)

    def extract_features_from_segments(self, segments, sample_rate):
        input_audio = [segment.float().detach().cpu().numpy() for segment in segments]
        normalized: np.ndarray = self.processor(input_audio, sampling_rate=sample_rate)["input_values"]
        # from_numpy zero copy, then convert to device
        input_values = torch.from_numpy(np.stack(normalized)).to(self.device, dtype=self.inference_dtype)

        with torch.no_grad():
            model_outputs = self.model(input_values, output_hidden_states=True)

        selected_hidden_states = torch.stack(
            (model_outputs.hidden_states[6], model_outputs.hidden_states[7])
        )
        all_layer_hidden_states = selected_hidden_states.mean(dim=2).permute(1, 0, 2)
        return all_layer_hidden_states.reshape(all_layer_hidden_states.shape[0], -1).float()

    def extract_features_from_segment(self, segment, sample_rate):
        return self.extract_features_from_segments([segment], sample_rate)
