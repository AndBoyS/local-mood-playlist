import torch
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, AutoModel


class FeatureExtractorMERT:
    def __init__(self, model_name: str | Path, device: torch.device, sr: int) -> None:
        """Load the project-cached MERT snapshot on the requested device.

        ``model_name`` must be the local snapshot directory staged by
        ``assets.stage_mert_weights``; a missing snapshot directory fails
        construction instead of silently fetching from the hub.
        """
        self.sr = sr
        self.device = torch.device(device)
        mert_dir = Path(model_name).resolve()
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

    def extract_features_from_segments(self, segments, sample_rate):
        input_audio = [segment.float().detach().cpu().numpy() for segment in segments]
        model_inputs = self.processor(input_audio, sampling_rate=sample_rate, return_tensors="pt")
        model_inputs = model_inputs.to(self.device)

        with torch.no_grad():
            model_outputs = self.model(**model_inputs, output_hidden_states=True)

        selected_hidden_states = torch.stack(
            (model_outputs.hidden_states[6], model_outputs.hidden_states[7])
        )
        all_layer_hidden_states = selected_hidden_states.mean(dim=2).permute(1, 0, 2)
        return all_layer_hidden_states.reshape(all_layer_hidden_states.shape[0], -1)

    def extract_features_from_segment(self, segment, sample_rate):
        return self.extract_features_from_segments([segment], sample_rate)
