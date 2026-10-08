import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModel

MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"


class FeatureExtractorMERT:
    def __init__(self, model_name="m-a-p/MERT-v1-95M", device = "None", sr=24000):
        self.model_name = model_name
        self.sr = sr
        if device == "None":
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = device
                
        self.model = AutoModel.from_pretrained(
            self.model_name,
            revision=MERT_REVISION,
            trust_remote_code=True,
        ).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name,
            revision=MERT_REVISION,
            trust_remote_code=True,
        )

    def extract_features_from_segment(self, segment, sample_rate, save_path):
        input_audio = segment.float()
        model_inputs = self.processor(input_audio, sampling_rate=sample_rate, return_tensors="pt")
        model_inputs = model_inputs.to(self.device)

        with torch.no_grad():
            model_outputs = self.model(**model_inputs, output_hidden_states=True)

        selected_hidden_states = torch.stack(
            (model_outputs.hidden_states[6], model_outputs.hidden_states[7])
        )
        all_layer_hidden_states = selected_hidden_states.mean(dim=2).permute(1, 0, 2)
        features = all_layer_hidden_states.cpu().detach().numpy()
        np.save(save_path, features)