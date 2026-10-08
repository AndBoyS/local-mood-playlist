import torch
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
        # Downstream classifier consumes only hidden states 6 and 7.
        # Hidden state 7 is output of encoder layer 7; omit unused layers 8-12.
        self.model.encoder.layers = torch.nn.ModuleList(list(self.model.encoder.layers[:7]))
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name,
            revision=MERT_REVISION,
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