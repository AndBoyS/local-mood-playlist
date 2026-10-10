import json
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import audioread
import librosa
import mir_eval
import numpy as np
import torch
import torchaudio.transforms as T
from safetensors import safe_open
from safetensors.torch import load_file
from music21 import note, stream

from .model.linear_mt_attn_ck import FeedforwardModelMTAttnCK
from .utils import logger
from .utils.btc_model import BTC_model
from .utils.hparams import HParams
from .utils.mert import FeatureExtractorMERT
from .utils.mir_eval_modules import audio_file_to_features, idx2voca_chord

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

pitch_num_dic = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

minor_major_dic2 = {
    'Db':'C#', 'Eb':'D#', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#'
}

shift_major_dic = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

shift_minor_dic = {
    'A': 0, 'A#': 1, 'B': 2, 'C': 3, 'C#': 4, 'D': 5,
    'D#': 6, 'E': 7, 'F': 8, 'F#': 9, 'G': 10, 'G#': 11,
}

segment_duration = 30
resample_rate = 24000
is_split = True
_MERT_BATCH_SIZE = 4
_BTC_BATCH_SIZE = 8
_MIN_MERT_TAIL_SECONDS = 0.1

def normalize_chord(file_path, key, key_type='major'):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if key == "None":
        new_key = "C major"
        shift = 0
    else:
        #print ("asdas",key)
        if len(key) == 1:
            key = key[0].upper()
        else:
            key = key[0].upper() + key[1:]

        if key in minor_major_dic2:
            key = minor_major_dic2[key]

        shift = 0

        if key_type == "major":
            new_key = "C major"

            shift = shift_major_dic[key]
        else:
            new_key = "A minor"
            shift = shift_minor_dic[key]

    converted_lines = []
    for line in lines:
        if line.strip():  # Skip empty lines
            parts = line.split()
            start_time = parts[0]
            end_time = parts[1]
            chord = parts[2]  # The chord is in the 3rd column
            if chord == "N":
                newchordnorm = "N"
            elif chord == "X":
                newchordnorm = "X"
            elif ":" in chord:
                pitch = chord.split(":")[0]
                attr = chord.split(":")[1]
                pnum = pitch_num_dic [pitch]
                new_idx = (pnum - shift)%12
                newchord = PITCH_CLASS[new_idx]
                newchordnorm = newchord + ":" + attr
            else:
                pitch = chord
                pnum = pitch_num_dic [pitch]
                new_idx = (pnum - shift)%12
                newchord = PITCH_CLASS[new_idx]
                newchordnorm = newchord

            converted_lines.append(f"{start_time} {end_time} {newchordnorm}\n")

    return converted_lines

def sanitize_key_signature(key):
    return key.replace('-', 'b')

def _load_audio(audio_path):
    if Path(audio_path).suffix.lower() != ".m4a":
        return librosa.load(audio_path, sr=None, mono=False)

    chunks = []
    with audioread.audio_open(str(audio_path)) as audio_file:
        sample_rate = audio_file.samplerate
        channel_count = audio_file.channels
        for frame in audio_file:
            chunks.append(np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0)

    waveform = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    if channel_count > 1:
        waveform = waveform.reshape((-1, channel_count)).T
    return waveform, sample_rate


def resample_waveform(waveform, original_sample_rate, target_sample_rate):
    if original_sample_rate != target_sample_rate:
        resampler = T.Resample(original_sample_rate, target_sample_rate)
        return resampler(waveform), target_sample_rate
    return waveform, original_sample_rate

def split_audio(waveform, sample_rate):
    segment_samples = segment_duration * sample_rate
    total_samples = waveform.size(0)
    segments = []
    # If the audio is shorter than the segment duration, just use the entire audio
    if total_samples <= segment_samples:
        segments.append(waveform)
    else:
        # Split the audio into segments of the specified duration
        for start in range(0, total_samples, segment_samples):
            end = min(start + segment_samples, total_samples)
            segment = waveform[start:end]
            segments.append(segment)

    # Ensure we have at least one segment
    if len(segments) == 0:
        segments.append(waveform)

    return segments

class Music2emo:
    def __init__(
        self,
        model_weights: str | Path | None = None,
        work_dir: str | Path | None = None,
        chord_model_weights: str | Path | None = None,
        *,
        mert_snapshot: str | Path,
    ) -> None:
        self.root = Path(__file__).resolve().parent
        self.model_weights = Path(model_weights) if model_weights is not None else self.root / "saved_models" / "J_all.safetensors"
        if not self.model_weights.is_file():
            raise FileNotFoundError(f"Music2Emotion checkpoint missing: {self.model_weights}")
        if work_dir is None:
            self._temporary_work_dir = tempfile.TemporaryDirectory(prefix="music2emotion-")
            self.work_dir = Path(self._temporary_work_dir.name)
        else:
            self._temporary_work_dir = None
            self.work_dir = Path(work_dir).expanduser().resolve()
            self.work_dir.mkdir(parents=True, exist_ok=True)

        requested_device = os.environ.get("SP_MUSIC2EMOTION_DEVICE", "auto").lower()
        if requested_device == "auto":
            if torch.cuda.is_available():
                requested_device = "cuda"
            elif torch.backends.mps.is_available():
                requested_device = "mps"
            else:
                requested_device = "cpu"
        if requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif requested_device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif requested_device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise RuntimeError(f"requested inference device is unavailable: {requested_device}")
        print(f"Music2Emotion device: {self.device}")

        self.feature_extractor = FeatureExtractorMERT(model_name=mert_snapshot, device=self.device, sr=resample_rate)

        self.music2emo_model = FeedforwardModelMTAttnCK(
            input_size= 768 * 2,
            output_size_classification=56,
            output_size_regression=2
        )

        state_dict = load_file(self.model_weights)

        # Adjust the keys in the state_dict
        state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
        # Filter state_dict to match model's keys
        model_keys = set(self.music2emo_model.state_dict().keys())
        filtered_state_dict = {key: value for key, value in state_dict.items() if key in model_keys}

        # Load the filtered state_dict and set the model to evaluation mode
        self.music2emo_model.load_state_dict(filtered_state_dict)

        self.music2emo_model.to(self.device)
        self.music2emo_model.eval()

        self.chord_config = HParams.load(self.root / "inference/data/run_config.yaml")
        self.chord_config.feature['large_voca'] = True
        self.chord_config.model['num_chords'] = 170
        chord_model_path = Path(chord_model_weights) if chord_model_weights is not None else self.root / "inference/data/btc_model_large_voca.safetensors"
        if not chord_model_path.is_file():
            raise FileNotFoundError(f"Music2Emotion chord weights missing: {chord_model_path}")
        chord_state = load_file(chord_model_path)
        with safe_open(chord_model_path, framework="pt") as f:
            chord_metadata = f.metadata() or {}
        self.chord_mean = float(chord_metadata['mean'])
        self.chord_std = float(chord_metadata['std'])
        self.chord_model = BTC_model(config=self.chord_config.model).to(self.device)
        self.chord_model.load_state_dict(chord_state)
        self.chord_model.eval()
        self.idx_to_chord = idx2voca_chord()

    def predict(self, audio: str | Path, threshold: float = 0.5) -> dict[str, object]:

        feature_dir = self.work_dir / "temp_out"

        if feature_dir.exists():
            shutil.rmtree(str(feature_dir))

        feature_dir.mkdir(parents=True)

        warnings.filterwarnings('ignore')
        logger.logging_verbosity(1)

        original_waveform, sample_rate = _load_audio(audio)
        waveform = torch.from_numpy(original_waveform)
        original_sample_rate = sample_rate
        audio_waveform_for_cqt = (
            librosa.to_mono(original_waveform)
            if original_waveform.ndim == 2
            else original_waveform
        )
        del original_waveform
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0).unsqueeze(0)
        waveform = waveform.squeeze()
        waveform, sample_rate = resample_waveform(waveform, sample_rate, resample_rate)

        segments = split_audio(waveform, sample_rate) if is_split else [waveform]
        segment_length = segment_duration * sample_rate
        full_segments = [segment for segment in segments if segment.size(0) == segment_length]
        partial_segments = [segment for segment in segments if segment.size(0) != segment_length]
        batched_embeddings = [
            self.feature_extractor.extract_features_from_segments(
                full_segments[start : start + _MERT_BATCH_SIZE], sample_rate
            )
            for start in range(0, len(full_segments), _MERT_BATCH_SIZE)
        ]
        for segment in partial_segments:
            try:
                embedding = self.feature_extractor.extract_features_from_segment(segment, sample_rate)
            except RuntimeError as error:
                is_tiny_tail = segment.size(0) < _MIN_MERT_TAIL_SECONDS * sample_rate
                is_convolution_length_error = "Kernel size: (3)" in str(error)
                if not (batched_embeddings and is_tiny_tail and is_convolution_length_error):
                    raise
                # Keep full-track CQT/chord features; omit only unembedable MERT tail.
            else:
                batched_embeddings.append(embedding)
        final_embedding_mert = torch.cat(batched_embeddings, dim=0).mean(dim=0)

        # --- Chord feature extract ---
        config = self.chord_config
        mean = self.chord_mean
        std = self.chord_std
        idx_to_chord = self.idx_to_chord
        model = self.chord_model

        audio_path = str(Path(audio).expanduser().resolve(strict=True))
        try:
            feature, feature_per_second, song_length_second = audio_file_to_features(
                audio_path,
                config,
                audio_waveform=audio_waveform_for_cqt,
                sample_rate=original_sample_rate,
            )
        except:
            logger.info("audio file failed to load : %s" % audio_path)
            assert(False)

        feature = feature.T
        feature = (feature - mean) / std
        time_unit = feature_per_second
        n_timestep = config.model['timestep']

        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep

        start_time = 0.0
        lines = []
        with torch.no_grad():
            model.eval()
            feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
            feature_chunks = feature.reshape(num_instance, n_timestep, feature.shape[-1])
            prediction_chunks: list[list[int]] = []
            for batch_start in range(0, num_instance, _BTC_BATCH_SIZE):
                feature_batch = feature_chunks[batch_start : batch_start + _BTC_BATCH_SIZE]
                self_attn_output, _ = model.self_attn_layers(feature_batch)
                prediction, _ = model.output_layer(self_attn_output)
                prediction_chunks.extend(
                    prediction.cpu().reshape(-1, n_timestep).tolist()
                )

            for t, prediction_indices in enumerate(prediction_chunks):
                for i in range(n_timestep):
                    if t == 0 and i == 0:
                        prev_chord = prediction_indices[i]
                        continue
                    if prediction_indices[i] != prev_chord:
                        lines.append(
                            '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                        start_time = time_unit * (n_timestep * t + i)
                        prev_chord = prediction_indices[i]
                    if t == num_instance - 1 and i + num_pad == n_timestep:
                        if start_time != time_unit * (n_timestep * t + i):
                            lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                        break

        save_path = os.path.join(feature_dir, os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav', '') + '.lab')
        with open(save_path, 'w') as f:
            for line in lines:
                f.write(line)

        # Build an in-memory score for key analysis (PrettyMIDI's default is 120 BPM).
        starts, ends, pitchs = list(), list(), list()

        intervals, chords = mir_eval.io.load_labeled_intervals(save_path)
        for p in range(12):
            for i, (interval, chord) in enumerate(zip(intervals, chords)):
                root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
                tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]
                if i == 0:
                    start_time = interval[0]
                    label = tmp_label
                    continue
                if tmp_label != label:
                    if label == 1.0:
                        starts.append(start_time), ends.append(interval[0]), pitchs.append(p + 48)
                    start_time = interval[0]
                    label = tmp_label
                if i == (len(intervals) - 1):
                    if label == 1.0:
                        starts.append(start_time), ends.append(interval[1]), pitchs.append(p + 48)

        key_score = stream.Stream()
        for start, end, pitch in zip(starts, ends, pitchs):
            key_note = note.Note(pitch)
            key_note.quarterLength = (end - start) * 2.0
            key_score.insert(start * 2.0, key_note)

        tonic_signatures = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        mode_signatures = ["major", "minor"]  # Major and minor modes

        tonic_to_idx = {tonic: idx for idx, tonic in enumerate(tonic_signatures)}
        mode_to_idx = {mode: idx for idx, mode in enumerate(mode_signatures)}
        idx_to_tonic = {idx: tonic for tonic, idx in tonic_to_idx.items()}
        idx_to_mode = {idx: mode for mode, idx in mode_to_idx.items()}

        with (self.root / 'inference/data/chord_root.json').open('r') as json_file:
            chordRootDic = json.load(json_file)
        with (self.root / 'inference/data/chord_attr.json').open('r') as json_file:
            chordAttrDic = json.load(json_file)

        try:
            key_signature = str(key_score.analyze('key'))
        except Exception as e:
            key_signature = "None"

        key_parts = key_signature.split()
        key_signature = sanitize_key_signature(key_parts[0])  # Sanitize key signature
        key_type = key_parts[1] if len(key_parts) > 1 else 'major'

        # --- Key feature (Tonic and Mode separation) ---
        if key_signature == "None":
            mode = "major"
        else:
            mode = key_signature.split()[-1]

        encoded_mode = mode_to_idx.get(mode, 0)
        mode_tensor = torch.tensor([encoded_mode], dtype=torch.long).to(self.device)

        converted_lines = normalize_chord(save_path, key_signature, key_type)

        lab_norm_path = save_path[:-4] + "_norm.lab"

        # Write the converted lines to the new file
        with open(lab_norm_path, 'w') as f:
            f.writelines(converted_lines)

        chords = []

        if not os.path.exists(lab_norm_path):
            chords.append((float(0), float(0), "N"))
        else:
            with open(lab_norm_path, 'r') as file:
                for line in file:
                    start, end, chord = line.strip().split()
                    chords.append((float(start), float(end), chord))

        encoded_root: list[int] = []
        encoded_attr: list[int] = []

        for start, end, chord in chords:
            chord_arr = chord.split(":")
            if len(chord_arr) == 1:
                if chord_arr[0] == "X":
                    chord_arr[0] = "N"
                chordRootID = chordRootDic[chord_arr[0]]
                if chord_arr[0] == "N" or chord_arr[0] == "X":
                    chordAttrID = 0
                else:
                    chordAttrID = 1
            elif len(chord_arr) == 2:
                chordRootID = chordRootDic[chord_arr[0]]
                chordAttrID = chordAttrDic[chord_arr[1]]
            encoded_root.append(chordRootID)
            encoded_attr.append(chordAttrID)

        encoded_chords_root = np.array(encoded_root)
        encoded_chords_attr = np.array(encoded_attr)

        max_sequence_length = 100
        if len(encoded_chords_root) > max_sequence_length:
            encoded_chords_root = encoded_chords_root[:max_sequence_length]
            encoded_chords_attr = encoded_chords_attr[:max_sequence_length]
        else:
            padding = [0] * (max_sequence_length - len(encoded_chords_root))
            encoded_chords_root = np.concatenate([encoded_chords_root, padding])
            encoded_chords_attr = np.concatenate([encoded_chords_attr, padding])

        chords_root_tensor = torch.tensor(encoded_chords_root, dtype=torch.long).to(self.device)
        chords_attr_tensor = torch.tensor(encoded_chords_attr, dtype=torch.long).to(self.device)

        model_input_dic = {
            "x_mert": final_embedding_mert.unsqueeze(0),
            "x_chord_root": chords_root_tensor.unsqueeze(0),
            "x_chord_attr": chords_attr_tensor.unsqueeze(0),
            "x_key": mode_tensor.unsqueeze(0)
        }

        model_input_dic = {k: v.to(self.device) for k, v in model_input_dic.items()}
        classification_output, regression_output = self.music2emo_model(model_input_dic)
        probs = torch.sigmoid(classification_output)

        tag_list = np.load(self.root / "inference/data/tag_list.npy")[127:]
        mood_list = [str(tag).replace("mood/theme---", "") for tag in tag_list]
        score_values = probs.squeeze().tolist()
        if len(tag_list) != len(score_values):
            raise RuntimeError("Music2Emotion output size does not match canonical mood tag list")
        predicted_moods = [mood_list[i] for i, score in enumerate(score_values) if score > threshold]
        mood_scores = {str(tag): float(score) for tag, score in zip(tag_list, score_values, strict=True)}

        valence, arousal = regression_output.squeeze().tolist()

        model_output_dic = {
            "valence": valence,
            "arousal": arousal,
            "predicted_moods": predicted_moods,
            "mood_scores": mood_scores,
        }

        return model_output_dic
