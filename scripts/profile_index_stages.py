"""Line/stage profile of Music2Emotion predict on a real test library.

Breaks each predict() into stages: decode, resample, MERT, CQT, chord model,
key analysis, and the final feed-forward. Prints per-track and per-stage timings.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from local_mood_playlist.vendor.music2emo import Music2emo  # noqa: E402
from local_mood_playlist.vendor.utils.mir_eval_modules import audio_file_to_features  # noqa: E402

CACHE_ROOT = Path(".cache")
LIBRARY = Path("temp/test-lib")

t0 = time.perf_counter()
with open(CACHE_ROOT / "profile-data.npz", "wb"):
    pass

model = Music2emo(
    model_weights=CACHE_ROOT / "assets/50b68ba9b28c03c9acbc0b18c6fefe3c4ccdda59/music2emo/J_all.safetensors",
    chord_model_weights=CACHE_ROOT / "assets/50b68ba9b28c03c9acbc0b18c6fefe3c4ccdda59/music2emo/btc_model_large_voca.safetensors",
    mert_snapshot=CACHE_ROOT / "assets/50b68ba9b28c03c9acbc0b18c6fefe3c4ccdda59/mert.zst-decompressed",
    device="auto",
)
print(f"model load: {time.perf_counter() - t0:.2f}s", file=sys.stderr)

FILES = sorted(p for p in LIBRARY.rglob("*.mp3"))
if len(sys.argv) > 1:
    FILES = [p for p in FILES if sys.argv[1] in p.name][:1]

stage_names = (
    "decode",
    "resample",
    "mert",
    "cqt",
    "chord_attn",
    "key_read_json",
    "key_analyze",
    "ff_model",
    "misc",
)
totals = {name: 0.0 for name in stage_names}
grand = 0.0

from contextlib import contextmanager


@contextmanager
def stage(name: str):
    global grand
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        totals[name] += elapsed
        grand += elapsed


for track in FILES:
    track_start = time.perf_counter()
    track_grand = grand

    # --- decode ---
    with stage("decode"):
        waveform_np, original_sample_rate = (
            __import__("local_mood_playlist.vendor.music2emo", fromlist=["_load_audio"])._load_audio(track)
        )
        original_waveform = waveform_np

    with stage("resample"):
        import librosa

        waveform_t = torch.from_numpy(original_waveform)
        if waveform_t.ndim == 1:
            waveform_t = waveform_t.unsqueeze(0)
        if waveform_t.shape[0] > 1:
            waveform_t = waveform_t.mean(dim=0).unsqueeze(0)
        waveform_t = waveform_t.squeeze()
        if original_sample_rate != model.feature_extractor.sr:
            import torchaudio.transforms as T

            waveform_t = T.Resample(original_sample_rate, model.feature_extractor.sr)(waveform_t)
        audio_waveform_for_cqt = (
            librosa.to_mono(original_waveform) if original_waveform.ndim == 2 else original_waveform
        )
        segments = []
        seg_len = 30 * model.feature_extractor.sr
        total = waveform_t.size(0)
        if total <= seg_len:
            segments.append(waveform_t)
        else:
            for s in range(0, total, seg_len):
                segments.append(waveform_t[s : min(s + seg_len, total)])

    with stage("mert"):
        full_segments = [s for s in segments if s.size(0) == seg_len]
        partial_segments = [s for s in segments if s.size(0) != seg_len]
        batched = [
            model.feature_extractor.extract_features_from_segments(
                full_segments[i : i + 4], model.feature_extractor.sr
            )
            for i in range(0, len(full_segments), 4)
        ]
        for segment in partial_segments:
            batched.append(model.feature_extractor.extract_features_from_segment(segment, model.feature_extractor.sr))
        final_embedding_mert = torch.cat(batched, dim=0).mean(dim=0)

    # --- cqt ---
    with stage("cqt"):
        config = model.chord_config
        feature, feature_per_second, song_length_second = audio_file_to_features(
            str(track),
            config,
            audio_waveform=audio_waveform_for_cqt,
            sample_rate=original_sample_rate,
        )
        feature = feature.T
        mean, std = model.chord_mean, model.chord_std
        feature = (feature - mean) / std
        n_timestep = config.model["timestep"]
        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep

    with stage("chord_attn"):
        feature_t = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(model.device)
        feature_chunks = feature_t.reshape(num_instance, n_timestep, feature_t.shape[-1])
        prediction_chunks: list[list[int]] = []
        for batch_start in range(0, num_instance, 8):
            feature_batch = feature_chunks[batch_start : batch_start + 8]
            with torch.no_grad():
                self_attn_output, _ = model.chord_model.self_attn_layers(feature_batch)
                prediction, _ = model.chord_model.output_layer(self_attn_output)
            prediction_chunks.extend(prediction.cpu().reshape(-1, n_timestep).tolist())

    with stage("key_read_json"):
        import json

        with (Path("src/local_mood_playlist/vendor") / "inference/data/chord_root.json").open("rb") as f:
            chord_root_dic = json.load(f)
        with (Path("src/local_mood_playlist/vendor") / "inference/data/chord_attr.json").open("rb") as f:
            chord_attr_dic = json.load(f)

    with stage("key_analyze"):
        import music21  # noqa: F401
        from music21 import note, stream

        starts, ends, pitchs = [], [], []
        intervals, chords = None, None  # replaced below

        # replicate the mir_eval re-load + key analysis
        feature_dir = model.work_dir / "temp_out"
        feature_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(feature_dir / "profile.lab")
        with open(save_path, "w") as f:
            f.writelines(
                "%.3f %.3f %s\n" % (i * feature_per_second, (i + 1) * feature_per_second, "N")
                for i in range(len(prediction_chunks) - 1)
            )
        # use mirror of upstream: we skip actual chord label writing detail
        import mir_eval

        intervals, chords = mir_eval.io.load_labeled_intervals(save_path)
        for p in range(12):
            for i, (interval, chord) in enumerate(zip(intervals, chords)):
                root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
                tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]
                if i == 0:
                    start_time_v = interval[0]
                    label = tmp_label
                    continue
                if tmp_label != label:
                    if label == 1.0:
                        starts.append(start_time_v), ends.append(interval[0]), pitchs.append(p + 48)
                    start_time_v = interval[0]
                    label = tmp_label
                if i == (len(intervals) - 1):
                    if label == 1.0:
                        starts.append(start_time_v), ends.append(interval[1]), pitchs.append(p + 48)

        key_score = stream.Stream()
        for start, end, pitch in zip(starts, ends, pitchs):
            key_note = note.Note(pitch)
            key_note.quarterLength = (end - start) * 2.0
            key_score.insert(start * 2.0, key_note)
        tonic_signatures = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        mode_signatures = ["major", "minor"]
        mode_to_idx = {m: idx for idx, m in enumerate(mode_signatures)}
        try:
            key_signature = str(key_score.analyze("key"))
        except Exception:
            key_signature = "None"
        mode = "major" if key_signature == "None" else key_signature.split()[-1]
        _mode_tensor = torch.tensor([mode_to_idx.get(mode, 0)], dtype=torch.long).to(model.device)

    with stage("ff_model"):
        encoded_mode = mode_to_idx.get(mode, 0)
        mode_tensor = torch.tensor([encoded_mode], dtype=torch.long).to(model.device)
        # root/attr encoding from prediction chunks (approximation, cost-equivalent)
        # valid embedding IDs: root in [0, 12), attr in [0, 14) — shape-equivalent to real pipeline
        encoded_root = [c % 12 for c in (chunk[0] for chunk in prediction_chunks[:100])]
        encoded_attr = [(c // 12) % 14 for c in (chunk[0] for chunk in prediction_chunks[:100])]
        if len(encoded_root) < 100:
            encoded_root += [0] * (100 - len(encoded_root))
            encoded_attr += [0] * (100 - len(encoded_attr))
        model_input = {
            "x_mert": final_embedding_mert.unsqueeze(0).to(model.device),
            "x_chord_root": torch.tensor([encoded_root], dtype=torch.long).to(model.device),
            "x_chord_attr": torch.tensor([encoded_attr], dtype=torch.long).to(model.device),
            "x_key": mode_tensor.unsqueeze(0),
        }
        with torch.no_grad():
            classification_output, regression_output = model.music2emo_model(model_input)

    totals["misc"] += time.perf_counter() - track_start - (grand - track_grand)
    print(
        f"{track.name[:44]:44s} total={time.perf_counter() - track_start:6.2f}s song={song_length_second:6.0f}s",
        file=sys.stderr,
    )

width = max(len(n) for n in stage_names)
print("\n=== stage totals (all tracks) ===", file=sys.stderr)
for name in stage_names:
    print(f"{name:{width}} {totals[name]:8.2f}s {100 * totals[name] / max(grand, 1e-9):5.1f}%", file=sys.stderr)
print(f"{'TOTAL':{width}} {grand:8.2f}s", file=sys.stderr)
