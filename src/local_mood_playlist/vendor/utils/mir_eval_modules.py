# Pruned upstream module: only the two helpers the inference entrypoint uses
# (idx2voca_chord, audio_file_to_features). Upstream training/eval scoring
# helpers are dropped; see UPSTREAM.md.

import librosa
import numpy as np

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

def idx2voca_chord():
    idx2voca_chord = {}
    idx2voca_chord[169] = 'N'
    idx2voca_chord[168] = 'X'
    for i in range(168):
        root = i // 14
        root = root_list[root]
        quality = i % 14
        quality = quality_list[quality]
        if i % 14 != 1:
            chord = root + ':' + quality
        else:
            chord = root
        idx2voca_chord[i] = chord
    return idx2voca_chord


def audio_file_to_features(
    audio_file,
    config,
    audio_waveform: np.ndarray | None = None,
    sample_rate: int | None = None,
):
    if audio_waveform is None:
        original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when reusing decoded audio")
        original_wav = np.asarray(audio_waveform)
        if original_wav.ndim == 2:
            original_wav = librosa.to_mono(original_wav)
        elif original_wav.ndim != 1:
            raise ValueError("audio_waveform must have shape (samples,) or (channels, samples)")
        sr = config.mp3['song_hz']
        if sample_rate != sr:
            original_wav = librosa.resample(original_wav, orig_sr=sample_rate, target_sr=sr)
    song_length_second = len(original_wav) / sr
    inst_samples = int(sr * config.mp3['inst_len'])
    currunt_sec_hz = 0
    feature = None

    while currunt_sec_hz + inst_samples <= len(original_wav):
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + inst_samples)
        tmp = librosa.cqt(
            original_wav[start_idx:end_idx],
            sr=sr,
            n_bins=config.feature['n_bins'],
            bins_per_octave=config.feature['bins_per_octave'],
            hop_length=config.feature['hop_length']
        )
        feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
        currunt_sec_hz = end_idx

    # Handle remaining samples (including case where audio < inst_len)
    if currunt_sec_hz < len(original_wav):
        tmp = librosa.cqt(
            original_wav[currunt_sec_hz:],
            sr=sr,
            n_bins=config.feature['n_bins'],
            bins_per_octave=config.feature['bins_per_octave'],
            hop_length=config.feature['hop_length']
        )
        feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)

    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config.mp3['inst_len'] / config.model['timestep']

    return feature, feature_per_second, song_length_second
