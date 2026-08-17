"""Audio loading, segmentation, and segment-level features."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import librosa
import numpy as np

@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 22_050
    segment_seconds: float = 5.0
    n_mels: int = 128

def load_audio(path: str | Path, sample_rate: int = 22_050) -> np.ndarray:
    waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return librosa.util.normalize(waveform)

def split_segments(waveform: np.ndarray, config: AudioFeatureConfig) -> list[np.ndarray]:
    window = int(config.sample_rate * config.segment_seconds)
    if window <= 0:
        raise ValueError("segment_seconds must be positive")
    segments = [waveform[i : i + window] for i in range(0, len(waveform), window)] or [waveform]
    return [np.pad(x, (0, max(0, window - len(x)))) for x in segments]

def extract_segment_features(segment: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=segment, sr=config.sample_rate, n_mels=config.n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=segment, sr=config.sample_rate)
    features = np.concatenate((log_mel.mean(axis=1), chroma.mean(axis=1))).astype(np.float32)
    return (features - features.mean()) / (features.std() + 1e-8)

def extract_track_features(path: str | Path, config: AudioFeatureConfig) -> np.ndarray:
    waveform = load_audio(path, config.sample_rate)
    return np.stack([extract_segment_features(segment, config) for segment in split_segments(waveform, config)])
