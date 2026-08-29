"""Deterministic audio loading and Task 2 segment-level feature extraction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioFeatureConfig:
    """Parameters shared by graph-node and CNN mel extraction."""

    sample_rate: int = 22_050
    segment_seconds: float = 1.0
    n_mels: int = 128
    n_mfcc: int = 20
    n_fft: int = 2_048
    hop_length: int = 512

    def validate(self) -> None:
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if self.n_mels < 1 or self.n_mfcc < 1:
            raise ValueError("n_mels and n_mfcc must be positive")
        if self.n_fft < 16 or self.hop_length < 1:
            raise ValueError("n_fft must be at least 16 and hop_length must be positive")

    @property
    def node_feature_dim(self) -> int:
        # log-mel mean/std + chroma + MFCC/delta mean + RMS/centroid/ZCR
        return 2 * self.n_mels + 12 + 2 * self.n_mfcc + 3

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio(path: str | Path, sample_rate: int = 22_050) -> np.ndarray:
    """Load a finite mono waveform at the requested sample rate."""
    waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Audio file contains NaN or infinity: {path}")
    peak = float(np.max(np.abs(waveform)))
    return waveform / peak if peak > 0 else waveform


def split_segments(waveform: np.ndarray, config: AudioFeatureConfig) -> list[np.ndarray]:
    """Split into fixed windows and zero-pad the final segment."""
    config.validate()
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    window = int(round(config.sample_rate * config.segment_seconds))
    segments = [waveform[start : start + window] for start in range(0, len(waveform), window)]
    if not segments:
        segments = [np.zeros(window, dtype=np.float32)]
    return [
        np.pad(segment, (0, window - len(segment))).astype(np.float32, copy=False)
        for segment in segments
    ]


def _power_to_log_mel(power: np.ndarray) -> np.ndarray:
    reference = max(float(np.max(power)), 1e-10)
    return librosa.power_to_db(power, ref=reference, top_db=80.0)


def _mfcc_delta(mfcc: np.ndarray) -> np.ndarray:
    frames = mfcc.shape[1]
    width = min(9, frames if frames % 2 == 1 else frames - 1)
    if width < 3:
        return np.zeros_like(mfcc)
    return librosa.feature.delta(mfcc, width=width, mode="nearest")


def extract_segment_features(segment: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    """Return one finite feature vector for a fixed audio segment."""
    config.validate()
    mel_power = librosa.feature.melspectrogram(
        y=segment,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    log_mel = _power_to_log_mel(mel_power)
    chroma = librosa.feature.chroma_stft(
        y=segment,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=config.n_mfcc)
    delta = _mfcc_delta(mfcc)
    rms = librosa.feature.rms(y=segment, frame_length=config.n_fft, hop_length=config.hop_length)
    centroid = librosa.feature.spectral_centroid(
        y=segment,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=segment, frame_length=config.n_fft, hop_length=config.hop_length
    )
    vector = np.concatenate(
        [
            log_mel.mean(axis=1),
            log_mel.std(axis=1),
            chroma.mean(axis=1),
            mfcc.mean(axis=1),
            delta.mean(axis=1),
            [rms.mean(), centroid.mean(), zcr.mean()],
        ]
    ).astype(np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    if vector.shape != (config.node_feature_dim,):
        raise RuntimeError(
            f"Expected {config.node_feature_dim} node features, got {vector.shape}"
        )
    return vector


def extract_track_features_from_waveform(
    waveform: np.ndarray, config: AudioFeatureConfig
) -> np.ndarray:
    """Extract a ``[segments, feature_dim]`` node-feature matrix."""
    return np.stack(
        [extract_segment_features(segment, config) for segment in split_segments(waveform, config)]
    ).astype(np.float32)


def extract_track_features(path: str | Path, config: AudioFeatureConfig) -> np.ndarray:
    return extract_track_features_from_waveform(load_audio(path, config.sample_rate), config)


def extract_log_mel_from_waveform(
    waveform: np.ndarray, config: AudioFeatureConfig
) -> np.ndarray:
    """Return a clip-level log-mel image scaled approximately to ``[0, 1]``."""
    config.validate()
    power = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    log_mel = _power_to_log_mel(power)
    return np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)


def extract_log_mel(path: str | Path, config: AudioFeatureConfig) -> np.ndarray:
    return extract_log_mel_from_waveform(load_audio(path, config.sample_rate), config)
