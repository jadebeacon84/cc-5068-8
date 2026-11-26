"""Feature extraction: local STFT + log-mel filterbank, fully inspectable.

No deep-learning framework is required. The model consumes fixed 1-second
mel-patch windows; the exact same patch shown as "evidence" in the UI is what
the classifier scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft
from scipy.signal import get_window

from .config import AudioConfig


def mel_filterbank(cfg: AudioConfig) -> np.ndarray:
    """Return an (n_mels, n_fft//2+1) triangular mel filterbank matrix."""
    n_bins = cfg.n_fft // 2 + 1
    hz = np.linspace(0, cfg.sample_rate / 2, n_bins)

    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax), cfg.n_mels + 2)
    points = mel_to_hz(mels)
    fb = np.zeros((cfg.n_mels, n_bins), dtype=np.float32)
    for i in range(cfg.n_mels):
        left, center, right = points[i], points[i + 1], points[i + 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            up = (hz - left) / max(center - left, 1e-9)
            down = (right - hz) / max(right - center, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def log_mel_spectrogram(audio: np.ndarray, cfg: AudioConfig,
                        filterbank: np.ndarray | None = None) -> np.ndarray:
    """Return log-mel spectrogram of shape (n_mels, n_frames)."""
    if filterbank is None:
        filterbank = mel_filterbank(cfg)
    window = get_window("hann", cfg.n_fft, fftbins=True).astype(np.float32)
    n_frames = 1 + max(0, (len(audio) - cfg.n_fft) // cfg.hop_length)
    if n_frames <= 0:
        pad = cfg.n_fft - len(audio)
        audio = np.pad(audio, (0, max(0, pad)))
        n_frames = 1
    spec = np.empty((cfg.n_fft // 2 + 1, n_frames), dtype=np.float32)
    for t in range(n_frames):
        start = t * cfg.hop_length
        frame = audio[start:start + cfg.n_fft]
        if len(frame) < cfg.n_fft:
            frame = np.pad(frame, (0, cfg.n_fft - len(frame)))
        spectrum = rfft(frame * window)
        spec[:, t] = np.abs(spectrum).astype(np.float32) ** 2
    mel = filterbank @ spec
    return np.log1p(mel).astype(np.float32)


def window_starts(n_frames: int, cfg: AudioConfig) -> list[int]:
    """All analysis-window start frame indices, last window clamped to the end."""
    if n_frames <= cfg.win_frames:
        return [0]
    starts = list(range(0, n_frames - cfg.win_frames + 1, cfg.hop_frames))
    last = n_frames - cfg.win_frames
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_windows(spectrogram: np.ndarray, cfg: AudioConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened mel patches (n_windows, n_mels*win_frames) and start frames."""
    n_mels, n_frames = spectrogram.shape
    starts = np.asarray(window_starts(n_frames, cfg), dtype=np.int64)
    patches = np.empty((len(starts), n_mels * cfg.win_frames), dtype=np.float32)
    for i, start in enumerate(starts):
        patches[i] = spectrogram[:, start:start + cfg.win_frames].reshape(-1)
    return patches, starts


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, patches: np.ndarray) -> "Standardizer":
        mean = patches.mean(axis=0)
        scale = patches.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, patches: np.ndarray) -> np.ndarray:
        return (patches - self.mean) / self.scale

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, raw: dict) -> "Standardizer":
        return cls(mean=np.asarray(raw["mean"], dtype=np.float32),
                   scale=np.asarray(raw["scale"], dtype=np.float32))


def frame_times(starts: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    return starts.astype(np.float64) * cfg.frame_seconds
