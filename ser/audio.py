"""Local audio I/O and waveform helpers (WAV only, no network services)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .config import AudioConfig


def load_wav(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32 in [-1, 1], optionally resampled."""
    sr, data = wavfile.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
        peak = np.max(np.abs(data)) if data.size else 0.0
        if peak > 1.5:  # float files written with headroom conventions vary
            data = data / peak
    if target_sr is not None and sr != target_sr:
        g = np.gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return data, sr


def save_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    wavfile.write(str(path), sr, (audio * 32767.0).astype(np.int16))


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-10)))


def adaptive_gain(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def mix_at(base: np.ndarray, layer: np.ndarray, start: int) -> np.ndarray:
    """Add `layer` into a copy of `base` starting at sample `start`."""
    out = base.copy()
    end = min(len(out), start + len(layer))
    if start >= len(out) or end <= start:
        return out
    out[start:end] += layer[: end - start]
    return out


def envelope_segments(audio: np.ndarray, sr: int, threshold_db: float = -45.0) -> list[tuple[float, float]]:
    """Energy-gate helper used to check genuine silences inside events."""
    frame = max(1, sr // 100)  # 10 ms frames
    if len(audio) < frame:
        return []
    n = len(audio) // frame
    energy = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    active = energy > (10.0 ** (threshold_db / 20.0))
    segments: list[tuple[float, float]] = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segments.append((start * frame / sr, i * frame / sr))
            start = None
    if start is not None:
        segments.append((start * frame / sr, len(audio) / sr))
    return segments


def audio_config_summary(cfg: AudioConfig) -> dict:
    return {
        "sample_rate": cfg.sample_rate,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
        "n_mels": cfg.n_mels,
        "fmin": cfg.fmin,
        "fmax": cfg.fmax,
        "win_frames": cfg.win_frames,
        "hop_frames": cfg.hop_frames,
        "frame_seconds": cfg.frame_seconds,
        "window_seconds": cfg.window_seconds,
        "window_hop_seconds": cfg.window_hop_seconds,
    }
