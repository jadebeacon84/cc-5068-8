"""Synthesize annotated local recordings with separate source tracks.

The prototype is trained on self-made, fully annotated recordings generated on
this machine. Each recording writes:
  * mix.wav        - the mono mixture the model actually analyzes
  * sources/*.wav  - one dry source track per event id (no class names in file
                     names, so nothing can infer labels from folder/file names)
  * labels.json    - event spans, source mapping and synthesis parameters

Three independent recording conditions are produced: clean room, reverberant
room, and dense interference mixes containing unknown (rejectable) sounds.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

from .audio import save_wav
from .config import DatasetConfig

SR = 16000
CLIP_SECONDS = 12.0

CONDITIONS = ("clean_room", "room_reverb", "interference")


@dataclass
class ScheduledEvent:
    event_id: str
    cls: str
    start: float
    end: float
    source_file: str
    audio: np.ndarray
    ring_count: int = 1          # >1: repeated rings of the same bell
    overlaps: list[str] = field(default_factory=list)


def _bandpass_noise(rng: np.random.Generator, duration: float,
                    low: float, high: float, sr: int = SR) -> np.ndarray:
    n = int(duration * sr)
    noise = rng.standard_normal(n).astype(np.float32)
    nyq = sr / 2
    b, a = butter(4, [max(low / nyq, 1e-4), min(high / nyq, 0.999)], btype="band")
    return lfilter(b, a, noise).astype(np.float32)


def _pinkish_noise(rng: np.random.Generator, duration: float, sr: int = SR) -> np.ndarray:
    n = int(duration * sr)
    white = rng.standard_normal(n).astype(np.float32)
    # crude 1/f shaping via one-pole low pass differences
    b, a = butter(2, 0.45)
    shaped = lfilter(b, a, white)
    return shaped.astype(np.float32)


def _env_exp(n: int, decay_s: float, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    return np.exp(-t / max(decay_s, 1e-3)).astype(np.float32)


def _env_ar(n: int, attack_s: float, decay_s: float, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    attack = np.minimum(t / max(attack_s, 1e-3), 1.0)
    release = np.exp(-np.maximum(t - attack_s, 0.0) / max(decay_s, 1e-3))
    return (attack * release).astype(np.float32)


def synth_knock(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """A train of 2-5 impulsive knocks; returns (audio, ring_count)."""
    n_pulses = int(rng.integers(2, 6))
    gap = rng.uniform(0.07, 0.16)
    center = rng.uniform(700, 2400)
    pieces: list[np.ndarray] = []
    for i in range(n_pulses):
        burst = _bandpass_noise(rng, 0.05, center * 0.6, center * 1.6)
        burst *= _env_exp(len(burst), rng.uniform(0.008, 0.02), SR)
        pieces.append(burst)
        if i < n_pulses - 1:
            pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
    audio = np.concatenate(pieces)
    return audio / (np.max(np.abs(audio)) + 1e-9) * 0.8, n_pulses


def synth_bell(rng: np.random.Generator, rings: int = 1,
               ring_gap: float | None = None) -> np.ndarray:
    """Inharmonic partials with exponential decay; `rings` repeats of one bell."""
    f0 = rng.uniform(350, 1400)
    partials = np.array([1.0, 2.0, 2.76, 3.92, 5.43, 6.72])
    decays = rng.uniform(0.25, 1.1, size=len(partials))
    gains = np.array([1.0, 0.55, 0.35, 0.22, 0.14, 0.09], dtype=np.float32)
    pieces: list[np.ndarray] = []
    gap = ring_gap if ring_gap is not None else rng.uniform(0.0, 0.25)
    for r in range(rings):
        dur = rng.uniform(0.9, 2.2)
        n = int(dur * SR)
        t = np.arange(n) / SR
        sig = np.zeros(n, dtype=np.float32)
        for p, d, g in zip(partials, decays, gains):
            sig += (g * np.sin(2 * np.pi * f0 * p * t + rng.uniform(0, 6.28))
                    * np.exp(-t / d)).astype(np.float32)
        sig *= _env_ar(n, 0.004, dur * 0.9, SR)
        pieces.append(sig)
        if r < rings - 1:
            pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
    audio = np.concatenate(pieces)
    return audio / (np.max(np.abs(audio)) + 1e-9) * 0.85


def synth_machine(rng: np.random.Generator) -> np.ndarray:
    """Steady mechanical hum: harmonics + amplitude modulation + air noise."""
    dur = rng.uniform(1.6, 4.5)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(70, 180)
    sig = np.zeros(n, dtype=np.float32)
    for h in range(1, rng.integers(6, 13)):
        sig += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28))
    am_freq = rng.uniform(6, 22)
    sig *= (0.75 + 0.25 * np.sin(2 * np.pi * am_freq * t)).astype(np.float32)
    air = _pinkish_noise(rng, dur) * rng.uniform(0.05, 0.12)
    sig = sig * 0.5 + air
    return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.7).astype(np.float32)


def synth_unknown(rng: np.random.Generator) -> np.ndarray:
    """Distractor sounds outside the target vocabulary (must be rejected)."""
    kind = rng.integers(0, 4)
    dur = rng.uniform(0.6, 2.0)
    n = int(dur * SR)
    t = np.arange(n) / SR
    if kind == 0:  # frequency sweep
        f0, f1 = rng.uniform(200, 900), rng.uniform(900, 3500)
        phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2)
        sig = np.sin(phase) * _env_ar(n, 0.05, dur * 0.6, SR)
    elif kind == 1:  # bird-like chirp sequence
        sig = np.zeros(n, dtype=np.float32)
        cursor = 0
        while cursor < n - SR // 20:
            c_len = int(rng.uniform(0.05, 0.12) * SR)
            ct = np.arange(c_len) / SR
            f = rng.uniform(1800, 4200)
            chirp = np.sin(2 * np.pi * f * ct + 40 * ct ** 2)
            sig[cursor:cursor + c_len] += chirp[: max(0, min(c_len, n - cursor))]
            cursor += c_len + int(rng.uniform(0.05, 0.2) * SR)
        sig *= _env_ar(n, 0.01, dur, SR)
    elif kind == 2:  # pink-ish noise burst, e.g. wind/cloth
        sig = _pinkish_noise(rng, dur) * _env_ar(n, 0.1, dur * 0.8, SR)
    else:  # formant-ish voiced blob, vaguely speech-like but not speech
        f0 = rng.uniform(110, 180)
        glottis = np.sign(np.sin(2 * np.pi * f0 * t))
        b, a = butter(3, rng.uniform(500, 1400) / (SR / 2))
        sig = lfilter(b, a, glottis) * _env_ar(n, 0.08, dur * 0.7, SR)
    return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.7).astype(np.float32)


def synthetic_rir(rng: np.random.Generator, rt60: float, sr: int = SR) -> np.ndarray:
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    rir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60)
    rir[0] += 1.0  # direct path
    return rir / np.max(np.abs(rir))


def _schedule_events(rng: np.random.Generator, condition: str,
                     background_only: bool = False) -> list[ScheduledEvent]:
    if background_only:
        return []
    n_target = {"clean_room": 3, "room_reverb": 4, "interference": 5}[condition]
    n_unknown = {"clean_room": 1, "room_reverb": 2, "interference": 3}[condition]
    events: list[ScheduledEvent] = []
    cursor = 0.5

    def free_span(start: float, end: float, margin: float = 0.25) -> bool:
        return all(end + margin <= e.start or start >= e.end + margin for e in events)

    def place(duration: float) -> float:
        nonlocal cursor
        for _ in range(60):
            start = float(rng.uniform(0.3, CLIP_SECONDS - duration - 0.3))
            if free_span(start, start + duration):
                return start
        start = cursor
        cursor = min(CLIP_SECONDS - duration - 0.2, cursor + duration + 0.4)
        return start

    target_classes = ["knock", "bell", "machine"]
    for i in range(n_target):
        cls = target_classes[i % 3] if i < 3 else str(rng.choice(target_classes))
        eid = uuid.uuid4().hex[:12]
        rings = 1
        if cls == "knock":
            audio, _ = synth_knock(rng)
        elif cls == "machine":
            audio = synth_machine(rng)
        else:
            # 40% of bells ring twice; same bell (ring_count=2) unless split below
            rings = 2 if rng.random() < 0.4 else 1
            audio = synth_bell(rng, rings=rings)
        duration = len(audio) / SR
        start = place(duration)
        events.append(ScheduledEvent(
            event_id=eid, cls=cls, start=round(start, 3),
            end=round(start + duration, 3),
            source_file=f"sources/{eid}.wav", audio=audio.astype(np.float32),
            ring_count=rings,
        ))

    # Half of interference/room recordings get a second distinct bell that
    # overlaps an existing bell tail — the ambiguous overlapping-bell case.
    if condition in ("room_reverb", "interference") and rng.random() < 0.6:
        bells = [e for e in events if e.cls == "bell"]
        if bells:
            anchor = bells[int(rng.integers(0, len(bells)))]
            second = synth_bell(rng, rings=1)
            duration = len(second) / SR
            start = anchor.end - rng.uniform(0.25, min(0.8, duration))
            start = max(0.2, min(start, CLIP_SECONDS - duration - 0.2))
            eid = uuid.uuid4().hex[:12]
            new = ScheduledEvent(
                event_id=eid, cls="bell", start=round(start, 3),
                end=round(start + duration, 3),
                source_file=f"sources/{eid}.wav",
                audio=second.astype(np.float32), ring_count=1,
            )
            anchor.overlaps.append(eid)
            new.overlaps.append(anchor.event_id)
            events.append(new)

    for _ in range(n_unknown):
        audio = synth_unknown(rng)
        duration = len(audio) / SR
        start = place(duration)
        eid = uuid.uuid4().hex[:12]
        events.append(ScheduledEvent(
            event_id=eid, cls="unknown", start=round(start, 3),
            end=round(start + duration, 3),
            source_file=f"sources/{eid}.wav", audio=audio.astype(np.float32),
        ))
    events.sort(key=lambda e: e.start)
    return events


def _render_event(ev: ScheduledEvent) -> np.ndarray:
    return ev.audio.astype(np.float32)


def synthesize_recording(out_dir: Path, condition: str, index: int,
                         seed: int, background_only: bool = False) -> dict:
    rng = np.random.default_rng(seed + index * 7919)
    out_dir = Path(out_dir)
    (out_dir / "sources").mkdir(parents=True, exist_ok=True)
    schedule = _schedule_events(rng, condition, background_only)

    noise_level = {"clean_room": 0.015, "room_reverb": 0.04, "interference": 0.09}[condition]
    rt60 = {"clean_room": 0.08, "room_reverb": 0.45, "interference": 0.35}[condition]
    reverb_mix = {"clean_room": 0.08, "room_reverb": 0.45, "interference": 0.3}[condition]
    rir = synthetic_rir(rng, rt60)

    total = int(CLIP_SECONDS * SR)
    mix = np.zeros(total, dtype=np.float32)
    labels_events = []
    for ev in schedule:
        dry = _render_event(ev)
        save_wav(out_dir / ev.source_file, dry, SR)
        wet = np.convolve(dry, rir)[:len(dry)].astype(np.float32)
        rendered = (1 - reverb_mix) * dry + reverb_mix * wet
        gain = rng.uniform(0.55, 0.9)
        rendered *= gain
        start_sample = int(ev.start * SR)
        end_sample = min(total, start_sample + len(rendered))
        mix[start_sample:end_sample] += rendered[:end_sample - start_sample]
        labels_events.append({
            "event_id": ev.event_id,
            "class": ev.cls,
            "start": ev.start,
            "end": ev.end,
            "source": ev.source_file,
            "ring_count": ev.ring_count,
            "overlaps": ev.overlaps,
        })

    noise = _pinkish_noise(rng, CLIP_SECONDS) * noise_level
    if condition == "interference":
        # steady interfering texture (fan-like), itself not a target class
        fan = synth_machine(np.random.default_rng(seed + 99)) * 0.12
        noise[:len(fan)] += fan[:len(noise)]
    mix += noise[:total]
    peak = np.max(np.abs(mix))
    if peak > 0.98:
        mix *= 0.98 / peak
    save_wav(out_dir / "mix.wav", mix, SR)

    meta = {
        "recording_id": uuid.uuid4().hex[:16],
        "condition": condition,
        "sample_rate": SR,
        "duration": CLIP_SECONDS,
        "seed": seed + index * 7919,
        "background_only": background_only,
        "synthesis": {
            "rt60": rt60, "reverb_mix": reverb_mix,
            "noise_level": noise_level,
        },
        "events": labels_events,
    }
    (out_dir / "labels.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def build_dataset(root: Path, cfg: DatasetConfig) -> dict:
    """Generate train/val/test/interference splits as disjoint recordings."""
    root = Path(root)
    plan = {
        "train": cfg.n_train,
        "val": cfg.n_val,
        "test": cfg.n_test,
        "interference": cfg.n_interference,
    }
    manifest: dict[str, list[str]] = {}
    for split, count in plan.items():
        ids = []
        for i in range(count):
            # Recording conditions are distributed across splits, but each
            # recording exists in exactly one split (no slice-level leakage).
            condition = CONDITIONS[(i + hash(split)) % len(CONDITIONS)]
            rec_id = f"{split[:2]}_{uuid.uuid4().hex[:10]}"
            synthesize_recording(root / split / rec_id, condition, i,
                                 cfg.seed + hash(split) % 100000)
            ids.append(rec_id)

        # Pure-background recordings teach rejection. They live only in
        # train/val (test keeps its natural background/unknown intervals).
        n_bg = {"train": cfg.n_background_train,
                "val": cfg.n_background_val}.get(split, 0)
        for i in range(n_bg):
            rec_id = f"{split[:2]}bg_{uuid.uuid4().hex[:10]}"
            condition = CONDITIONS[(i + 1) % len(CONDITIONS)]
            synthesize_recording(
                root / split / rec_id, condition,
                index=1000 + i,
                seed=cfg.seed + hash(split) % 100000,
                background_only=True,
            )
            ids.append(rec_id)
        manifest[split] = ids
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
