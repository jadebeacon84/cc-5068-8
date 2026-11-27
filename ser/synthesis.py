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
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

from .audio import save_wav
from .config import (
    AUDIT_AUDITED,
    AUDIT_PENDING,
    DatasetConfig,
    SCOPE_SEGMENT,
    SCOPE_SINGLE,
)
from .labeling import build_legacy_seed, recording_digests, sha256_file

SR = 16000
CLIP_SECONDS = 12.0

CONDITIONS = ("clean_room", "room_reverb", "interference")

# Fixed offsets keep split/condition assignment and RNG seeds deterministic
# (no reliance on PYTHONHASHSEED).
_SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2, "interference": 3}


@dataclass
class ScheduledEvent:
    event_id: str
    cls: str
    start: float
    end: float
    source_file: str
    audio: np.ndarray
    ring_count: int = 1          # >1: repeated rings/beeps of the same source
    overlaps: list[str] = field(default_factory=list)
    audit_status: str = AUDIT_AUDITED
    identity_uncertain: bool = False
    audit_note: str = ""


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
    """A train of 3-6 impulsive knocks; returns (audio, pulse_count)."""
    n_pulses = int(rng.integers(3, 7))
    gap = rng.uniform(0.09, 0.18)
    center = rng.uniform(700, 2400)
    pieces: list[np.ndarray] = []
    for i in range(n_pulses):
        burst = _bandpass_noise(rng, 0.07, center * 0.6, center * 1.6)
        burst *= _env_exp(len(burst), rng.uniform(0.012, 0.03), SR)
        pieces.append(burst)
        if i < n_pulses - 1:
            pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
    audio = np.concatenate(pieces)
    return audio / (np.max(np.abs(audio)) + 1e-9) * 0.85, n_pulses


def synth_bell(rng: np.random.Generator, rings: int = 1,
               ring_gap: float | None = None) -> np.ndarray:
    """Inharmonic partials with exponential decay; `rings` repeats of one bell."""
    f0 = rng.uniform(300, 900)
    partials = np.array([1.0, 2.0, 2.76, 3.92, 5.43, 6.72, 8.21])
    decays = rng.uniform(0.35, 1.4, size=len(partials))
    gains = np.array([1.0, 0.6, 0.42, 0.28, 0.18, 0.11, 0.06],
                     dtype=np.float32)
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


def synth_chime(rng: np.random.Generator, beeps: int = 1,
                beep_gap: float | None = None) -> tuple[np.ndarray, int]:
    """Electronic prompt tone: stable harmonic beeps with a hard envelope.

    Distinct from the inharmonic, exponentially decaying mechanical bell:
    a strongly dominant fundamental, near-integer harmonics, near-instant
    attack, flat sustain and an abrupt stop with regular on/off timing.
    Returns (audio, beep_count).
    """
    f0 = rng.uniform(1200, 2600)
    partials = np.array([1.0, 2.0, 3.0])
    gains = np.array([1.0, 0.15, 0.04], dtype=np.float32)
    pieces: list[np.ndarray] = []
    gap = beep_gap if beep_gap is not None else rng.uniform(0.1, 0.2)
    for _ in range(beeps):
        dur = rng.uniform(0.22, 0.42)
        n = int(dur * SR)
        t = np.arange(n) / SR
        sig = np.zeros(n, dtype=np.float32)
        for p, g in zip(partials, gains):
            # No per-partial random phase: stable phase gives the regular
            # electronic timbre the classifier can separate from bells.
            sig += (g * np.sin(2 * np.pi * f0 * p * t)).astype(np.float32)
        # hard electronic envelope: 3 ms attack, flat sustain, 10 ms release
        attack = int(0.003 * SR)
        release = int(0.01 * SR)
        env = np.ones(n, dtype=np.float32)
        if attack:
            env[:attack] = np.linspace(0, 1, attack)
        if release:
            env[-release:] = np.linspace(1, 0, release)
        pieces.append(sig * env)
        if len(pieces) < 2 * beeps - 1:
            pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
    audio = np.concatenate(pieces)
    return audio / (np.max(np.abs(audio)) + 1e-9) * 0.85, beeps


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


def _place_legacy_overlap(rng: np.random.Generator,
                          events: list[ScheduledEvent],
                          pending: bool) -> tuple[ScheduledEvent, ScheduledEvent,
                                                  list[dict]]:
    """Force a time-overlapping bell/chime pair sharing the legacy ``bell`` label.

    Returns the two scheduled events plus the legacy annotation spans: one
    single-event span (the bell) and one whole-segment span covering both.
    When ``pending`` is true the chime is left for listening review and the
    whole-segment legacy span is explicitly unresolved.
    """
    def free_span(start: float, end: float, margin: float = 0.25) -> bool:
        return all(end + margin <= e.start or start >= e.end + margin
                   for e in events)

    bell_audio = synth_bell(rng, rings=1)
    bell_dur = len(bell_audio) / SR
    # Only accept a strictly free slot; never fall back onto an occupied span.
    start = None
    for _ in range(120):
        candidate = float(rng.uniform(0.3, CLIP_SECONDS - bell_dur - 0.3))
        if free_span(candidate, candidate + bell_dur):
            start = candidate
            break
    if start is None:
        raise RuntimeError("no free slot for the legacy bell pair")
    bell_id = uuid.uuid4().hex[:12]
    bell = ScheduledEvent(
        event_id=bell_id, cls="bell", start=round(start, 3),
        end=round(start + bell_dur, 3),
        source_file=f"sources/{bell_id}.wav",
        audio=bell_audio.astype(np.float32), ring_count=1,
    )
    # The chime starts inside the bell's ringing tail: genuinely overlapping.
    chime_audio, n_beeps = synth_chime(rng, beeps=2)
    chime_dur = len(chime_audio) / SR

    def blocked_by_other(s: float) -> bool:
        e = s + chime_dur
        return any(
            other.event_id != bell.event_id
            and min(e, other.end) - max(s, other.start) > 0
            for other in events)

    chime_start = None
    for _ in range(120):
        candidate = max(0.2, min(
            bell.end - rng.uniform(0.2, min(0.6, chime_dur)),
            CLIP_SECONDS - chime_dur - 0.2,
        ))
        if not blocked_by_other(candidate):
            chime_start = candidate
            break
    if chime_start is None:
        raise RuntimeError("no free slot for the legacy chime")
    chime_id = uuid.uuid4().hex[:12]
    chime = ScheduledEvent(
        event_id=chime_id, cls="chime", start=round(chime_start, 3),
        end=round(chime_start + chime_dur, 3),
        source_file=f"sources/{chime_id}.wav",
        audio=chime_audio.astype(np.float32), ring_count=n_beeps,
        audit_status=AUDIT_PENDING if pending else AUDIT_AUDITED,
        identity_uncertain=pending,
        audit_note=("legacy bell split; awaiting listening review"
                    if pending else ""),
    )
    bell.overlaps.append(chime_id)
    chime.overlaps.append(bell_id)
    legacy = [
        build_legacy_seed(
            "bell", bell.start, bell.end, SCOPE_SINGLE,
            f"leg_single_{bell_id[:8]}",
            note="legacy single-event marker for the bicycle bell",
        ),
        build_legacy_seed(
            "bell", min(bell.start, chime.start),
            max(bell.end, chime.end), SCOPE_SEGMENT,
            f"leg_segment_{bell_id[:8]}",
            note=("whole-segment marker covering bell + chime; "
                  + ("unresolved split" if pending else "split confirmed by review")),
        ),
    ]
    return bell, chime, legacy


def _schedule_events(rng: np.random.Generator, condition: str,
                     background_only: bool = False,
                     force_legacy_overlap: bool = False,
                     legacy_pending: bool = False
                     ) -> tuple[list[ScheduledEvent], list[dict]]:
    if background_only:
        return [], []
    n_target = {"clean_room": 4, "room_reverb": 4, "interference": 4}[condition]
    n_unknown = {"clean_room": 1, "room_reverb": 2, "interference": 2}[condition]
    events: list[ScheduledEvent] = []
    legacy_entries: list[dict] = []
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

    # The forced legacy pair is inserted last (after targets and unknowns), so
    # no unrelated interval can ever land inside its legacy marker spans.
    if force_legacy_overlap:
        n_target -= 2

    # Rotation guarantees every current class (including the new chime) is
    # represented across the first scheduled target slots. After one full
    # rotation, prompt tones are favored so the newly split class has enough
    # audited positive windows despite pending items being withheld.
    target_classes = ["knock", "bell", "chime", "machine"]
    for i in range(n_target):
        if i < len(target_classes):
            cls = target_classes[i % len(target_classes)]
        else:
            cls = str(rng.choice(
                ["chime", "chime", "chime", "bell", "knock", "machine"]))
        eid = uuid.uuid4().hex[:12]
        rings = 1
        if cls == "knock":
            audio, _ = synth_knock(rng)
        elif cls == "machine":
            audio = synth_machine(rng)
        elif cls == "chime":
            beeps = 2 if rng.random() < 0.5 else 1
            audio, rings = synth_chime(rng, beeps=beeps)
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

    if force_legacy_overlap:
        bell, chime, legacy = _place_legacy_overlap(rng, events, legacy_pending)
        events.extend([bell, chime])
        legacy_entries.extend(legacy)

    events.sort(key=lambda e: e.start)
    return events, legacy_entries


def _render_event(ev: ScheduledEvent) -> np.ndarray:
    return ev.audio.astype(np.float32)


def synthesize_recording(out_dir: Path, condition: str, index: int,
                         seed: int, background_only: bool = False,
                         force_legacy_overlap: bool = False,
                         legacy_pending: bool = False) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "sources").mkdir(parents=True, exist_ok=True)
    schedule, legacy_entries = [], []
    for attempt in range(8):
        attempt_rng = np.random.default_rng(seed + index * 7919 + attempt * 101)
        try:
            schedule, legacy_entries = _schedule_events(
                attempt_rng, condition, background_only,
                force_legacy_overlap=force_legacy_overlap,
                legacy_pending=legacy_pending,
            )
            break
        except RuntimeError:
            if attempt == 7:
                raise
    rng = np.random.default_rng(seed + index * 7919)

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
            "audit_status": ev.audit_status,
            "identity_uncertain": ev.identity_uncertain,
            "audit_note": ev.audit_note,
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
        "legacy_annotations": legacy_entries,
        "file_digests": recording_digests(out_dir),
    }
    (out_dir / "labels.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def build_dataset(root: Path, cfg: DatasetConfig) -> dict:
    """Generate train/val/test/interference splits as disjoint recordings."""
    root = Path(root)
    # Remove recordings left by earlier generations so a rebuild cannot leave
    # stale directories that are absent from the new manifest. Runtime files at
    # the root (cue sheet, feedback, config snapshot) are left untouched.
    for split in ("train", "val", "test", "interference"):
        split_dir = root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
    plan = {
        "train": cfg.n_train,
        "val": cfg.n_val,
        "test": cfg.n_test,
        "interference": cfg.n_interference,
    }
    manifest: dict[str, list[str]] = {}
    for split, count in plan.items():
        ids = []
        split_seed = cfg.seed + _SPLIT_OFFSETS[split] * 100003
        for i in range(count):
            # Recording conditions are distributed across splits, but each
            # recording exists in exactly one split (no slice-level leakage).
            condition = CONDITIONS[(i + _SPLIT_OFFSETS[split]) % len(CONDITIONS)]
            rec_id = f"{split[:2]}_{uuid.uuid4().hex[:10]}"
            # First two indexed slots carry the forced overlapping bell/chime
            # pair inherited from the shared legacy label. The val copy keeps
            # its prompt tone pending listening review; the train copy is
            # already audited.
            force_overlap = i in cfg.legacy_overlap_indices
            pending = force_overlap and split == "val"
            synthesize_recording(
                root / split / rec_id, condition, i, split_seed,
                force_legacy_overlap=force_overlap,
                legacy_pending=pending,
            )
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
                seed=split_seed,
                background_only=True,
            )
            ids.append(rec_id)
        manifest[split] = ids
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
