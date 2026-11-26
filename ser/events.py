"""Sound-event kernel: window scoring -> frame evidence -> event intervals.

Design points required by the brief:
  * the same event spanning two analysis windows is merged into one interval;
  * a genuine acoustic pause breaks the event (frame-level assembly plus an
    explicit energy-gated silence check);
  * overlapping bells get a `possible_count`; when dry source tracks exist they
    are matched to identities, otherwise the count is explicitly uncertain;
  * nothing is ever emitted as background/unknown-by-default — windows below
    every calibrated threshold are rejected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .audio import envelope_segments
from .config import Config, TARGET_CLASSES
from .dataset import Recording
from .features import extract_windows, log_mel_spectrogram
from .model import SoundEventModel


@dataclass
class SoundEvent:
    event_uid: str
    cls: str
    start: float
    end: float
    confidence: float
    possible_count: int = 1
    source_event_id: str | None = None
    source_file: str | None = None
    source_match: str = "none"   # exact | count_only | none
    windows: list[int] = field(default_factory=list)
    note: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


def window_to_frame_prob(window_probs: np.ndarray, window_starts: np.ndarray,
                         n_frames: int, win_frames: int) -> np.ndarray:
    """Average overlapping window probabilities into per-frame evidence."""
    n_classes = window_probs.shape[1]
    acc = np.zeros((n_frames, n_classes), dtype=np.float64)
    weight = np.zeros((n_frames, 1), dtype=np.float64)
    for i, start in enumerate(window_starts):
        end = min(n_frames, start + win_frames)
        acc[start:end] += window_probs[i]
        weight[start:end] += 1.0
    weight[weight == 0] = 1.0
    return (acc / weight).astype(np.float32)


def _runs_from_mask(mask: np.ndarray, frame_seconds: float) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _merge_runs(runs: list[tuple[int, int]], max_gap_frames: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def assemble_events(frame_probs: np.ndarray, thresholds: np.ndarray,
                    cfg: Config, audio: np.ndarray,
                    window_starts: np.ndarray, window_probs: np.ndarray
                    ) -> list[SoundEvent]:
    ac, ec = cfg.audio, cfg.events
    events: list[SoundEvent] = []
    uid = 0
    max_gap_frames = max(1, int(ec.merge_gap_seconds / ac.frame_seconds))
    min_frames = max(1, int(ec.min_event_seconds / ac.frame_seconds))

    for c, cls in enumerate(TARGET_CLASSES):
        mask = frame_probs[:, c] >= thresholds[c]
        runs = _merge_runs(_runs_from_mask(mask, ac.frame_seconds), max_gap_frames)
        for f_start, f_end in runs:
            if f_end - f_start < min_frames:
                continue
            start = f_start * ac.frame_seconds
            end = min(len(audio) / ac.sample_rate, f_end * ac.frame_seconds)

            # Real stop check: if the mixture itself is near-silent for longer
            # than the allowed merge gap, split the interval at that pause.
            split_points = _silence_breaks(audio, ac.sample_rate, start, end,
                                           ec.merge_gap_seconds)
            spans = _apply_breaks(start, end, split_points)
            for seg_start, seg_end in spans:
                if seg_end - seg_start < ec.min_event_seconds:
                    continue
                win_idx = [
                    i for i, ws in enumerate(window_starts * ac.frame_seconds)
                    if ws < seg_end and ws + ac.window_seconds > seg_start
                ]
                if not win_idx:
                    continue
                window_scores = window_probs[win_idx, c]
                confidence = float(np.mean(window_scores))
                # Event-level gate: a real event needs at least one strongly
                # positive window and an average that clears the class
                # threshold. This rejects long, weak frame-level excursions
                # over background/unknown audio.
                if (confidence < thresholds[c]
                        or float(np.max(window_scores)) < thresholds[c]):
                    continue
                ev = SoundEvent(
                    event_uid=f"ev_{uid:04d}",
                    cls=cls,
                    start=round(seg_start, 3),
                    end=round(seg_end, 3),
                    confidence=round(confidence, 4),
                    windows=win_idx,
                )
                uid += 1
                events.append(ev)
    events.sort(key=lambda e: (e.start, e.end))
    return _renumber(events)


def _silence_breaks(audio: np.ndarray, sr: int, start: float, end: float,
                    min_pause: float, threshold_db: float = -48.0) -> list[float]:
    s0, s1 = int(start * sr), int(end * sr)
    segments = envelope_segments(audio[s0:s1], sr, threshold_db)
    if len(segments) < 2:
        return []
    breaks = []
    for (a0, a1), (b0, b1) in zip(segments[:-1], segments[1:]):
        if b0 - a1 >= min_pause:
            breaks.append(start + (a1 + b0) / 2.0)
    return breaks


def _apply_breaks(start: float, end: float,
                  breaks: list[float]) -> list[tuple[float, float]]:
    if not breaks:
        return [(start, end)]
    spans = []
    cursor = start
    for b in breaks:
        spans.append((cursor, b))
        cursor = b
    spans.append((cursor, end))
    return spans


def count_bell_rings(frame_probs: np.ndarray, cfg: Config,
                     event: SoundEvent) -> tuple[int, list[float]]:
    """Count repeated rings inside one bell event from probability peaks."""
    bell_idx = TARGET_CLASSES.index("bell")
    f0 = int(event.start / cfg.audio.frame_seconds)
    f1 = max(f0 + 1, int(event.end / cfg.audio.frame_seconds))
    curve = frame_probs[f0:f1, bell_idx]
    distance = max(1, int(cfg.events.bell_peak_min_distance / cfg.audio.frame_seconds))
    peaks, _ = find_peaks(curve,
                          prominence=cfg.events.bell_peak_prominence,
                          distance=distance)
    times = [event.start + p * cfg.audio.frame_seconds for p in peaks]
    return max(1, len(peaks)), times


def annotate_bell_counts(events: list[SoundEvent],
                         frame_probs: np.ndarray, cfg: Config) -> None:
    for ev in events:
        if ev.cls == "bell":
            count, peaks = count_bell_rings(frame_probs, cfg, ev)
            ev.possible_count = count
            if count > 1:
                ev.note = f"{count} ring peaks detected in mixture"


def match_sources(events: list[SoundEvent], recording: Recording,
                  iou_threshold: float = 0.3) -> None:
    """Attach dry source-track identities where available.

    Overlapping same-class events are matched as a group with the Hungarian
    algorithm. If the number of source tracks disagrees with the predicted
    count, events keep `count_only` (possible event count, no hard identity).
    """
    from scipy.optimize import linear_sum_assignment

    sources = [e for e in recording.labels["events"] if e["class"] in TARGET_CLASSES]
    for cls in TARGET_CLASSES:
        pred = [e for e in events if e.cls == cls]
        gt = [g for g in sources if g["class"] == cls]
        if not pred:
            continue
        if not gt:
            for e in pred:
                e.source_match = "none"
            continue
        cost = np.array([[1.0 - _iou(e, g) for g in gt] for e in pred])
        rows, cols = linear_sum_assignment(cost)
        matched_gts = set()
        for r, col in zip(rows, cols):
            e, g = pred[r], gt[col]
            if 1.0 - cost[r, col] >= iou_threshold:
                e.source_event_id = g["event_id"]
                e.source_file = g["source"]
                e.source_match = "exact"
                e.possible_count = max(e.possible_count, g.get("ring_count", 1))
                matched_gts.add(g["event_id"])
            else:
                e.source_match = "count_only"
                e.note = (e.note + "; source identity below IoU threshold").strip("; ")
        if len(gt) != len(pred):
            for e in pred:
                if e.source_match != "exact":
                    e.source_match = "count_only"
            for e in pred:
                if e.source_match == "exact" and (len(gt) != len(pred)):
                    # identities are ambiguous when counts disagree
                    e.note = (e.note + "; overlapping source count disagreement").strip("; ")


def _iou(ev: SoundEvent, gt: dict) -> float:
    inter = max(0.0, min(ev.end, gt["end"]) - max(ev.start, gt["start"]))
    union = max(1e-9, max(ev.end, gt["end"]) - min(ev.start, gt["start"]))
    return inter / union


def infer_recording(model: SoundEventModel, recording: Recording,
                    with_sources: bool = True) -> dict:
    """Full local inference chain for one recording."""
    cfg = model.cfg
    spec = log_mel_spectrogram(recording.audio, cfg.audio, model.filterbank)
    patches, starts = extract_windows(spec, cfg.audio)
    probs = model.score_windows(patches)
    n_frames = spec.shape[1]
    frame_probs = window_to_frame_prob(probs, starts, n_frames, cfg.audio.win_frames)
    events = assemble_events(frame_probs, model.thresholds, cfg,
                             recording.audio, starts, probs)
    annotate_bell_counts(events, frame_probs, cfg)
    if with_sources:
        match_sources(events, recording)
    return {
        "recording_id": recording.recording_id,
        "condition": recording.labels.get("condition"),
        "window_starts": starts,
        "window_times": (starts * cfg.audio.frame_seconds).round(3),
        "window_probs": probs,
        "thresholds": model.thresholds,
        "frame_probs": frame_probs,
        "spectrogram": spec,
        "events": events,
    }


def merge_events(events: list[SoundEvent], first_uid: str,
                 second_uid: str) -> SoundEvent | None:
    """Cue-sheet merge: combine two adjacent events (class may differ only
    when the operator explicitly overrides; here same class is enforced)."""
    a = next((e for e in events if e.event_uid == first_uid), None)
    b = next((e for e in events if e.event_uid == second_uid), None)
    if a is None or b is None or a.cls != b.cls:
        return None
    merged = SoundEvent(
        event_uid=a.event_uid,
        cls=a.cls,
        start=min(a.start, b.start),
        end=max(a.end, b.end),
        confidence=round(max(a.confidence, b.confidence), 4),
        possible_count=a.possible_count + b.possible_count,
        windows=sorted(set(a.windows + b.windows)),
        note="manual merge",
    )
    return merged


def split_event(events: list[SoundEvent], uid: str,
                at: float, gap: float = 0.12) -> tuple[SoundEvent, SoundEvent] | None:
    """Cue-sheet split: cut one event into two around time `at`."""
    ev = next((e for e in events if e.event_uid == uid), None)
    if ev is None or not (ev.start + 0.03 < at < ev.end - 0.03):
        return None
    left = SoundEvent(
        event_uid=ev.event_uid,
        cls=ev.cls, start=ev.start, end=round(at - gap / 2, 3),
        confidence=ev.confidence, possible_count=ev.possible_count,
        windows=list(ev.windows), note="manual split",
    )
    right = SoundEvent(
        event_uid=ev.event_uid + "_b",
        cls=ev.cls, start=round(at + gap / 2, 3), end=ev.end,
        confidence=ev.confidence, possible_count=1,
        windows=list(ev.windows), note="manual split",
    )
    return left, right


def _renumber(events: list[SoundEvent]) -> list[SoundEvent]:
    for i, ev in enumerate(events):
        ev.event_uid = f"ev_{i:04d}"
    return events


def events_to_dicts(events: list[SoundEvent]) -> list[dict]:
    return [e.to_dict() for e in events]


def dicts_to_events(rows: list[dict]) -> list[SoundEvent]:
    out = []
    for row in rows:
        allowed = {k: v for k, v in row.items()
                   if k in SoundEvent.__dataclass_fields__}
        out.append(SoundEvent(**allowed))
    return out
