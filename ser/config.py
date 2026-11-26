"""Central, inspectable configuration for the sound-event recognition prototype.

Every value here is also serialized into each experiment output directory so a
past run can be reproduced and audited without guessing parameters.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

TARGET_CLASSES = ("knock", "bell", "machine")
BACKGROUND_CLASS = "background"
UNKNOWN_CLASS = "unknown"
ALL_EVENT_CLASSES = TARGET_CLASSES + (UNKNOWN_CLASS,)

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "data"
EXPERIMENTS_ROOT = WORKSPACE_ROOT / "experiments"
CUESHEET_PATH = DATA_ROOT / "cue_sheet.json"
FEEDBACK_PATH = DATA_ROOT / "feedback.jsonl"


@dataclasses.dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 320          # 20 ms per analysis frame
    n_mels: int = 40
    fmin: float = 50.0
    fmax: float = 7500.0
    win_frames: int = 50           # 1.0 s model input window
    hop_frames: int = 10           # 200 ms window hop
    min_label_overlap: float = 0.25  # min overlap (s) to tag a training window
    clip_seconds: float = 12.0
    frame_seconds: float = dataclasses.field(init=False)
    window_seconds: float = dataclasses.field(init=False)
    window_hop_seconds: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_seconds", self.hop_length / self.sample_rate)
        object.__setattr__(self, "window_seconds", self.win_frames * self.frame_seconds)
        object.__setattr__(
            self, "window_hop_seconds", self.hop_frames * self.frame_seconds
        )


@dataclasses.dataclass(frozen=True)
class EventConfig:
    merge_gap_seconds: float = 0.30   # gaps <= this inside one class get merged
    min_event_seconds: float = 0.15
    detection_min_prob: float = 0.5   # fallback; calibrated thresholds usually differ
    bell_peak_prominence: float = 0.15
    bell_peak_min_distance: float = 0.18


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    n_train: int = 24
    n_val: int = 8
    n_test: int = 10
    n_interference: int = 8
    n_background_train: int = 6
    n_background_val: int = 3
    seed: int = 20261126


@dataclasses.dataclass(frozen=True)
class Config:
    audio: AudioConfig = dataclasses.field(default_factory=AudioConfig)
    events: EventConfig = dataclasses.field(default_factory=EventConfig)
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


DEFAULT_CONFIG = Config()


def load_config(path: Path | None = None) -> Config:
    if path is None:
        return DEFAULT_CONFIG
    raw = json.loads(Path(path).read_text())

    def filtered(cls, values):
        names = {f.name for f in dataclasses.fields(cls) if f.init}
        return {k: v for k, v in values.items() if k in names}

    return Config(
        audio=AudioConfig(**filtered(AudioConfig, raw["audio"])),
        events=EventConfig(**filtered(EventConfig, raw["events"])),
        dataset=DatasetConfig(**filtered(DatasetConfig, raw["dataset"])),
    )
