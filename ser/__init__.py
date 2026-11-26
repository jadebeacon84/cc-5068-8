"""Local sound event recognition prototype package."""
from .config import (
    ALL_EVENT_CLASSES,
    BACKGROUND_CLASS,
    DEFAULT_CONFIG,
    TARGET_CLASSES,
    UNKNOWN_CLASS,
    Config,
)

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "TARGET_CLASSES",
    "BACKGROUND_CLASS",
    "UNKNOWN_CLASS",
    "ALL_EVENT_CLASSES",
]
