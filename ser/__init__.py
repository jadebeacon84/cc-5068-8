"""Local sound event recognition prototype package."""
from .config import (
    ALL_EVENT_CLASSES,
    AUDIT_AUDITED,
    AUDIT_PENDING,
    AUDIT_STATUSES,
    AUDIT_UNCLAIMED,
    BACKGROUND_CLASS,
    DEFAULT_CONFIG,
    LEGACY_TARGET_CLASSES,
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
    "LEGACY_TARGET_CLASSES",
    "AUDIT_STATUSES",
    "AUDIT_AUDITED",
    "AUDIT_PENDING",
    "AUDIT_UNCLAIMED",
]
