"""
ecgxai.config

Central configuration values used across the project.

Environment overrides:
    ECGXAI_MAXLEN
    ECGXAI_RANDOM_SEED
    ECGXAI_DATA_ROOT
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAXLEN: int = 5000
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_DATA_ROOT: Path = Path("/kaggle/input/")

MAXLEN: int = int(os.getenv("ECGXAI_MAXLEN", DEFAULT_MAXLEN))
RANDOM_SEED: int = int(os.getenv("ECGXAI_RANDOM_SEED", DEFAULT_RANDOM_SEED))
DATA_ROOT: Path = Path(os.getenv("ECGXAI_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()

if MAXLEN <= 0:
    raise ValueError(f"MAXLEN must be > 0, got {MAXLEN}")
# RANDOM_SEED can be any int; no need to validate more.

__all__ = ["MAXLEN", "RANDOM_SEED", "DATA_ROOT", "Settings", "get_settings"]

@dataclass(frozen=True)
class Settings:
    """Typed bundle of active configuration values (useful for passing to pipelines)."""
    maxlen: int = MAXLEN
    random_seed: int = RANDOM_SEED
    data_root: Path = DATA_ROOT


def get_settings() -> Settings:
    """Return a frozen Settings object representing the active configuration."""
    return Settings()
