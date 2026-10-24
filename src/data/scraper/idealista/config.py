"""Runtime settings for the scraper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Settings:
    # Pacing between page loads, in seconds. Requests are spaced randomly within
    # this window so the traffic does not follow a fixed machine-like rhythm.
    min_delay: float = 6.0
    max_delay: float = 12.0

    # Stop the run after this many consecutive challenge/blocked responses.
    # Continuing after blocks only escalates the block, so the run aborts.
    max_consecutive_blocks: int = 3

    # Cached HTML older than this is refetched.
    cache_ttl_hours: int = 24 * 7

    # Headed by default: a visible window is the ordinary way a person browses.
    headless: bool = False
    timeout_ms: int = 45_000

    db_path: Path = DATA_DIR / "idealista.sqlite3"
    cache_dir: Path = DATA_DIR / "html_cache"
    # Persistent browser profile, so cookies survive between runs the way they
    # would for a returning visitor.
    profile_dir: Path = DATA_DIR / "browser_profile"
