"""Shared configuration for the Instamart capstone project."""

from pathlib import Path

# Project directories
PROJECT_ROOT=Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SQL_DIR = PROJECT_ROOT / "sql"
STATE_DIR = PROJECT_ROOT / "state"

# Database
DATABASE_PATH = DATA_DIR / "instamart.db"


def ensure_directories() -> None:
    """Create directories used for generated outputs."""
    for directory in (CLEAN_DIR, OUTPUT_DIR, STATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
