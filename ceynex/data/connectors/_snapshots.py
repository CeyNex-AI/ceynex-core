"""Resolve dated raw-source snapshots while retaining legacy direct paths."""

import re
from datetime import UTC, date, datetime
from pathlib import Path

_SNAPSHOT_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def latest_snapshot_dir(raw_path: Path) -> Path:
    """Return the latest dated child directory, or a legacy raw directory itself."""
    raw_path = Path(raw_path)
    if not raw_path.is_dir():
        raise FileNotFoundError(raw_path)
    snapshots = sorted(
        (entry for entry in raw_path.iterdir() if entry.is_dir() and _SNAPSHOT_NAME.fullmatch(entry.name)),
        key=lambda entry: entry.name,
    )
    return snapshots[-1] if snapshots else raw_path


def resolve_snapshot_file(raw_path: Path, filename: str) -> Path:
    """Resolve either an explicit file or ``filename`` in the latest snapshot."""
    raw_path = Path(raw_path)
    if raw_path.is_file():
        return raw_path
    return latest_snapshot_dir(raw_path) / filename


def recent_enough(folder_name: str, max_age_days: int | None, today: date | None = None) -> bool:
    """Whether a cached pull in dated folder `folder_name` may be reused.

    `max_age_days` None means always (the default, and what an admin's Ingest
    button and `--offline` want). The monthly refresh sets it, so a pull older
    than that is fetched again rather than replayed: Comtrade revises past years
    and new ones appear, and a cache that is reused forever hides both.
    """
    if max_age_days is None:
        return True
    if not _SNAPSHOT_NAME.fullmatch(folder_name):
        return False
    pulled = date.fromisoformat(folder_name)
    return ((today or datetime.now(UTC).date()) - pulled).days < max_age_days
