"""Resolve dated raw-source snapshots while retaining legacy direct paths."""

import re
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
