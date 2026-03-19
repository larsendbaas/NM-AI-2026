"""Filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_file(path: Path) -> Any:
    """Load JSON written in either UTF-8 or UTF-16."""
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"Could not decode JSON file: {path}")
