"""Shared constants for the Astar Island challenge."""

from __future__ import annotations

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

TERRAIN_NAMES = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
    10: "Ocean",
    11: "Plains",
}

TERRAIN_TO_CLASS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    10: 0,
    11: 0,
}

STATIC_TERRAINS = {5, 10}
BUILDABLE_TERRAINS = {0, 11}
SETTLEMENT_TERRAINS = {1, 2, 3}


def terrain_to_class(terrain_code: int) -> int:
    """Map raw terrain codes to submission class indices."""
    return TERRAIN_TO_CLASS[int(terrain_code)]
