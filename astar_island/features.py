"""Initial-state feature extraction for Astar Island cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .constants import terrain_to_class


def _cap(value: int, maximum: int) -> int:
    return min(int(value), maximum)


def _distance_bin(distance: int | None) -> int:
    if distance is None:
        return 5
    return min(distance, 5)


@dataclass(frozen=True)
class CellFeatures:
    terrain: int
    terrain_class: int
    coastal: int
    adj_settlement: int
    adj_port: int
    adj_ruin: int
    adj_forest: int
    adj_mountain: int
    ring2_settlement: int
    ring2_port: int
    ring2_ruin: int
    ring2_forest: int
    dist_settlement: int
    dist_port: int
    dist_ruin: int
    dist_ocean: int

    def terrain_key(self) -> tuple[int]:
        return (self.terrain,)

    def broad_key(self) -> tuple[int, ...]:
        return (
            self.terrain,
            self.coastal,
            self.dist_settlement,
            self.dist_port,
            self.dist_ocean,
        )

    def medium_key(self) -> tuple[int, ...]:
        return (
            self.terrain,
            self.coastal,
            self.adj_settlement,
            self.adj_port,
            self.ring2_settlement,
            self.ring2_ruin,
            self.dist_ocean,
        )

    def specific_key(self) -> tuple[int, ...]:
        return (
            self.terrain,
            self.coastal,
            self.adj_settlement,
            self.adj_port,
            self.adj_ruin,
            self.adj_forest,
            self.ring2_settlement,
            self.ring2_port,
            self.ring2_ruin,
            self.ring2_forest,
            self.dist_settlement,
            self.dist_port,
            self.dist_ruin,
            self.dist_ocean,
        )


class SeedFeatureMap:
    """Precomputed features for every cell in one seed."""

    def __init__(self, state: dict[str, object]) -> None:
        self.grid = [[int(cell) for cell in row] for row in state["grid"]]  # type: ignore[index]
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.grid else 0
        self.features = [[self._build_cell(x, y) for x in range(self.width)] for y in range(self.height)]

    def get(self, x: int, y: int) -> CellFeatures:
        return self.features[y][x]

    def cells(self) -> Iterable[tuple[int, int, CellFeatures]]:
        for y, row in enumerate(self.features):
            for x, feature in enumerate(row):
                yield x, y, feature

    def _build_cell(self, x: int, y: int) -> CellFeatures:
        terrain = int(self.grid[y][x])

        def count(radius: int, terrain_codes: set[int], include_self: bool = False) -> int:
            total = 0
            for ny in range(max(0, y - radius), min(self.height, y + radius + 1)):
                for nx in range(max(0, x - radius), min(self.width, x + radius + 1)):
                    if not include_self and nx == x and ny == y:
                        continue
                    chebyshev = max(abs(nx - x), abs(ny - y))
                    if chebyshev > radius:
                        continue
                    if self.grid[ny][nx] in terrain_codes:
                        total += 1
            return total

        def nearest_distance(terrain_codes: set[int]) -> int | None:
            best: int | None = None
            for ny in range(self.height):
                for nx in range(self.width):
                    if self.grid[ny][nx] not in terrain_codes:
                        continue
                    distance = abs(nx - x) + abs(ny - y)
                    if best is None or distance < best:
                        best = distance
            return best

        adj_settlement = _cap(count(1, {1}), 3)
        adj_port = _cap(count(1, {2}), 2)
        adj_ruin = _cap(count(1, {3}), 2)
        adj_forest = _cap(count(1, {4}), 4)
        adj_mountain = _cap(count(1, {5}), 3)
        ring2_settlement = _cap(count(2, {1, 2}) - adj_settlement - adj_port, 4)
        ring2_port = _cap(count(2, {2}) - adj_port, 3)
        ring2_ruin = _cap(count(2, {3}) - adj_ruin, 3)
        ring2_forest = _cap(count(2, {4}) - adj_forest, 5)
        coastal = 1 if count(1, {10}) > 0 else 0

        return CellFeatures(
            terrain=terrain,
            terrain_class=terrain_to_class(terrain),
            coastal=coastal,
            adj_settlement=adj_settlement,
            adj_port=adj_port,
            adj_ruin=adj_ruin,
            adj_forest=adj_forest,
            adj_mountain=adj_mountain,
            ring2_settlement=ring2_settlement,
            ring2_port=ring2_port,
            ring2_ruin=ring2_ruin,
            ring2_forest=ring2_forest,
            dist_settlement=_distance_bin(nearest_distance({1, 2})),
            dist_port=_distance_bin(nearest_distance({2})),
            dist_ruin=_distance_bin(nearest_distance({3})),
            dist_ocean=_distance_bin(nearest_distance({10})),
        )


def build_feature_maps(round_detail: dict[str, object]) -> list[SeedFeatureMap]:
    return [SeedFeatureMap(state) for state in round_detail["initial_states"]]  # type: ignore[index]
