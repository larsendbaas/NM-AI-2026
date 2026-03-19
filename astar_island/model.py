"""Probabilistic transition model for Astar Island."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .constants import BUILDABLE_TERRAINS, STATIC_TERRAINS, terrain_to_class
from .features import CellFeatures, SeedFeatureMap


def normalize(weights: list[float], floor: float = 0.01) -> list[float]:
    clipped = [max(floor, float(value)) for value in weights]
    total = sum(clipped)
    return [value / total for value in clipped]


@dataclass
class ObservationSummary:
    total_queries: int
    total_cells: int


class TransitionModel:
    """Learns smoothed transition distributions from sampled viewports."""

    def __init__(self, floor: float = 0.01) -> None:
        self.floor = floor
        self.direct_counts: dict[tuple[int, int, int], list[float]] = defaultdict(lambda: [0.0] * 6)
        self.terrain_counts: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0] * 6)
        self.broad_counts: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0] * 6)
        self.medium_counts: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0] * 6)
        self.specific_counts: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0] * 6)
        self.summary = ObservationSummary(total_queries=0, total_cells=0)

    def fit(
        self,
        feature_maps: list[SeedFeatureMap],
        observations: Iterable[dict[str, object]],
    ) -> ObservationSummary:
        total_queries = 0
        total_cells = 0
        for observation in observations:
            total_queries += 1
            seed_index = int(observation["seed_index"])
            viewport = observation["viewport"]
            base_x = int(viewport["x"])
            base_y = int(viewport["y"])
            grid = observation["grid"]
            feature_map = feature_maps[seed_index]

            for dy, row in enumerate(grid):
                for dx, terrain_code in enumerate(row):
                    total_cells += 1
                    x = base_x + dx
                    y = base_y + dy
                    class_index = terrain_to_class(int(terrain_code))
                    features = feature_map.get(x, y)
                    self._add(self.direct_counts[(seed_index, x, y)], class_index, 1.0)
                    self._add(self.terrain_counts[features.terrain_key()], class_index, 1.0)
                    self._add(self.broad_counts[features.broad_key()], class_index, 1.0)
                    self._add(self.medium_counts[features.medium_key()], class_index, 1.0)
                    self._add(self.specific_counts[features.specific_key()], class_index, 1.0)

        self.summary = ObservationSummary(total_queries=total_queries, total_cells=total_cells)
        return self.summary

    def predict_seed(self, seed_index: int, feature_map: SeedFeatureMap) -> list[list[list[float]]]:
        prediction: list[list[list[float]]] = []
        for y in range(feature_map.height):
            row: list[list[float]] = []
            for x in range(feature_map.width):
                row.append(self.predict_cell(seed_index, x, y, feature_map.get(x, y)))
            prediction.append(row)
        return prediction

    def predict_cell(self, seed_index: int, x: int, y: int, features: CellFeatures) -> list[float]:
        prior = self._heuristic_prior(features)
        posterior = [3.0 * value for value in prior]

        self._blend(posterior, self.terrain_counts.get(features.terrain_key()), 0.35)
        self._blend(posterior, self.broad_counts.get(features.broad_key()), 0.6)
        self._blend(posterior, self.medium_counts.get(features.medium_key()), 1.0)
        self._blend(posterior, self.specific_counts.get(features.specific_key()), 1.4)
        self._blend(posterior, self.direct_counts.get((seed_index, x, y)), 4.5)

        return normalize(posterior, self.floor)

    @staticmethod
    def _add(bucket: list[float], class_index: int, weight: float) -> None:
        bucket[class_index] += weight

    @staticmethod
    def _blend(target: list[float], counts: list[float] | None, scale: float) -> None:
        if not counts:
            return
        for index, value in enumerate(counts):
            target[index] += scale * value

    def _heuristic_prior(self, features: CellFeatures) -> list[float]:
        terrain = features.terrain

        if terrain == 10:
            return normalize([900.0, 0.2, 0.1, 0.1, 0.2, 0.1], self.floor)
        if terrain == 5:
            return normalize([0.1, 0.05, 0.05, 0.05, 0.05, 900.0], self.floor)
        if terrain == 2:
            return normalize(
                [
                    0.5,
                    6.0,
                    12.0 + 0.8 * features.coastal,
                    2.2 + 0.2 * features.ring2_ruin,
                    0.2,
                    0.05,
                ],
                self.floor,
            )
        if terrain == 1:
            return normalize(
                [
                    0.6,
                    11.0 + 0.8 * features.ring2_settlement,
                    2.5 if features.coastal else 0.5,
                    2.5 + 0.3 * features.ring2_ruin,
                    0.2,
                    0.05,
                ],
                self.floor,
            )
        if terrain == 3:
            return normalize(
                [
                    1.0,
                    2.5 + 0.8 * features.ring2_settlement,
                    1.8 if features.coastal else 0.2,
                    6.5,
                    4.5 + 0.2 * features.adj_forest,
                    0.05,
                ],
                self.floor,
            )
        if terrain == 4:
            settle_pressure = 1.2 * features.adj_settlement + 0.7 * features.ring2_settlement + 0.4 * features.adj_port
            return normalize(
                [
                    1.0,
                    1.4 + settle_pressure,
                    0.8 if features.coastal and settle_pressure > 0 else 0.15,
                    0.4 + 0.3 * features.adj_ruin,
                    max(2.5, 8.0 - settle_pressure),
                    0.05,
                ],
                self.floor,
            )
        if terrain in BUILDABLE_TERRAINS:
            expansion = 1.8 * features.adj_settlement + 1.0 * features.ring2_settlement + 0.8 * features.adj_port
            return normalize(
                [
                    max(1.5, 9.0 - expansion),
                    0.8 + expansion,
                    1.6 if features.coastal and expansion > 0 else 0.1,
                    0.4 + 0.2 * features.adj_ruin,
                    0.6 + 0.4 * features.adj_forest,
                    0.05,
                ],
                self.floor,
            )

        if terrain in STATIC_TERRAINS:
            output = [0.01] * 6
            output[terrain_to_class(terrain)] = 0.95
            return normalize(output, self.floor)

        return normalize([1.0, 1.0, 0.5, 0.5, 0.5, 0.1], self.floor)
