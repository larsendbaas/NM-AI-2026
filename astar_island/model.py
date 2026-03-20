"""Probabilistic transition model for Astar Island."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .constants import BUILDABLE_TERRAINS, STATIC_TERRAINS, terrain_to_class
from .features import CellFeatures, SeedFeatureMap


MAX_CLASS_ENTROPY = math.log(6.0)


def normalize(weights: list[float], floor: float = 0.01) -> list[float]:
    clipped = [max(floor, float(value)) for value in weights]
    total = sum(clipped)
    return [value / total for value in clipped]


@dataclass
class ObservationSummary:
    total_queries: int
    total_cells: int
    historical_cells: int
    historical_rounds: int


@dataclass
class CountTables:
    global_counts: list[float] = field(default_factory=lambda: [0.0] * 6)
    direct_counts: dict[tuple[int, int, int], list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0] * 6))
    terrain_counts: dict[tuple[int, ...], list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0] * 6))
    broad_counts: dict[tuple[int, ...], list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0] * 6))
    medium_counts: dict[tuple[int, ...], list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0] * 6))
    specific_counts: dict[tuple[int, ...], list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0] * 6))
    observed_global_counts: dict[tuple[int, ...], list[float]] = field(
        default_factory=lambda: defaultdict(lambda: [0.0] * 6)
    )
    observed_terrain_counts: dict[tuple[int, ...], list[float]] = field(
        default_factory=lambda: defaultdict(lambda: [0.0] * 6)
    )
    observed_context_counts: dict[tuple[int, ...], list[float]] = field(
        default_factory=lambda: defaultdict(lambda: [0.0] * 6)
    )


class TransitionModel:
    """Learns smoothed transition distributions from sampled viewports."""

    def __init__(self, floor: float = 0.01) -> None:
        self.floor = floor
        self.live = CountTables()
        self.history = CountTables()
        self.calibration_global_counts: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0] * 6)
        self.calibration_terrain_counts: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0] * 6)
        self.calibration_context_counts: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0] * 6)
        self.summary = ObservationSummary(total_queries=0, total_cells=0, historical_cells=0, historical_rounds=0)

    def add_historical_seed(
        self,
        feature_map: SeedFeatureMap,
        ground_truth: list[list[list[float]]],
        cell_weight: float = 0.28,
    ) -> int:
        added_cells = 0
        for y, row in enumerate(ground_truth):
            for x, distribution in enumerate(row):
                added_cells += 1
                features = feature_map.get(x, y)
                entropy_weight = self._historical_cell_weight(distribution, base_weight=cell_weight)
                self._observe(
                    self.history,
                    seed_index=None,
                    x=x,
                    y=y,
                    features=features,
                    distribution=distribution,
                    weight=entropy_weight,
                    include_direct=False,
                )
        self.summary = ObservationSummary(
            total_queries=self.summary.total_queries,
            total_cells=self.summary.total_cells,
            historical_cells=self.summary.historical_cells + added_cells,
            historical_rounds=self.summary.historical_rounds,
        )
        return added_cells

    def add_historical_observation_seed(
        self,
        feature_map: SeedFeatureMap,
        observed_counts: dict[tuple[int, int], list[float]],
        ground_truth: list[list[list[float]]],
        cell_weight: float = 0.42,
    ) -> int:
        added_cells = 0
        for y, row in enumerate(ground_truth):
            for x, distribution in enumerate(row):
                sample_counts = observed_counts.get((x, y))
                if not sample_counts:
                    continue
                added_cells += 1
                features = feature_map.get(x, y)
                sample_key = self._observed_sample_key(sample_counts)
                entropy_weight = self._historical_cell_weight(distribution, base_weight=cell_weight)
                self._add(self.history.observed_global_counts[sample_key], distribution, entropy_weight)
                self._add(
                    self.history.observed_terrain_counts[(features.terrain,) + sample_key],
                    distribution,
                    entropy_weight,
                )
                self._add(
                    self.history.observed_context_counts[self._observed_context_key(features, sample_key)],
                    distribution,
                    entropy_weight,
                )
        return added_cells

    def add_residual_calibration_seed(
        self,
        feature_map: SeedFeatureMap,
        observed_counts: dict[tuple[int, int], list[float]],
        ground_truth: list[list[list[float]]],
        cell_weight: float = 0.34,
    ) -> int:
        added_cells = 0
        for y, row in enumerate(ground_truth):
            for x, distribution in enumerate(row):
                features = feature_map.get(x, y)
                sample_counts = observed_counts.get((x, y))
                base_prediction = self._predict_base_distribution(features, sample_counts)
                calibration_key = self._calibration_key(base_prediction, sample_counts)
                context_key = self._calibration_context_key(features, calibration_key)
                entropy_weight = self._historical_cell_weight(distribution, base_weight=cell_weight)
                self._add(self.calibration_global_counts[calibration_key], distribution, entropy_weight)
                self._add(
                    self.calibration_terrain_counts[(features.terrain,) + calibration_key],
                    distribution,
                    entropy_weight,
                )
                self._add(self.calibration_context_counts[context_key], distribution, entropy_weight)
                added_cells += 1
        return added_cells

    def register_historical_round(self) -> None:
        self.summary = ObservationSummary(
            total_queries=self.summary.total_queries,
            total_cells=self.summary.total_cells,
            historical_cells=self.summary.historical_cells,
            historical_rounds=self.summary.historical_rounds + 1,
        )

    def fit(
        self,
        feature_maps: list[SeedFeatureMap],
        observations: Iterable[dict[str, object]],
    ) -> ObservationSummary:
        self.live = CountTables()
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
                    features = feature_map.get(x, y)
                    class_index = terrain_to_class(int(terrain_code))
                    distribution = [0.0] * 6
                    distribution[class_index] = 1.0
                    self._observe(
                        self.live,
                        seed_index=seed_index,
                        x=x,
                        y=y,
                        features=features,
                        distribution=distribution,
                        weight=1.0,
                        include_direct=True,
                    )

        self.summary = ObservationSummary(
            total_queries=total_queries,
            total_cells=total_cells,
            historical_cells=self.summary.historical_cells,
            historical_rounds=self.summary.historical_rounds,
        )
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
        sample_counts = self.live.direct_counts.get((seed_index, x, y))
        return self._predict_local_distribution(features, sample_counts)

    @staticmethod
    def _add(bucket: list[float], distribution: list[float], weight: float) -> None:
        for index, value in enumerate(distribution):
            bucket[index] += weight * float(value)

    @staticmethod
    def _distribution(counts: list[float] | None) -> list[float] | None:
        if not counts:
            return None
        total = sum(counts)
        if total <= 0.0:
            return None
        return [value / total for value in counts]

    @classmethod
    def _blend(cls, target: list[float], counts: list[float] | None, scale: float, cap: float | None = None) -> None:
        if not counts:
            return
        total = sum(counts)
        if total <= 0.0:
            return
        strength = math.log1p(total)
        if cap is not None:
            strength = min(strength, cap)
        distribution = cls._distribution(counts)
        if distribution is None:
            return
        for index, value in enumerate(distribution):
            target[index] += scale * strength * value

    def _blend_observed_cell(
        self,
        target: list[float],
        sample_counts: list[float] | None,
        features: CellFeatures,
    ) -> None:
        if not sample_counts:
            return

        sample_key = self._observed_sample_key(sample_counts)
        self._blend(target, self.history.observed_global_counts.get(sample_key), 1.1)
        self._blend(target, self.history.observed_terrain_counts.get((features.terrain,) + sample_key), 1.8)
        self._blend(target, self.history.observed_context_counts.get(self._observed_context_key(features, sample_key)), 2.4)

        observed_total = sum(sample_counts)
        direct_scale = 0.3 + 0.35 * min(3.0, observed_total)
        direct_cap = 0.6 + 0.45 * min(3.0, observed_total)
        self._blend(target, sample_counts, direct_scale, cap=direct_cap)

    def _predict_base_distribution(self, features: CellFeatures, sample_counts: list[float] | None) -> list[float]:
        prior = self._prior_distribution(features)
        posterior = [6.0 * value for value in prior]

        self._blend(posterior, self.history.global_counts, 0.18)
        self._blend(posterior, self.history.terrain_counts.get(features.terrain_key()), 0.45)
        self._blend(posterior, self.history.broad_counts.get(features.broad_key()), 0.85)
        self._blend(posterior, self.history.medium_counts.get(features.medium_key()), 1.25)
        self._blend(posterior, self.history.specific_counts.get(features.specific_key()), 1.2)

        self._blend(posterior, self.live.global_counts, 0.35)
        self._blend(posterior, self.live.terrain_counts.get(features.terrain_key()), 0.65)
        self._blend(posterior, self.live.broad_counts.get(features.broad_key()), 1.0)
        self._blend(posterior, self.live.medium_counts.get(features.medium_key()), 1.45)
        self._blend(posterior, self.live.specific_counts.get(features.specific_key()), 1.7)
        self._blend_observed_cell(posterior, sample_counts, features)
        return normalize(posterior, self.floor)

    def _predict_local_distribution(self, features: CellFeatures, sample_counts: list[float] | None) -> list[float]:
        base_prediction = self._predict_base_distribution(features, sample_counts)
        return self._apply_residual_calibration(base_prediction, features, sample_counts)

    def _apply_residual_calibration(
        self,
        base_prediction: list[float],
        features: CellFeatures,
        sample_counts: list[float] | None,
    ) -> list[float]:
        calibration_key = self._calibration_key(base_prediction, sample_counts)
        context_key = self._calibration_context_key(features, calibration_key)
        posterior = [4.2 * value for value in base_prediction]

        self._blend(posterior, self.calibration_global_counts.get(calibration_key), 0.75)
        self._blend(posterior, self.calibration_terrain_counts.get((features.terrain,) + calibration_key), 1.15)
        self._blend(posterior, self.calibration_context_counts.get(context_key), 1.35)
        return normalize(posterior, self.floor)

    def _prior_distribution(self, features: CellFeatures) -> list[float]:
        heuristic = self._heuristic_prior(features)
        terrain_history = self._distribution(self.history.terrain_counts.get(features.terrain_key()))
        if terrain_history is None:
            return heuristic
        return normalize(
            [0.78 * heuristic[index] + 0.22 * terrain_history[index] for index in range(6)],
            self.floor,
        )

    def _observe(
        self,
        tables: CountTables,
        seed_index: int | None,
        x: int,
        y: int,
        features: CellFeatures,
        distribution: list[float],
        weight: float,
        include_direct: bool,
    ) -> None:
        self._add(tables.global_counts, distribution, weight)
        if include_direct and seed_index is not None:
            self._add(tables.direct_counts[(seed_index, x, y)], distribution, weight)
        self._add(tables.terrain_counts[features.terrain_key()], distribution, weight)
        self._add(tables.broad_counts[features.broad_key()], distribution, weight)
        self._add(tables.medium_counts[features.medium_key()], distribution, weight)
        self._add(tables.specific_counts[features.specific_key()], distribution, weight)

    @staticmethod
    def _entropy(distribution: list[float]) -> float:
        return -sum(value * math.log(max(value, 1e-12)) for value in distribution if value > 0.0)

    @staticmethod
    def _observed_sample_key(sample_counts: list[float]) -> tuple[int, ...]:
        integer_counts = tuple(min(3, int(round(value))) for value in sample_counts)
        total_count = min(3, sum(integer_counts))
        return (total_count,) + integer_counts

    @staticmethod
    def _observed_context_key(features: CellFeatures, sample_key: tuple[int, ...]) -> tuple[int, ...]:
        return (
            features.terrain,
            features.coastal,
            features.adj_settlement,
            features.adj_port,
            features.adj_ruin,
            features.adj_forest,
            features.dist_settlement,
            features.dist_ruin,
            features.dist_ocean,
        ) + sample_key

    @classmethod
    def _calibration_key(cls, prediction: list[float], sample_counts: list[float] | None) -> tuple[int, ...]:
        dominant_class = max(range(6), key=prediction.__getitem__)
        sorted_prediction = sorted(prediction, reverse=True)
        top_probability = prediction[dominant_class]
        entropy_bin = min(5, int(6.0 * cls._entropy(prediction) / MAX_CLASS_ENTROPY))
        confidence_bin = min(5, int(6.0 * top_probability))
        margin_bin = min(5, int(10.0 * max(0.0, sorted_prediction[0] - sorted_prediction[1])))
        observed_total = min(3, int(sum(sample_counts))) if sample_counts else 0
        observed_class = 6 if not sample_counts else max(range(6), key=sample_counts.__getitem__)
        return (observed_total, dominant_class, observed_class, confidence_bin, entropy_bin, margin_bin)

    @staticmethod
    def _calibration_context_key(features: CellFeatures, calibration_key: tuple[int, ...]) -> tuple[int, ...]:
        return (
            features.terrain,
            features.coastal,
            features.adj_settlement,
            features.adj_port,
            features.adj_ruin,
            features.adj_forest,
            features.dist_settlement,
            features.dist_ruin,
            features.dist_ocean,
        ) + calibration_key

    @classmethod
    def _historical_cell_weight(cls, distribution: list[float], base_weight: float) -> float:
        normalized_entropy = cls._entropy(distribution) / MAX_CLASS_ENTROPY
        return base_weight * (0.4 + 1.2 * normalized_entropy)

    def _heuristic_prior(self, features: CellFeatures) -> list[float]:
        terrain = features.terrain

        if terrain == 10:
            return normalize([900.0, 0.2, 0.1, 0.1, 0.2, 0.1], self.floor)
        if terrain == 5:
            return normalize([0.1, 0.05, 0.05, 0.05, 0.05, 900.0], self.floor)
        if terrain == 2:
            return normalize(
                [
                    0.8,
                    4.8 + 0.5 * features.ring2_settlement,
                    10.0 + 1.0 * features.coastal + 0.3 * features.ocean_neighbors,
                    1.8 + 0.3 * features.ring2_ruin,
                    0.35,
                    0.05,
                ],
                self.floor,
            )
        if terrain == 1:
            return normalize(
                [
                    1.0,
                    8.5 + 0.9 * features.ring2_settlement + 0.4 * features.ring3_settlement,
                    1.8 if features.coastal else 0.35,
                    2.0 + 0.35 * features.ring2_ruin,
                    0.35,
                    0.05,
                ],
                self.floor,
            )
        if terrain == 3:
            return normalize(
                [
                    1.6,
                    2.2 + 0.8 * features.ring2_settlement + 0.4 * features.ring3_settlement,
                    1.2 if features.coastal else 0.2,
                    5.6 + 0.2 * features.ring2_ruin,
                    3.8 + 0.3 * features.adj_forest + 0.2 * features.ring3_forest,
                    0.05,
                ],
                self.floor,
            )
        if terrain == 4:
            settle_pressure = (
                0.9 * features.adj_settlement
                + 0.55 * features.ring2_settlement
                + 0.25 * features.ring3_settlement
                + 0.35 * features.adj_port
            )
            return normalize(
                [
                    1.8,
                    0.7 + settle_pressure,
                    0.45 if features.coastal and settle_pressure > 0 else 0.1,
                    0.35 + 0.35 * features.adj_ruin + 0.1 * features.ring3_ruin,
                    max(3.8, 7.0 - 0.75 * settle_pressure + 0.2 * features.ring3_forest),
                    0.05,
                ],
                self.floor,
            )
        if terrain in BUILDABLE_TERRAINS:
            expansion = (
                1.5 * features.adj_settlement
                + 0.95 * features.ring2_settlement
                + 0.45 * features.ring3_settlement
                + 0.8 * features.adj_port
            )
            forestry = 0.55 * features.adj_forest + 0.3 * features.ring2_forest + 0.15 * features.ring3_forest
            return normalize(
                [
                    max(2.5, 10.5 - expansion - 0.25 * forestry),
                    0.7 + expansion,
                    1.2 if features.coastal and expansion > 0 else 0.08,
                    0.35 + 0.25 * features.adj_ruin + 0.2 * features.ring2_ruin,
                    0.7 + forestry,
                    0.05,
                ],
                self.floor,
            )

        if terrain in STATIC_TERRAINS:
            output = [0.01] * 6
            output[terrain_to_class(terrain)] = 0.95
            return normalize(output, self.floor)

        return normalize([1.0, 1.0, 0.5, 0.5, 0.5, 0.1], self.floor)
