"""Query-planning logic for the live budget."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .constants import BUILDABLE_TERRAINS, SETTLEMENT_TERRAINS
from .features import CellFeatures, SeedFeatureMap


@dataclass(frozen=True)
class QueryWindow:
    seed_index: int
    x: int
    y: int
    w: int
    h: int
    label: str

    def key(self) -> tuple[int, int, int, int, int]:
        return (self.seed_index, self.x, self.y, self.w, self.h)


def coverage_positions(length: int, window: int, overlap: int = 2) -> list[int]:
    if window >= length:
        return [0]
    step = max(1, window - overlap)
    positions = list(range(0, max(1, length - window + 1), step))
    final_start = length - window
    if positions[-1] != final_start:
        positions.append(final_start)
    return positions


def dynamic_weight(feature: CellFeatures) -> float:
    terrain = feature.terrain
    if terrain == 10:
        return 0.01
    if terrain == 5:
        return 0.05
    if terrain == 2:
        return 8.0 + 0.6 * feature.ring2_settlement + 0.5 * feature.coastal
    if terrain == 1:
        return 7.0 + 0.5 * feature.ring2_settlement + 0.4 * feature.adj_port
    if terrain == 3:
        return 6.0 + 0.4 * feature.ring2_settlement + 0.4 * feature.coastal
    if terrain == 4:
        return (
            1.5
            + 1.3 * feature.adj_settlement
            + 0.6 * feature.ring2_settlement
            + 0.4 * feature.adj_ruin
        )
    if terrain in BUILDABLE_TERRAINS:
        return (
            1.0
            + 1.8 * feature.adj_settlement
            + 1.0 * feature.adj_port
            + 0.7 * feature.ring2_settlement
            + 0.3 * feature.ring2_ruin
            + 0.2 * feature.coastal
        )
    if terrain in SETTLEMENT_TERRAINS:
        return 6.0
    return 1.0


def score_window(feature_map: SeedFeatureMap, x: int, y: int, window: int) -> float:
    total = 0.0
    for ny in range(y, min(feature_map.height, y + window)):
        for nx in range(x, min(feature_map.width, x + window)):
            total += dynamic_weight(feature_map.get(nx, ny))
    return total


def build_query_plan(
    round_detail: dict[str, object],
    feature_maps: list[SeedFeatureMap],
    total_budget: int = 50,
    window: int = 15,
) -> list[QueryWindow]:
    width = int(round_detail["map_width"])
    height = int(round_detail["map_height"])
    coverage_x = coverage_positions(width, window)
    coverage_y = coverage_positions(height, window)

    plan: list[QueryWindow] = []
    for seed_index in range(int(round_detail["seeds_count"])):
        for y in coverage_y:
            for x in coverage_x:
                plan.append(QueryWindow(seed_index, x, y, window, window, "coverage"))

    remaining = max(0, total_budget - len(plan))
    if remaining == 0:
        return plan

    hotspot_candidates: list[tuple[float, QueryWindow]] = []
    for seed_index, feature_map in enumerate(feature_maps):
        for y in range(0, max(1, height - window + 1), 2):
            for x in range(0, max(1, width - window + 1), 2):
                score = score_window(feature_map, x, y, window)
                hotspot_candidates.append((score, QueryWindow(seed_index, x, y, window, window, "hotspot")))

    hotspot_candidates.sort(key=lambda item: item[0], reverse=True)
    picked_per_seed: Counter[int] = Counter()
    for _, query in hotspot_candidates:
        if remaining == 0:
            break
        if picked_per_seed[query.seed_index] >= 1:
            continue
        plan.append(query)
        picked_per_seed[query.seed_index] += 1
        remaining -= 1

    index = 0
    while remaining > 0 and hotspot_candidates:
        plan.append(hotspot_candidates[index % len(hotspot_candidates)][1])
        remaining -= 1
        index += 1

    return plan
