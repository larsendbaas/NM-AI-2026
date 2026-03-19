"""Query-planning logic for the live budget."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

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


@dataclass(frozen=True)
class ScoredWindow:
    score: float
    query: QueryWindow


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
        return 8.5 + 0.8 * feature.ring2_settlement + 0.3 * feature.ring3_settlement + 0.7 * feature.coastal
    if terrain == 1:
        return 7.0 + 0.8 * feature.ring2_settlement + 0.4 * feature.ring3_settlement + 0.5 * feature.adj_port
    if terrain == 3:
        return 5.5 + 0.6 * feature.ring2_settlement + 0.3 * feature.ring3_ruin + 0.4 * feature.coastal
    if terrain == 4:
        return (
            1.2
            + 1.0 * feature.adj_settlement
            + 0.7 * feature.ring2_settlement
            + 0.35 * feature.ring3_settlement
            + 0.4 * feature.adj_ruin
        )
    if terrain in BUILDABLE_TERRAINS:
        return (
            1.2
            + 2.0 * feature.adj_settlement
            + 1.0 * feature.adj_port
            + 0.9 * feature.ring2_settlement
            + 0.5 * feature.ring3_settlement
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


def build_coverage_plan(
    round_detail: dict[str, object],
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
    return plan


def build_hotspot_candidates(
    round_detail: dict[str, object],
    feature_maps: list[SeedFeatureMap],
    window: int = 15,
    step: int = 2,
    label: str = "hotspot",
) -> list[ScoredWindow]:
    width = int(round_detail["map_width"])
    height = int(round_detail["map_height"])
    candidates: list[ScoredWindow] = []
    for seed_index, feature_map in enumerate(feature_maps):
        for y in range(0, max(1, height - window + 1), step):
            for x in range(0, max(1, width - window + 1), step):
                score = score_window(feature_map, x, y, window)
                candidates.append(ScoredWindow(score, QueryWindow(seed_index, x, y, window, window, label)))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def overlap_ratio(left: QueryWindow, right: QueryWindow) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.w, right.x + right.w)
    y2 = min(left.y + left.h, right.y + right.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    smaller_area = min(left.w * left.h, right.w * right.h)
    return intersection / smaller_area


def pick_diverse_windows(
    scored_windows: Iterable[ScoredWindow],
    count: int,
    per_seed_cap: int | None = 2,
    max_overlap: float = 0.78,
) -> list[QueryWindow]:
    selected: list[QueryWindow] = []
    picked_per_seed: Counter[int] = Counter()
    for item in scored_windows:
        if len(selected) >= count:
            break
        query = item.query
        if per_seed_cap is not None and picked_per_seed[query.seed_index] >= per_seed_cap:
            continue
        if any(existing.seed_index == query.seed_index and overlap_ratio(existing, query) > max_overlap for existing in selected):
            continue
        selected.append(query)
        picked_per_seed[query.seed_index] += 1
    return selected


def build_query_plan(
    round_detail: dict[str, object],
    feature_maps: list[SeedFeatureMap],
    total_budget: int = 50,
    window: int = 15,
) -> list[QueryWindow]:
    plan = build_coverage_plan(round_detail, window=window)
    remaining = max(0, total_budget - len(plan))
    if remaining == 0:
        return plan

    hotspot_candidates = build_hotspot_candidates(round_detail, feature_maps, window=window)
    plan.extend(pick_diverse_windows(hotspot_candidates, remaining, per_seed_cap=2))
    return plan
