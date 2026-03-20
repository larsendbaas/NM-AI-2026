"""High-level orchestration for fetching, collecting, predicting, and submitting."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import AstarIslandClient
from .features import build_feature_maps
from .io import load_json_file
from .model import TransitionModel
from .planner import (
    QueryWindow,
    ScoredWindow,
    build_coverage_plan,
    build_hotspot_candidates,
    dynamic_weight,
    pick_diverse_windows,
)
from .storage import RunStore


@dataclass
class CollectionResult:
    executed_queries: int
    remaining_budget: int


class AstarIslandSolver:
    """End-to-end solver workflow."""

    def __init__(self, workspace_root: Path, client: AstarIslandClient | None = None) -> None:
        self.workspace_root = workspace_root
        self.client = client or AstarIslandClient()

    def fetch_active_round(self) -> tuple[dict[str, Any], RunStore]:
        round_item = self.client.get_active_round()
        detail = self.client.get_round_detail(round_item["id"])
        store = RunStore.for_round(self.workspace_root, round_item["id"])
        store.save_json("round.json", round_item)
        store.save_json("round_detail.json", detail)
        return detail, store

    def load_round(self, round_id: str) -> tuple[dict[str, Any], RunStore]:
        store = RunStore.for_round(self.workspace_root, round_id)
        detail = store.load_json("round_detail.json")
        if detail is None:
            raise FileNotFoundError(f"No saved round_detail.json for round {round_id}")
        return detail, store

    def import_bootstrap_samples(self, round_id: str, store: RunStore) -> int:
        imported = 0
        observations = store.load_observations()
        seen = {self._observation_signature(item) for item in observations}
        docs_dir = self.workspace_root / "docs"
        if not docs_dir.exists():
            return 0

        for sample_path in sorted(docs_dir.glob("sim_seed*.json")):
            sample = load_json_file(sample_path)
            if sample.get("round_id") != round_id:
                continue
            if "viewport" not in sample or "grid" not in sample:
                continue
            seed_match = re.search(r"seed(\d+)", sample_path.stem)
            if not seed_match:
                continue
            payload = {
                "round_id": round_id,
                "seed_index": int(seed_match.group(1)),
                "viewport": sample["viewport"],
                "grid": sample["grid"],
                "settlements": sample.get("settlements", []),
                "source": f"bootstrap:{sample_path.name}",
            }
            signature = self._observation_signature(payload)
            if signature in seen:
                continue
            store.append_observation(payload)
            seen.add(signature)
            imported += 1
        return imported

    def collect(
        self,
        detail: dict[str, Any],
        store: RunStore,
        max_queries: int | None = None,
        dry_run: bool = False,
    ) -> CollectionResult:
        round_id = detail["id"]
        feature_maps = build_feature_maps(detail)
        coverage_plan = build_coverage_plan(detail)

        observations = store.load_observations()
        observed_counts = Counter(self._window_key(observation) for observation in observations)

        budget = self.client.get_budget()
        api_remaining_budget = int(budget["queries_max"]) - int(budget["queries_used"])
        remaining_to_execute = api_remaining_budget
        if max_queries is not None:
            remaining_to_execute = min(remaining_to_execute, max_queries)

        executed = 0
        coverage_counts = Counter(query.key() for query in coverage_plan)
        coverage_executed, remaining_to_execute = self._execute_plan(
            plan=coverage_plan,
            planned_counts=coverage_counts,
            observed_counts=observed_counts,
            remaining_to_execute=remaining_to_execute,
            round_id=round_id,
            store=store,
            dry_run=dry_run,
        )
        executed += coverage_executed

        hotspot_plan: list[QueryWindow] = []
        if remaining_to_execute > 0:
            observations = store.load_observations()
            hotspot_plan = self._build_adaptive_hotspot_plan(detail, feature_maps, observations, remaining_to_execute)
            combined_counts = coverage_counts + Counter(query.key() for query in hotspot_plan)
            hotspot_executed, remaining_to_execute = self._execute_plan(
                plan=hotspot_plan,
                planned_counts=combined_counts,
                observed_counts=observed_counts,
                remaining_to_execute=remaining_to_execute,
                round_id=round_id,
                store=store,
                dry_run=dry_run,
            )
            executed += hotspot_executed

        full_plan = coverage_plan + hotspot_plan
        store.save_json("query_plan.json", [query.__dict__ for query in full_plan])

        actual_remaining_budget = api_remaining_budget if dry_run else api_remaining_budget - executed
        return CollectionResult(executed_queries=executed, remaining_budget=actual_remaining_budget)

    def predict(self, detail: dict[str, Any], store: RunStore) -> dict[str, Any]:
        feature_maps = build_feature_maps(detail)
        observations = store.load_observations()
        model = self._build_model(detail, feature_maps, observations)
        summary = model.summary

        prediction_paths: list[str] = []
        for seed_index, feature_map in enumerate(feature_maps):
            prediction = model.predict_seed(seed_index, feature_map)
            path = store.save_prediction(seed_index, prediction)
            prediction_paths.append(str(path))

        report = {
            "round_id": detail["id"],
            "seeds": int(detail["seeds_count"]),
            "queries_loaded": summary.total_queries,
            "observed_cells": summary.total_cells,
            "historical_cells": summary.historical_cells,
            "historical_rounds": summary.historical_rounds,
            "prediction_paths": prediction_paths,
        }
        store.save_json("prediction_report.json", report)
        return report

    def submit(self, detail: dict[str, Any], store: RunStore) -> list[dict[str, Any]]:
        responses: list[dict[str, Any]] = []
        for seed_index in range(int(detail["seeds_count"])):
            prediction = store.load_prediction(seed_index)
            if prediction is None:
                raise FileNotFoundError(f"Missing prediction file for seed {seed_index}")
            responses.append(self.client.submit(detail["id"], seed_index, prediction))
        store.save_json("submit_response.json", responses)
        return responses

    @staticmethod
    def _window_key(payload: dict[str, Any]) -> tuple[int, int, int, int, int]:
        viewport = payload["viewport"]
        return (
            int(payload["seed_index"]),
            int(viewport["x"]),
            int(viewport["y"]),
            int(viewport["w"]),
            int(viewport["h"]),
        )

    @staticmethod
    def _observation_signature(payload: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
        viewport = payload["viewport"]
        grid_blob = json.dumps(payload["grid"], separators=(",", ":"))
        return (
            int(payload["seed_index"]),
            int(viewport["x"]),
            int(viewport["y"]),
            int(viewport["w"]),
            int(viewport["h"]),
            grid_blob,
        )

    def _build_model(
        self,
        detail: dict[str, Any],
        feature_maps: list[Any],
        observations: list[dict[str, Any]],
        exclude_round_ids: set[str] | None = None,
    ) -> TransitionModel:
        model = TransitionModel()
        round_ids = set(exclude_round_ids or ())
        round_ids.add(str(detail["id"]))
        self._load_historical_analysis(model, exclude_round_ids=round_ids)
        model.fit(feature_maps, observations)
        return model

    def _load_historical_analysis(self, model: TransitionModel, exclude_round_ids: set[str]) -> None:
        runs_root = self.workspace_root / "runs"
        if not runs_root.exists():
            return
        for round_dir in sorted(runs_root.iterdir()):
            if not round_dir.is_dir():
                continue
            detail_path = round_dir / "round_detail.json"
            if not detail_path.exists():
                continue
            round_detail = load_json_file(detail_path)
            round_id = str(round_detail.get("id", ""))
            if round_id in exclude_round_ids:
                continue
            analysis_paths = sorted(round_dir.glob("analysis_seed_*.json"))
            if not analysis_paths:
                continue
            feature_maps = build_feature_maps(round_detail)
            loaded_any = False
            for path in analysis_paths:
                seed_match = re.search(r"analysis_seed_(\d+)", path.stem)
                if not seed_match:
                    continue
                seed_index = int(seed_match.group(1))
                if seed_index >= len(feature_maps):
                    continue
                analysis = load_json_file(path)
                ground_truth = analysis.get("ground_truth")
                if not ground_truth:
                    continue
                model.add_historical_seed(feature_maps[seed_index], ground_truth)
                loaded_any = True
            if loaded_any:
                model.register_historical_round()

    def _build_adaptive_hotspot_plan(
        self,
        detail: dict[str, Any],
        feature_maps: list[Any],
        observations: list[dict[str, Any]],
        budget: int,
    ) -> list[QueryWindow]:
        if budget <= 0:
            return []

        model = self._build_model(detail, feature_maps, observations)
        predictions = [model.predict_seed(seed_index, feature_map) for seed_index, feature_map in enumerate(feature_maps)]
        observed_cell_counts = self._observed_cell_counts(observations)

        scored_windows: list[ScoredWindow] = []
        for candidate in build_hotspot_candidates(detail, feature_maps, label="adaptive_hotspot"):
            score = self._adaptive_window_score(candidate.query, feature_maps, predictions, observed_cell_counts)
            scored_windows.append(ScoredWindow(score, candidate.query))

        scored_windows.sort(key=lambda item: item.score, reverse=True)
        return pick_diverse_windows(scored_windows, budget, per_seed_cap=2)

    def _adaptive_window_score(
        self,
        query: QueryWindow,
        feature_maps: list[Any],
        predictions: list[list[list[list[float]]]],
        observed_cell_counts: Counter[tuple[int, int, int]],
    ) -> float:
        feature_map = feature_maps[query.seed_index]
        prediction = predictions[query.seed_index]
        total = 0.0
        for y in range(query.y, min(feature_map.height, query.y + query.h)):
            for x in range(query.x, min(feature_map.width, query.x + query.w)):
                cell_prediction = prediction[y][x]
                entropy = -sum(value * math.log(max(value, 1e-12)) for value in cell_prediction)
                dynamic_mass = cell_prediction[1] + cell_prediction[2] + cell_prediction[3] + 0.55 * cell_prediction[4]
                importance = 0.55 + math.log1p(dynamic_weight(feature_map.get(x, y)))
                repeat_penalty = math.sqrt(1.0 + observed_cell_counts[(query.seed_index, x, y)])
                total += (entropy + 0.45 * dynamic_mass) * importance / repeat_penalty
        return total

    @staticmethod
    def _observed_cell_counts(observations: list[dict[str, Any]]) -> Counter[tuple[int, int, int]]:
        counts: Counter[tuple[int, int, int]] = Counter()
        for observation in observations:
            viewport = observation["viewport"]
            seed_index = int(observation["seed_index"])
            for y in range(int(viewport["y"]), int(viewport["y"]) + int(viewport["h"])):
                for x in range(int(viewport["x"]), int(viewport["x"]) + int(viewport["w"])):
                    counts[(seed_index, x, y)] += 1
        return counts

    def _execute_plan(
        self,
        plan: list[QueryWindow],
        planned_counts: Counter[tuple[int, int, int, int, int]],
        observed_counts: Counter[tuple[int, int, int, int, int]],
        remaining_to_execute: int,
        round_id: str,
        store: RunStore,
        dry_run: bool,
    ) -> tuple[int, int]:
        executed = 0
        for query in plan:
            if remaining_to_execute <= 0:
                break
            if observed_counts[query.key()] >= planned_counts[query.key()]:
                continue
            if dry_run:
                observed_counts[query.key()] += 1
                remaining_to_execute -= 1
                executed += 1
                continue

            result = self.client.simulate(
                round_id=round_id,
                seed_index=query.seed_index,
                viewport_x=query.x,
                viewport_y=query.y,
                viewport_w=query.w,
                viewport_h=query.h,
            )
            payload = {
                "round_id": round_id,
                "seed_index": query.seed_index,
                "viewport": result["viewport"],
                "grid": result["grid"],
                "settlements": result.get("settlements", []),
                "label": query.label,
            }
            store.append_observation(payload)
            observed_counts[query.key()] += 1
            remaining_to_execute -= 1
            executed += 1
            time.sleep(0.25)
        return executed, remaining_to_execute
