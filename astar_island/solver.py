"""High-level orchestration for fetching, collecting, predicting, and submitting."""

from __future__ import annotations

import json
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
from .planner import build_query_plan
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
        plan = build_query_plan(detail, feature_maps)
        store.save_json("query_plan.json", [query.__dict__ for query in plan])

        observations = store.load_observations()
        observed_counts = Counter(self._window_key(observation) for observation in observations)
        planned_counts = Counter(query.key() for query in plan)

        budget = self.client.get_budget()
        api_remaining_budget = int(budget["queries_max"]) - int(budget["queries_used"])
        remaining_to_execute = api_remaining_budget
        if max_queries is not None:
            remaining_to_execute = min(remaining_to_execute, max_queries)

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

        actual_remaining_budget = api_remaining_budget if dry_run else api_remaining_budget - executed
        return CollectionResult(executed_queries=executed, remaining_budget=actual_remaining_budget)

    def predict(self, detail: dict[str, Any], store: RunStore) -> dict[str, Any]:
        feature_maps = build_feature_maps(detail)
        observations = store.load_observations()
        model = TransitionModel()
        summary = model.fit(feature_maps, observations)

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
