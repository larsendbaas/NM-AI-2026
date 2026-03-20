"""Backtest the solver against completed rounds with saved analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .features import build_feature_maps
from .io import load_json_file
from .scoring import score_round
from .solver import AstarIslandSolver
from .storage import RunStore


def discover_completed_rounds(workspace_root: Path) -> list[tuple[int, str]]:
    rounds: list[tuple[int, str]] = []
    runs_root = workspace_root / "runs"
    if not runs_root.exists():
        return rounds
    for round_dir in sorted(runs_root.iterdir()):
        if not round_dir.is_dir():
            continue
        detail_path = round_dir / "round_detail.json"
        if not detail_path.exists():
            continue
        detail = load_json_file(detail_path)
        if not any(round_dir.glob("analysis_seed_*.json")):
            continue
        rounds.append((int(detail["round_number"]), str(detail["id"])))
    rounds.sort()
    return rounds


def run_backtest(workspace_root: Path, round_id: str) -> dict[str, object]:
    solver = AstarIslandSolver(workspace_root=workspace_root)
    detail, store = solver.load_round(round_id)
    feature_maps = build_feature_maps(detail)
    observations = store.load_observations()
    model = solver._build_model(detail, feature_maps, observations, exclude_round_ids={round_id})
    predictions = [model.predict_seed(seed_index, feature_map) for seed_index, feature_map in enumerate(feature_maps)]

    ground_truths: list[list[list[list[float]]]] = []
    saved_predictions: list[list[list[list[float]]]] = []
    for seed_index in range(int(detail["seeds_count"])):
        analysis = load_json_file(store.root / f"analysis_seed_{seed_index}.json")
        ground_truths.append(analysis["ground_truth"])
        saved_prediction = store.load_prediction(seed_index)
        if saved_prediction is not None:
            saved_predictions.append(saved_prediction)

    return {
        "round_id": round_id,
        "round_number": int(detail["round_number"]),
        "queries": len(observations),
        "historical_rounds": model.summary.historical_rounds,
        "recomputed_score": score_round(ground_truths, predictions),
        "saved_score": score_round(ground_truths, saved_predictions) if len(saved_predictions) == len(ground_truths) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Astar Island solver on completed rounds.")
    parser.add_argument("--workspace", default=".", help="Workspace root containing runs/.")
    parser.add_argument("--round-id", action="append", help="Round id to evaluate. May be repeated.")
    args = parser.parse_args()

    workspace_root = Path(args.workspace).resolve()
    round_ids = args.round_id or [round_id for _, round_id in discover_completed_rounds(workspace_root)]
    if not round_ids:
        print("No completed rounds with analysis found.")
        return

    for round_id in round_ids:
        result = run_backtest(workspace_root, round_id)
        print(
            "Round {round_number} {round_id}: queries={queries}, history={historical_rounds}, "
            "recomputed={recomputed_score:.4f}, saved={saved_score}".format(**result)
        )


if __name__ == "__main__":
    main()
