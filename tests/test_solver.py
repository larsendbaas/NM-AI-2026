import json
import tempfile
import unittest
from pathlib import Path

from astar_island.features import build_feature_maps
from astar_island.io import load_json_file
from astar_island.model import TransitionModel
from astar_island.planner import build_query_plan
from astar_island.solver import AstarIslandSolver
from astar_island.storage import RunStore


ROOT = Path(__file__).resolve().parents[1]


class SolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.round_detail = load_json_file(ROOT / "docs" / "round_detail.json")
        self.sample = load_json_file(ROOT / "docs" / "sim_seed0_0_0.json")
        self.sample_with_round = dict(self.sample)
        self.sample_with_round["round_id"] = self.round_detail["id"]

    def test_query_plan_uses_full_budget(self) -> None:
        feature_maps = build_feature_maps(self.round_detail)
        plan = build_query_plan(self.round_detail, feature_maps)
        self.assertEqual(50, len(plan))

    def test_prediction_shape_and_normalization(self) -> None:
        feature_maps = build_feature_maps(self.round_detail)
        model = TransitionModel()
        observation = {
            "round_id": self.round_detail["id"],
            "seed_index": 0,
            "viewport": self.sample["viewport"],
            "grid": self.sample["grid"],
            "settlements": self.sample["settlements"],
        }
        model.fit(feature_maps, [observation])
        prediction = model.predict_seed(0, feature_maps[0])

        self.assertEqual(self.round_detail["map_height"], len(prediction))
        self.assertEqual(self.round_detail["map_width"], len(prediction[0]))
        for row in prediction:
            for cell in row:
                self.assertEqual(6, len(cell))
                self.assertAlmostEqual(1.0, sum(cell), places=6)
                self.assertTrue(all(value > 0.0 for value in cell))

    def test_bootstrap_import_picks_up_existing_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "docs").mkdir()
            (workspace / "docs" / "sim_seed0_0_0.json").write_text(
                json.dumps(self.sample_with_round),
                encoding="utf-8",
            )
            solver = AstarIslandSolver(workspace)
            store = RunStore.for_round(workspace, self.round_detail["id"])
            imported = solver.import_bootstrap_samples(self.round_detail["id"], store)
            self.assertEqual(1, imported)
            observations = store.load_observations()
            self.assertEqual(1, len(observations))
            self.assertEqual(self.round_detail["id"], observations[0]["round_id"])

    def test_bootstrap_import_skips_sample_without_round_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / "docs").mkdir()
            (workspace / "docs" / "sim_seed0_0_0.json").write_text(
                json.dumps(self.sample),
                encoding="utf-8",
            )
            solver = AstarIslandSolver(workspace)
            store = RunStore.for_round(workspace, self.round_detail["id"])
            imported = solver.import_bootstrap_samples(self.round_detail["id"], store)
            self.assertEqual(0, imported)
            self.assertEqual([], store.load_observations())

    def test_run_store_resolves_labeled_round_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            labeled = workspace / "runs" / f"{self.round_detail['id']} (Round 1)"
            labeled.mkdir(parents=True)
            store = RunStore.for_round(workspace, self.round_detail["id"])
            self.assertEqual(labeled, store.root)

    def test_sync_completed_analyses_saves_round_and_seed_files(self) -> None:
        class FakeClient:
            def get_my_rounds(self) -> list[dict[str, object]]:
                return [
                    {
                        "id": self.round_detail["id"],
                        "status": "completed",
                        "round_score": 77.9,
                        "seeds_submitted": 5,
                    }
                ]

            def get_round_detail(self, round_id: str) -> dict[str, object]:
                self.assertEqual(self.round_detail["id"], round_id)
                return self.round_detail

            def get_analysis(self, round_id: str, seed_index: int) -> dict[str, object]:
                self.assertEqual(self.round_detail["id"], round_id)
                return {
                    "prediction": [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
                    "ground_truth": [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
                    "score": 100.0,
                    "width": 1,
                    "height": 1,
                    "seed_index": seed_index,
                }

            def __init__(self, round_detail: dict[str, object], testcase: unittest.TestCase) -> None:
                self.round_detail = round_detail
                self.testcase = testcase

            def assertEqual(self, left: object, right: object) -> None:
                self.testcase.assertEqual(left, right)

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            solver = AstarIslandSolver(workspace, client=FakeClient(self.round_detail, self))
            result = solver.sync_completed_analyses()
            self.assertEqual(1, result.synced_rounds)
            self.assertEqual(5, result.synced_seeds)
            store = RunStore.for_round(workspace, self.round_detail["id"])
            self.assertTrue((store.root / "round.json").exists())
            self.assertTrue((store.root / "round_detail.json").exists())
            for seed_index in range(5):
                self.assertTrue((store.root / f"analysis_seed_{seed_index}.json").exists())


if __name__ == "__main__":
    unittest.main()
