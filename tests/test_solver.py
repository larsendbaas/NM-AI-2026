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
                json.dumps(self.sample),
                encoding="utf-8",
            )
            solver = AstarIslandSolver(workspace)
            store = RunStore.for_round(workspace, self.round_detail["id"])
            imported = solver.import_bootstrap_samples(self.round_detail["id"], store)
            self.assertEqual(1, imported)
            observations = store.load_observations()
            self.assertEqual(1, len(observations))


if __name__ == "__main__":
    unittest.main()
