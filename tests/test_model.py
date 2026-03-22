import types
import unittest

from astar_island.features import CellFeatures
from astar_island.model import TransitionModel


def make_features() -> CellFeatures:
    return CellFeatures(
        terrain=11,
        terrain_class=0,
        coastal=0,
        ocean_neighbors=0,
        adj_settlement=0,
        adj_port=0,
        adj_ruin=0,
        adj_forest=0,
        adj_mountain=0,
        adj_buildable=0,
        ring2_settlement=0,
        ring2_port=0,
        ring2_ruin=0,
        ring2_forest=0,
        ring2_buildable=0,
        ring3_settlement=0,
        ring3_ruin=0,
        ring3_forest=0,
        dist_settlement=5,
        dist_port=5,
        dist_ruin=5,
        dist_ocean=5,
        dist_mountain=5,
        edge_distance=0,
    )


class ModelTests(unittest.TestCase):
    def test_predict_cell_blends_local_and_observation_corrected_predictions(self) -> None:
        model = TransitionModel(floor=1e-6)
        model.live.direct_counts[(0, 0, 0)] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        def fake_predict_local_distribution(
            self: TransitionModel,
            features: CellFeatures,
            sample_counts: list[float] | None,
        ) -> list[float]:
            if sample_counts:
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        def fake_predict_observation_corrected_distribution(
            self: TransitionModel,
            features: CellFeatures,
            sample_counts: list[float],
        ) -> list[float]:
            return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        model._predict_local_distribution = types.MethodType(fake_predict_local_distribution, model)
        model._predict_observation_corrected_distribution = types.MethodType(
            fake_predict_observation_corrected_distribution,
            model,
        )
        prediction = model.predict_cell(0, 0, 0, make_features())

        self.assertAlmostEqual(0.4, prediction[0], places=6)
        self.assertAlmostEqual(0.6, prediction[1], places=6)
        self.assertAlmostEqual(1.0, sum(prediction), places=6)


if __name__ == "__main__":
    unittest.main()
