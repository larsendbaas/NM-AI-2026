import math
import unittest

from astar_island.scoring import score_prediction, weighted_kl


class ScoringTests(unittest.TestCase):
    def test_perfect_prediction_scores_100(self) -> None:
        ground_truth = [[[0.2, 0.3, 0.1, 0.1, 0.2, 0.1]]]
        prediction = [[[0.2, 0.3, 0.1, 0.1, 0.2, 0.1]]]
        self.assertAlmostEqual(0.0, weighted_kl(ground_truth, prediction), places=9)
        self.assertAlmostEqual(100.0, score_prediction(ground_truth, prediction), places=9)

    def test_static_cells_do_not_penalize_score(self) -> None:
        ground_truth = [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        prediction = [[[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]]]
        self.assertAlmostEqual(0.0, weighted_kl(ground_truth, prediction), places=9)
        self.assertAlmostEqual(100.0, score_prediction(ground_truth, prediction), places=9)

    def test_worse_prediction_scores_lower(self) -> None:
        ground_truth = [[[0.0, 0.6, 0.2, 0.2, 0.0, 0.0]]]
        better = [[[0.05, 0.5, 0.2, 0.2, 0.03, 0.02]]]
        worse = [[[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]]]
        self.assertLess(weighted_kl(ground_truth, better), weighted_kl(ground_truth, worse))
        self.assertGreater(score_prediction(ground_truth, better), score_prediction(ground_truth, worse))


if __name__ == "__main__":
    unittest.main()
