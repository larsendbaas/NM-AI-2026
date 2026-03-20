import json
import unittest
from pathlib import Path

from astar_island.backtest import discover_completed_rounds, run_backtest


ROOT = Path(__file__).resolve().parents[1]


class BacktestTests(unittest.TestCase):
    def test_completed_rounds_backtest(self) -> None:
        rounds = discover_completed_rounds(ROOT)
        self.assertTrue(rounds)

        for _, round_id in rounds:
            result = run_backtest(ROOT, round_id)
            print(json.dumps(result, sort_keys=True))
            self.assertGreater(result["recomputed_score"], 50.0)


if __name__ == "__main__":
    unittest.main()
