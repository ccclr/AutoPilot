from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from coverage_round_robin import (
    CoverageRoundRobinTrainer,
    ShuffledRoundRobinSchedule,
)
from offline_dataset import TransitionDatasetWriter


ARMS = (
    "batch_size=100000,header_size=32,cut_condition_type=2,"
    "fast_path_timeout=0,k=1",
    "batch_size=500000,header_size=64,cut_condition_type=3,"
    "fast_path_timeout=100,k=4",
    "batch_size=1000000,header_size=32,cut_condition_type=1,"
    "fast_path_timeout=300,k=1",
)


class ScheduleTests(unittest.TestCase):
    def test_each_seeded_cycle_contains_every_action_once(self) -> None:
        first = ShuffledRoundRobinSchedule(action_count=72, seed=17)
        second = ShuffledRoundRobinSchedule(action_count=72, seed=17)

        first_cycle = []
        second_cycle = []
        for _ in range(72):
            first_cycle.append(first.current_action_id)
            second_cycle.append(second.current_action_id)
            first.advance()
            second.advance()

        self.assertEqual(first_cycle, second_cycle)
        self.assertEqual(set(first_cycle), set(range(72)))
        self.assertEqual(len(first_cycle), len(set(first_cycle)))
        self.assertEqual(first.cycle, 1)


class _FakeBroadcaster:
    def __init__(self) -> None:
        self.calls = []

    def broadcast(self, **kwargs):
        self.calls.append(kwargs)
        success = len(self.calls) != 1
        return SimpleNamespace(
            success=success,
            failed_nodes=[] if success else [2],
        )


class CoverageTrainerTests(unittest.TestCase):
    @staticmethod
    def _write_state(root: Path, epoch: int, reward: float) -> Path:
        path = root / f"global_state_epoch_{epoch}.json"
        path.write_text(
            json.dumps(
                {
                    "state_4_lane_vector": {
                        "growth_rates": {
                            "lane-0": 10.0 + epoch,
                            "lane-1": 20.0 + epoch,
                            "lane-2": 30.0 + epoch,
                            "lane-3": 40.0 + epoch,
                        }
                    },
                    "global_fast_path_ratio": 0.5,
                    "global_reward": reward,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_failed_broadcast_retries_same_action_before_advancing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            epoch = 987650
            paths = [
                self._write_state(root, epoch + offset, 1.0 + offset / 10)
                for offset in range(4)
            ]
            writer = TransitionDatasetWriter(
                root_dir=root / "offline",
                environment="A",
                run_id="coverage-run",
                arms=ARMS,
                node_index=0,
                behavior_policy="coverage_round_robin",
            )
            transition_path = writer.transition_path
            manifest_path = writer.manifest_path
            broadcaster = _FakeBroadcaster()
            trainer = CoverageRoundRobinTrainer(
                metrics_dir=str(root),
                arms=ARMS,
                decode_arm=lambda arm: {"arm": arm},
                broadcaster=broadcaster,
                transition_writer=writer,
                seed=11,
            )
            following = iter(paths[1:])
            trainer._get_latest_metrics_file = lambda: paths[0]
            trainer._wait_for_new_metrics_file = (
                lambda _last, _timeout: next(following)
            )

            trainer.run(num_transitions=2)

            action_ids = [call["action_id"] for call in broadcaster.calls]
            self.assertEqual(len(action_ids), 3)
            self.assertEqual(action_ids[0], action_ids[1])
            self.assertNotEqual(action_ids[1], action_ids[2])

            records = [
                json.loads(line)
                for line in transition_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(
                [record["source_epoch"] for record in records],
                [epoch + 1, epoch + 2],
            )
            self.assertEqual(
                {record["behavior_policy"] for record in records},
                {"coverage_round_robin"},
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["behavior_policy"], "coverage_round_robin"
            )


if __name__ == "__main__":
    unittest.main()
