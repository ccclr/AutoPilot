from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from actions.state_encode import build_dqn_state
from controllers.action_transport import ActionBroadcaster
from offline_dataset import TransitionDatasetWriter


logger = logging.getLogger(__name__)


class ShuffledRoundRobinSchedule:
    """Visit every action once per reproducibly shuffled cycle."""

    def __init__(self, action_count: int, seed: int = 0) -> None:
        if action_count <= 0:
            raise ValueError("coverage schedule requires at least one action")
        if seed < 0:
            raise ValueError("coverage seed must be non-negative")
        self.action_count = int(action_count)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.cycle = -1
        self.position = 0
        self.order: list[int] = []
        self._start_cycle()

    @property
    def current_action_id(self) -> int:
        return self.order[self.position]

    def advance(self) -> None:
        self.position += 1
        if self.position >= len(self.order):
            self._start_cycle()

    def _start_cycle(self) -> None:
        self.cycle += 1
        self.position = 0
        self.order = list(range(self.action_count))
        self._rng.shuffle(self.order)
        logger.info(
            "COVERAGE_CYCLE_STARTED cycle=%d seed=%d order=%s",
            self.cycle,
            self.seed,
            self.order,
        )


class CoverageRoundRobinTrainer:
    """Collect strict transitions while cycling through the action catalog."""

    def __init__(
        self,
        *,
        metrics_dir: str,
        arms: Sequence[str],
        decode_arm,
        broadcaster: ActionBroadcaster,
        transition_writer: TransitionDatasetWriter,
        seed: int = 0,
        metrics_timeout: int = 300,
    ) -> None:
        if not arms:
            raise ValueError("coverage trainer requires a non-empty action catalog")
        if metrics_timeout <= 0:
            raise ValueError("metrics timeout must be positive")
        self.metrics_dir = Path(metrics_dir)
        self.arms = tuple(str(arm) for arm in arms)
        self.decode_arm = decode_arm
        self.broadcaster = broadcaster
        self.transition_writer = transition_writer
        self.schedule = ShuffledRoundRobinSchedule(len(self.arms), seed=seed)
        self.metrics_timeout = int(metrics_timeout)
        self.collection_active = True

    def run(self, num_transitions: int | None = None) -> None:
        if num_transitions is not None and num_transitions <= 0:
            raise ValueError("number of coverage transitions must be positive")

        stored = 0
        attempts = 0
        last_metrics = self._get_latest_metrics_file()
        if last_metrics is None:
            last_metrics = self._wait_for_first_available_metrics_file(
                self.metrics_timeout
            )

        try:
            while self.collection_active:
                if num_transitions is not None and stored >= num_transitions:
                    logger.info(
                        "COVERAGE_TARGET_REACHED stored=%d attempts=%d",
                        stored,
                        attempts,
                    )
                    break

                current_epoch = self._get_epoch(last_metrics)
                if current_epoch is None:
                    logger.warning(
                        "COVERAGE_INVALID_METRICS_FILE path=%s", last_metrics
                    )
                    next_metrics = self._wait_for_new_metrics_file(
                        last_metrics, self.metrics_timeout
                    )
                    if next_metrics is not None:
                        last_metrics = next_metrics
                    continue

                state = self._build_state(last_metrics)
                action_id = self.schedule.current_action_id
                arm = self.arms[action_id]
                params = self.decode_arm(arm)
                attempts += 1
                decision_id = (
                    f"coverage-cycle-{self.schedule.cycle}-position-"
                    f"{self.schedule.position}-epoch-{current_epoch}-"
                    f"attempt-{attempts}-action-{action_id}"
                )
                result = self.broadcaster.broadcast(
                    decision_id=decision_id,
                    signal_epoch=current_epoch,
                    action_id=action_id,
                    arm=arm,
                    params=params,
                )
                logger.info(
                    "COVERAGE_DECISION epoch=%d cycle=%d position=%d "
                    "action_id=%d arm=%s broadcast_success=%s",
                    current_epoch,
                    self.schedule.cycle,
                    self.schedule.position,
                    action_id,
                    arm,
                    result.success,
                )

                next_metrics = self._wait_for_new_metrics_file(
                    last_metrics, self.metrics_timeout
                )
                if next_metrics is None:
                    logger.warning(
                        "COVERAGE_METRICS_TIMEOUT source_epoch=%d", current_epoch
                    )
                    break

                next_epoch = self._get_epoch(next_metrics)
                reward = self._extract_reward(next_metrics)
                next_state = self._build_state(next_metrics)
                contiguous = next_epoch == current_epoch + 1
                local_abandon = Path(
                    f"/tmp/autopilot_rl_param_abandon_{current_epoch}.signal"
                ).exists()
                valid_reward = math.isfinite(reward) and 0 < reward <= 15
                accepted = (
                    result.success
                    and contiguous
                    and not local_abandon
                    and valid_reward
                )

                if accepted:
                    exported = self.transition_writer.write(
                        source_epoch=current_epoch,
                        reward_epoch=next_epoch,
                        state=state,
                        arm=arm,
                        reward=reward,
                        next_state=next_state,
                        done=False,
                        truncated=False,
                    )
                    if exported:
                        stored += 1
                        logger.info(
                            "COVERAGE_TRANSITION_STORED source_epoch=%d "
                            "reward_epoch=%d cycle=%d position=%d action_id=%d "
                            "reward=%.6f stored=%d",
                            current_epoch,
                            next_epoch,
                            self.schedule.cycle,
                            self.schedule.position,
                            action_id,
                            reward,
                            stored,
                        )
                        self.schedule.advance()
                else:
                    logger.warning(
                        "COVERAGE_TRANSITION_DROPPED source_epoch=%s "
                        "reward_epoch=%s action_id=%d broadcast_success=%s "
                        "failed_nodes=%s contiguous=%s local_abandon=%s "
                        "reward=%.6f valid_reward=%s retry_same_action=True",
                        current_epoch,
                        next_epoch,
                        action_id,
                        result.success,
                        result.failed_nodes,
                        contiguous,
                        local_abandon,
                        reward,
                        valid_reward,
                    )
                last_metrics = next_metrics
        finally:
            self.transition_writer.close()

    def stop(self) -> None:
        self.collection_active = False

    def _get_latest_metrics_file(self) -> Path | None:
        candidates = [
            (epoch, path)
            for path in self.metrics_dir.glob("global_state_epoch_*.json")
            if (epoch := self._get_epoch(path)) is not None
        ]
        return max(candidates, default=(None, None), key=lambda item: item[0])[1]

    def _wait_for_first_available_metrics_file(self, timeout: int) -> Path:
        start = time.time()
        while time.time() - start < timeout:
            latest = self._get_latest_metrics_file()
            if latest is not None:
                return latest
            time.sleep(0.1)
        raise TimeoutError("timeout waiting for the first coverage global state")

    def _wait_for_new_metrics_file(
        self, last_metrics: Path, timeout: int
    ) -> Path | None:
        current_epoch = self._get_epoch(last_metrics)
        if current_epoch is None:
            return None
        start = time.time()
        while time.time() - start < timeout:
            candidates = []
            for path in self.metrics_dir.glob("global_state_epoch_*.json"):
                epoch = self._get_epoch(path)
                if epoch is not None and epoch > current_epoch:
                    candidates.append((epoch, path))
            if candidates:
                newest_epoch, newest_path = max(candidates, key=lambda item: item[0])
                if newest_epoch > current_epoch + 1:
                    logger.warning(
                        "COVERAGE_EPOCH_GAP current=%d newest=%d",
                        current_epoch,
                        newest_epoch,
                    )
                return newest_path
            time.sleep(0.1)
        return None

    @staticmethod
    def _get_epoch(path: Path) -> int | None:
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError, AttributeError):
            return None

    @staticmethod
    def _load_json(path: Path) -> dict:
        for attempt in range(5):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    raw = handle.read()
                if not raw.strip():
                    raise json.JSONDecodeError("empty file", raw, 0)
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("global state root must be an object")
                return data
            except json.JSONDecodeError:
                if attempt == 4:
                    raise
                time.sleep(0.1)
        raise ValueError(f"failed to read {path}")

    def _build_state(self, path: Path) -> np.ndarray:
        return build_dqn_state(self._load_json(path))

    def _extract_reward(self, path: Path) -> float:
        try:
            return float(self._load_json(path).get("global_reward", 0.0))
        except (TypeError, ValueError):
            return 0.0
