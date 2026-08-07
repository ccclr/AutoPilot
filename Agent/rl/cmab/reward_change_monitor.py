#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from .reward_change_detector import RewardChangeDetector, RewardChangeResult
except ImportError:
    # Support direct execution with this file's directory on sys.path.
    from reward_change_detector import RewardChangeDetector, RewardChangeResult


logger = logging.getLogger(__name__)


class RewardChangeMonitor:
    """Process global reward files in epoch order independently of training."""

    def __init__(
        self,
        metrics_dir: str,
        node_index: Optional[int] = None,
        poll_interval: float = 0.2,
        detector: Optional[RewardChangeDetector] = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.node_index = node_index
        self.poll_interval = poll_interval
        self.detector = detector or RewardChangeDetector()
        self.next_epoch = 0
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def process_available(self) -> List[Tuple[int, RewardChangeResult]]:
        """Process all currently available contiguous epochs."""
        processed: List[Tuple[int, RewardChangeResult]] = []

        while not self._stop_event.is_set():
            metrics_file = self.metrics_dir / f"global_state_epoch_{self.next_epoch}.json"
            if not metrics_file.exists():
                break

            try:
                with metrics_file.open(encoding="utf-8") as file:
                    state = json.load(file)
            except (OSError, json.JSONDecodeError) as error:
                # The primary may still be writing this file. Retry on the next poll.
                logger.debug("Reward state not ready: file=%s error=%s", metrics_file, error)
                break

            epoch = state.get("epoch", self.next_epoch)
            if epoch != self.next_epoch:
                logger.warning(
                    "REWARD_CHANGE_EPOCH_MISMATCH expected=%d actual=%s file=%s",
                    self.next_epoch,
                    epoch,
                    metrics_file,
                )

            reward = state.get("global_reward")
            count_before = self.detector.observation_count
            result = self.detector.observe(reward)
            self.next_epoch += 1

            if result is None:
                if self.detector.observation_count > count_before:
                    logger.info(
                        "REWARD_CHANGE_WARMUP node=%s epoch=%s reward=%.6f "
                        "samples=%d required=%d",
                        self.node_index,
                        epoch,
                        reward,
                        self.detector.observation_count,
                        self.detector.required_observations,
                    )
                else:
                    logger.warning(
                        "REWARD_CHANGE_SKIPPED node=%s epoch=%s invalid_reward=%s",
                        self.node_index,
                        epoch,
                        reward,
                    )
                continue

            processed.append((epoch, result))
            logger.info(
                "REWARD_CHANGE_SCORE node=%s epoch=%s reward=%.6f old_mean=%.6f "
                "new_mean=%.6f score=%.6f threshold=%.6f "
                "threshold_exceeded=%s consecutive=%d/%d detected=%s",
                self.node_index,
                epoch,
                reward,
                result.old_mean,
                result.new_mean,
                result.score,
                result.threshold,
                result.threshold_exceeded,
                result.consecutive_exceedances,
                result.confirmation_count,
                result.detected,
            )
            if result.detected:
                logger.warning(
                    "ENVIRONMENT_CHANGE_DETECTED node=%s epoch=%s score=%.6f "
                    "threshold=%.6f consecutive=%d/%d",
                    self.node_index,
                    epoch,
                    result.score,
                    result.threshold,
                    result.consecutive_exceedances,
                    result.confirmation_count,
                )

        return processed

    def run_forever(self) -> None:
        detector = self.detector
        logger.info(
            "REWARD_CHANGE_MONITOR_STARTED node=%s metrics_dir=%s window=%d lag=%d "
            "threshold=%.3f confirmations=%d poll_interval=%.3f",
            self.node_index,
            self.metrics_dir,
            detector.window_size,
            detector.lag,
            detector.threshold,
            detector.confirmation_count,
            self.poll_interval,
        )
        while not self._stop_event.is_set():
            self.process_available()
            self._stop_event.wait(self.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor global rewards and report confirmed environment changes."
    )
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--node-index", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--lag", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--confirmations", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    detector = RewardChangeDetector(
        window_size=args.window_size,
        lag=args.lag,
        threshold=args.threshold,
        confirmation_count=args.confirmations,
    )
    monitor = RewardChangeMonitor(
        metrics_dir=args.metrics_dir,
        node_index=args.node_index,
        poll_interval=args.poll_interval,
        detector=detector,
    )

    def request_stop(_signal_number, _frame) -> None:
        monitor.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    monitor.run_forever()


if __name__ == "__main__":
    main()
