#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import threading
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from .reward_change_detector import RewardChangeDetector, RewardChangeResult
    from .experience_matcher import ABExperienceMatcher
except ImportError:
    # Support direct execution with this file's directory on sys.path.
    from reward_change_detector import RewardChangeDetector, RewardChangeResult
    from experience_matcher import ABExperienceMatcher


logger = logging.getLogger(__name__)


class RewardChangeMonitor:
    """Process global reward files in epoch order independently of training."""

    def __init__(
        self,
        metrics_dir: str,
        node_index: Optional[int] = None,
        poll_interval: float = 0.2,
        detector: Optional[RewardChangeDetector] = None,
        experience_matcher: Optional[ABExperienceMatcher] = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.node_index = node_index
        self.poll_interval = poll_interval
        self.detector = detector or RewardChangeDetector()
        self.experience_matcher = experience_matcher
        reward_count = experience_matcher.reward_count if experience_matcher else 3
        self._recent_valid_rewards: deque[float] = deque(maxlen=reward_count)
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
            try:
                value = float(reward)
            except (TypeError, ValueError):
                value = None
            if value is not None and math.isfinite(value) and value > 0:
                self._recent_valid_rewards.append(value)
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
                self._log_experience_match(epoch)

        return processed

    def _log_experience_match(self, epoch: int) -> None:
        matcher = self.experience_matcher
        if matcher is None:
            return
        try:
            match = matcher.match(self._recent_valid_rewards)
        except ValueError as error:
            logger.error(
                "EXPERIENCE_MATCH_SKIPPED node=%s epoch=%s reason=%s bucket_modified=False",
                self.node_index,
                epoch,
                error,
            )
            return

        logger.warning(
            "EXPERIENCE_MATCH_RESULT node=%s epoch=%s recent_rewards=%s "
            "query_mean=%.6f A_mean=%.6f A_distance=%.6f "
            "B_mean=%.6f B_distance=%.6f matched_pool=%s bucket_modified=False",
            self.node_index,
            epoch,
            [round(reward, 6) for reward in match.recent_rewards],
            match.query_mean,
            match.a_mean,
            match.a_distance,
            match.b_mean,
            match.b_distance,
            match.matched_pool,
        )

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
    parser.add_argument("--experience-checkpoint-a", type=str, default=None)
    parser.add_argument("--experience-checkpoint-b", type=str, default=None)
    parser.add_argument("--experience-pool-size", type=int, default=200)
    parser.add_argument("--experience-match-reward-count", type=int, default=3)
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
    if bool(args.experience_checkpoint_a) != bool(args.experience_checkpoint_b):
        parser.error(
            "--experience-checkpoint-a and --experience-checkpoint-b must be provided together"
        )
    experience_matcher = None
    if args.experience_checkpoint_a and args.experience_checkpoint_b:
        experience_matcher = ABExperienceMatcher(
            checkpoint_a=args.experience_checkpoint_a,
            checkpoint_b=args.experience_checkpoint_b,
            pool_size=args.experience_pool_size,
            reward_count=args.experience_match_reward_count,
        )
        for pool in experience_matcher.pools:
            logger.info(
                "EXPERIENCE_POOL_LOADED node=%s pool=%s path=%s total_samples=%d "
                "used_samples=%d reward_mean=%.6f read_only=True",
                args.node_index,
                pool.label,
                pool.path,
                pool.total_samples,
                pool.used_samples,
                pool.reward_mean,
            )
    monitor = RewardChangeMonitor(
        metrics_dir=args.metrics_dir,
        node_index=args.node_index,
        poll_interval=args.poll_interval,
        detector=detector,
        experience_matcher=experience_matcher,
    )

    def request_stop(_signal_number, _frame) -> None:
        monitor.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    monitor.run_forever()


if __name__ == "__main__":
    main()
