from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RewardChangeResult:
    old_mean: float
    new_mean: float
    score: float
    threshold: float
    detected: bool


class RewardChangeDetector:
    """Detect relative changes between two overlapping reward windows."""

    def __init__(
        self,
        window_size: int = 20,
        lag: int = 5,
        threshold: float = 0.20,
        epsilon: float = 1e-12,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if lag <= 0:
            raise ValueError("lag must be positive")
        if threshold < 0 or not math.isfinite(threshold):
            raise ValueError("threshold must be a finite non-negative number")
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be a finite positive number")

        self.window_size = window_size
        self.lag = lag
        self.threshold = threshold
        self.epsilon = epsilon
        self._rewards: deque[float] = deque(maxlen=window_size + lag)

    @property
    def observation_count(self) -> int:
        return len(self._rewards)

    @property
    def required_observations(self) -> int:
        return self.window_size + self.lag

    def observe(self, reward: float) -> Optional[RewardChangeResult]:
        """Add one raw reward and return a score once both windows are available."""
        try:
            value = float(reward)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value) or value <= 0:
            return None

        self._rewards.append(value)
        if len(self._rewards) < self.required_observations:
            return None

        values = list(self._rewards)
        old_mean = math.fsum(values[: self.window_size]) / self.window_size
        new_mean = math.fsum(values[self.lag :]) / self.window_size
        score = abs(new_mean - old_mean) / max(abs(old_mean), self.epsilon)

        return RewardChangeResult(
            old_mean=old_mean,
            new_mean=new_mean,
            score=score,
            threshold=self.threshold,
            detected=score > self.threshold,
        )
