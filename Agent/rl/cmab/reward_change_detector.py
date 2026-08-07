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
    threshold_exceeded: bool
    consecutive_exceedances: int
    confirmation_count: int
    detected: bool


class RewardChangeDetector:
    """Confirm relative changes between two overlapping reward windows."""

    def __init__(
        self,
        window_size: int = 8,
        lag: int = 3,
        threshold: float = 0.30,
        confirmation_count: int = 3,
        epsilon: float = 1e-12,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if lag <= 0:
            raise ValueError("lag must be positive")
        if threshold < 0 or not math.isfinite(threshold):
            raise ValueError("threshold must be a finite non-negative number")
        if not isinstance(confirmation_count, int) or isinstance(
            confirmation_count, bool
        ):
            raise ValueError("confirmation_count must be an integer")
        if confirmation_count <= 0:
            raise ValueError("confirmation_count must be positive")
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be a finite positive number")

        self.window_size = window_size
        self.lag = lag
        self.threshold = threshold
        self.confirmation_count = confirmation_count
        self.epsilon = epsilon
        self._rewards: deque[float] = deque(maxlen=window_size + lag)
        self._consecutive_exceedances = 0

    @property
    def observation_count(self) -> int:
        return len(self._rewards)

    @property
    def required_observations(self) -> int:
        return self.window_size + self.lag

    @property
    def consecutive_exceedances(self) -> int:
        return self._consecutive_exceedances

    def observe(self, reward: float) -> Optional[RewardChangeResult]:
        """Add one raw reward and return a score once both windows are available."""
        try:
            value = float(reward)
        except (TypeError, ValueError):
            self._consecutive_exceedances = 0
            return None

        if not math.isfinite(value) or value <= 0:
            self._consecutive_exceedances = 0
            return None

        self._rewards.append(value)
        if len(self._rewards) < self.required_observations:
            return None

        values = list(self._rewards)
        old_mean = math.fsum(values[: self.window_size]) / self.window_size
        new_mean = math.fsum(values[self.lag :]) / self.window_size
        score = abs(new_mean - old_mean) / max(abs(old_mean), self.epsilon)
        threshold_exceeded = score > self.threshold
        if threshold_exceeded:
            self._consecutive_exceedances += 1
        else:
            self._consecutive_exceedances = 0

        # Fire only when the run first reaches the required length. Further
        # above-threshold scores remain part of the same change event.
        detected = self._consecutive_exceedances == self.confirmation_count

        return RewardChangeResult(
            old_mean=old_mean,
            new_mean=new_mean,
            score=score,
            threshold=self.threshold,
            threshold_exceeded=threshold_exceeded,
            consecutive_exceedances=self._consecutive_exceedances,
            confirmation_count=self.confirmation_count,
            detected=detected,
        )
