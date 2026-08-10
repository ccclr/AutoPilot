from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import joblib


@dataclass(frozen=True)
class ExperiencePoolSummary:
    label: str
    path: Path
    total_samples: int
    used_samples: int
    reward_mean: float


@dataclass(frozen=True)
class ExperienceMatchResult:
    recent_rewards: Tuple[float, ...]
    query_mean: float
    a_mean: float
    b_mean: float
    a_distance: float
    b_distance: float
    matched_pool: str


class ABExperienceMatcher:
    """Match a confirmed reward change to one of two read-only checkpoints."""

    def __init__(
        self,
        checkpoint_a: str,
        checkpoint_b: str,
        pool_size: int = 200,
        reward_count: int = 3,
    ) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if reward_count <= 0:
            raise ValueError("reward_count must be positive")

        self.pool_size = int(pool_size)
        self.reward_count = int(reward_count)
        self.pool_a = self._load_pool("A", checkpoint_a)
        self.pool_b = self._load_pool("B", checkpoint_b)

    def _load_pool(self, label: str, checkpoint_path: str) -> ExperiencePoolSummary:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"experience checkpoint {label} does not exist: {path}")

        checkpoint = joblib.load(path)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"experience checkpoint {label} is not a dictionary: {path}")
        if "X" not in checkpoint or "y" not in checkpoint:
            raise ValueError(f"experience checkpoint {label} is missing X/y: {path}")

        features = checkpoint["X"]
        rewards = checkpoint["y"]
        if len(features) != len(rewards):
            raise ValueError(
                f"experience checkpoint {label} has mismatched X/y lengths: "
                f"{len(features)} != {len(rewards)}"
            )
        if len(rewards) == 0:
            raise ValueError(f"experience checkpoint {label} has no rewards: {path}")

        recent_rewards = rewards[-self.pool_size :]
        valid_rewards = []
        for reward in recent_rewards:
            value = float(reward)
            if math.isfinite(value) and value > 0:
                valid_rewards.append(value)
        if not valid_rewards:
            raise ValueError(
                f"experience checkpoint {label} has no valid rewards in its "
                f"last {self.pool_size} samples: {path}"
            )

        reward_mean = math.fsum(valid_rewards) / len(valid_rewards)
        return ExperiencePoolSummary(
            label=label,
            path=path,
            total_samples=len(rewards),
            used_samples=len(valid_rewards),
            reward_mean=reward_mean,
        )

    @property
    def pools(self) -> Tuple[ExperiencePoolSummary, ExperiencePoolSummary]:
        return self.pool_a, self.pool_b

    def match(self, recent_rewards: Iterable[float]) -> ExperienceMatchResult:
        values = tuple(float(reward) for reward in recent_rewards)
        if len(values) < self.reward_count:
            raise ValueError(
                f"need at least {self.reward_count} rewards for matching, got {len(values)}"
            )
        values = values[-self.reward_count :]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("matching rewards must be finite and positive")

        query_mean = math.fsum(values) / len(values)
        a_distance = abs(query_mean - self.pool_a.reward_mean)
        b_distance = abs(query_mean - self.pool_b.reward_mean)
        # Ties are deterministic so every controller makes the same decision.
        matched_pool = "A" if a_distance <= b_distance else "B"
        return ExperienceMatchResult(
            recent_rewards=values,
            query_mean=query_mean,
            a_mean=self.pool_a.reward_mean,
            b_mean=self.pool_b.reward_mean,
            a_distance=a_distance,
            b_distance=b_distance,
            matched_pool=matched_pool,
        )
