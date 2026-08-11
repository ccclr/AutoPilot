"""Continuous-timeout KernelUCB policy for Autobahn tuning."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Optional

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar

from actions.mixed_action_space import MixedActionSpace, decode_arm_values

logger = logging.getLogger(__name__)


class ContinuousKernelUCBPolicy:
    """KernelUCB over discrete Autobahn parameters and continuous timeout.

    ``fast_path_timeout=0`` is evaluated as a separate fast-path-disabled
    action.  Positive timeouts are optimized continuously and rounded only
    when the selected action is emitted to the runtime.
    """

    def __init__(
        self,
        mixed_space: MixedActionSpace,
        policy_name: str = "kernel_ucb",
        ucb_alpha: float = 1.0,
        regularization: float = 0.1,
        length_scale: float = 1.0,
        replay_window: int = 200,
        min_samples_to_fit: int = 1,
        fit_every: int = 1,
        uses_context: bool = True,
        random_state: int = 0,
        positive_timeout_min: float = 1.0,
        optimizer_restarts: int = 5,
    ):
        if ucb_alpha < 0:
            raise ValueError("ucb_alpha must be non-negative")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        if length_scale <= 0:
            raise ValueError("length_scale must be positive")
        if positive_timeout_min <= 0:
            raise ValueError("positive_timeout_min must be positive")
        if positive_timeout_min > mixed_space.timeout_hi:
            raise ValueError("positive_timeout_min cannot exceed timeout upper bound")

        self.mixed_space = mixed_space
        self._bases = mixed_space.list_bases()
        self._arms = mixed_space.list_placeholder_arms()
        self.policy_name = policy_name
        self.ucb_alpha = float(ucb_alpha)
        self.regularization = float(regularization)
        self.length_scale = float(length_scale)
        self._replay_window = max(1, int(replay_window))
        self._min_samples_to_fit = max(1, int(min_samples_to_fit))
        self._fit_every = max(1, int(fit_every))
        self._uses_context = bool(uses_context)
        self.uses_context = self._uses_context
        self._random_state = int(random_state)
        self.positive_timeout_min = float(positive_timeout_min)
        self.optimizer_restarts = max(1, int(optimizer_restarts))

        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._update_count = 0
        self._is_fitted = False
        self._active_X: Optional[np.ndarray] = None
        self._chol: Optional[np.ndarray] = None
        self._dual_weights: Optional[np.ndarray] = None
        self._reward_mean = 0.0
        self._reward_scale = 1.0
        self.arm_counts: dict[str, int] = {}
        self._monitor_topk = 5

        codec = mixed_space.codec
        self._base_lo = np.asarray(
            [
                min(codec.batch_size_values),
                min(codec.header_size_values),
                min(codec.cut_condition_type_values),
                min(codec.parallel_proposals_values),
            ],
            dtype=np.float64,
        )
        self._base_hi = np.asarray(
            [
                max(codec.batch_size_values),
                max(codec.header_size_values),
                max(codec.cut_condition_type_values),
                max(codec.parallel_proposals_values),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _stable_int(*parts) -> int:
        payload = "|".join(str(p) for p in parts).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _shared_index(self, size: int, seed_hex: str | None, label: str) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        return self._stable_int(
            seed_hex if seed_hex is not None else self._random_state,
            label,
            self._update_count,
            len(self._y),
        ) % size

    def _normalize_context(self, context) -> np.ndarray:
        if context is None or not self._uses_context:
            return np.empty(0, dtype=np.float64)
        values = np.asarray(context, dtype=np.float64).flatten().copy()
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        if values.size > 1:
            # Current context is [lane growth rates..., fast_path_ratio].
            values[:-1] = np.clip(values[:-1] / 20.0, 0.0, 1.0)
            values[-1] = np.clip(values[-1], 0.0, 1.0)
        elif values.size == 1:
            values[0] = np.clip(values[0], 0.0, 1.0)
        return values

    def _normalize_base(self, base: tuple[int, int, int, int]) -> np.ndarray:
        values = np.asarray(base, dtype=np.float64)
        denominator = np.where(
            self._base_hi > self._base_lo,
            self._base_hi - self._base_lo,
            1.0,
        )
        return (values - self._base_lo) / denominator

    def _feature_for_base(self, context, base, timeout_ms: float) -> np.ndarray:
        context_vector = self._normalize_context(context)
        base_vector = self._normalize_base(base)
        enabled = 1.0 if float(timeout_ms) > 0.0 else 0.0
        timeout_span = max(self.mixed_space.timeout_hi, 1.0)
        timeout_normalized = np.clip(float(timeout_ms) / timeout_span, 0.0, 1.0)
        action_vector = np.concatenate(
            [base_vector, np.asarray([enabled, timeout_normalized])]
        )
        return np.concatenate([context_vector, action_vector])

    def _feature_row(self, context, arm: str) -> np.ndarray:
        batch, header, cut, timeout, parallel = decode_arm_values(arm)
        base = (int(batch), int(header), int(cut), int(parallel))
        return self._feature_for_base(context, base, timeout)

    def _kernel_matrix(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_sq = np.sum(left * left, axis=1)[:, None]
        right_sq = np.sum(right * right, axis=1)[None, :]
        distance_sq = np.maximum(left_sq + right_sq - 2.0 * left @ right.T, 0.0)
        return np.exp(-0.5 * distance_sq / (self.length_scale ** 2))

    def _fit_active_model(self) -> None:
        replay_length = min(len(self._y), self._replay_window)
        if replay_length < self._min_samples_to_fit:
            self._is_fitted = False
            return

        active_X = np.asarray(self._X[-replay_length:], dtype=np.float64)
        active_y = np.asarray(self._y[-replay_length:], dtype=np.float64)
        self._reward_mean = float(np.mean(active_y))
        # Centering gives unseen actions the observed reward baseline instead
        # of an artificial zero-reward prior.  A scale floor of 1 reward unit
        # keeps the UCB bonus meaningful during initially low-variance runs.
        self._reward_scale = max(float(np.std(active_y)), 1.0)
        normalized_y = (active_y - self._reward_mean) / self._reward_scale
        kernel = self._kernel_matrix(active_X, active_X)
        identity = np.eye(replay_length, dtype=np.float64)
        jitter = 1e-10
        chol = None
        for _ in range(6):
            try:
                chol = np.linalg.cholesky(
                    kernel + (self.regularization + jitter) * identity
                )
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if chol is None:
            raise np.linalg.LinAlgError("KernelUCB kernel matrix is not positive definite")

        lower_solution = solve_triangular(chol, normalized_y, lower=True)
        dual_weights = solve_triangular(chol.T, lower_solution, lower=False)
        self._active_X = active_X
        self._chol = chol
        self._dual_weights = dual_weights
        self._is_fitted = True

        fitted = self._reward_mean + self._reward_scale * (kernel @ dual_weights)
        mse = float(np.mean((fitted - active_y) ** 2))
        logger.info(
            "KERNEL_UCB_UPDATED samples=%d replay=%d mse=%.6f jitter=%.1e",
            len(self._y),
            replay_length,
            mse,
            jitter,
        )

    def _predict_feature(self, feature: np.ndarray) -> tuple[float, float, float]:
        if (
            not self._is_fitted
            or self._active_X is None
            or self._chol is None
            or self._dual_weights is None
        ):
            return 0.0, 1.0, self.ucb_alpha

        row = np.asarray(feature, dtype=np.float64).reshape(1, -1)
        kernel_vector = self._kernel_matrix(self._active_X, row).reshape(-1)
        normalized_mean = float(kernel_vector @ self._dual_weights)
        mean = self._reward_mean + self._reward_scale * normalized_mean
        projected = solve_triangular(self._chol, kernel_vector, lower=True)
        variance = max(1.0 - float(projected @ projected), 1e-12)
        std = self._reward_scale * float(np.sqrt(variance))
        return mean, std, mean + self.ucb_alpha * std

    def _cold_start_arm(self, shared_seed_hex: str | None) -> str:
        base_index = self._shared_index(
            len(self._bases), shared_seed_hex, "kernel_ucb_cold_base"
        )
        first_positive = int(math.ceil(self.positive_timeout_min))
        last_positive = int(math.floor(self.mixed_space.timeout_hi))
        positive_count = max(0, last_positive - first_positive + 1)
        timeout_index = self._shared_index(
            positive_count + 1, shared_seed_hex, "kernel_ucb_cold_timeout"
        )
        timeout_ms = 0 if timeout_index == 0 else first_positive + timeout_index - 1
        base = self._bases[base_index]
        arm = self.mixed_space.make_arm(base, timeout_ms)
        logger.info(
            "KERNEL_UCB_COLD_START base=%s timeout_ms=%d arm=%s samples=%d/%d",
            base,
            timeout_ms,
            arm,
            len(self._y),
            self._min_samples_to_fit,
        )
        return arm

    def _continuous_candidates(self, context) -> list[tuple]:
        candidates = []
        timeout_lo = self.positive_timeout_min
        timeout_hi = self.mixed_space.timeout_hi
        interval_edges = np.linspace(
            timeout_lo, timeout_hi, self.optimizer_restarts + 1
        )

        for base in self._bases:
            disabled_feature = self._feature_for_base(context, base, 0.0)
            disabled_mean, disabled_std, disabled_ucb = self._predict_feature(
                disabled_feature
            )
            candidates.append(
                (disabled_ucb, disabled_mean, disabled_std, base, 0.0)
            )

            def negative_ucb(timeout_ms: float) -> float:
                feature = self._feature_for_base(context, base, timeout_ms)
                return -self._predict_feature(feature)[2]

            positive_timeouts = {float(timeout_lo), float(timeout_hi)}
            for left, right in zip(interval_edges[:-1], interval_edges[1:]):
                if right <= left:
                    continue
                result = minimize_scalar(
                    negative_ucb,
                    bounds=(float(left), float(right)),
                    method="bounded",
                    options={"xatol": 0.05, "maxiter": 40},
                )
                if result.success and np.isfinite(result.x):
                    positive_timeouts.add(float(result.x))

            for timeout_ms in positive_timeouts:
                feature = self._feature_for_base(context, base, timeout_ms)
                mean, std, ucb = self._predict_feature(feature)
                candidates.append((ucb, mean, std, base, timeout_ms))

        return candidates

    def select_arm(self, context, shared_seed_hex: str | None = None):
        if not self._is_fitted or len(self._y) < self._min_samples_to_fit:
            return self._cold_start_arm(shared_seed_hex)

        logger.info(
            "KERNEL_UCB_INFERENCE_START bases=%d timeout=[%.3f,%.3f] restarts=%d",
            len(self._bases),
            self.positive_timeout_min,
            self.mixed_space.timeout_hi,
            self.optimizer_restarts,
        )
        candidates = self._continuous_candidates(context)
        ucb_values = np.asarray([item[0] for item in candidates], dtype=np.float64)
        best_ucb = float(np.max(ucb_values))
        best_indices = np.flatnonzero(np.isclose(ucb_values, best_ucb))
        if len(best_indices) > 1:
            tie_position = self._shared_index(
                len(best_indices), shared_seed_hex, "kernel_ucb_tie"
            )
            best_index = int(best_indices[tie_position])
        else:
            best_index = int(best_indices[0])

        topk = min(self._monitor_topk, len(candidates))
        top_indices = np.argsort(ucb_values)[::-1][:topk]
        for rank, index in enumerate(top_indices, start=1):
            ucb, mean, std, base, timeout = candidates[int(index)]
            logger.info(
                "KERNEL_UCB_TOP rank=%d base=%s timeout_float=%.6f "
                "mean=%.6f std=%.6f ucb=%.6f",
                rank,
                base,
                timeout,
                mean,
                std,
                ucb,
            )

        ucb, mean, std, base, timeout_float = candidates[best_index]
        arm = self.mixed_space.make_arm(base, timeout_float)
        applied_timeout = int(round(timeout_float))
        logger.info(
            "KERNEL_UCB_SELECTED base=%s optimized_timeout_float=%.6f "
            "applied_timeout_ms=%d mean=%.6f std=%.6f ucb=%.6f arm=%s",
            base,
            timeout_float,
            applied_timeout,
            mean,
            std,
            ucb,
            arm,
        )
        return arm

    def update(self, decisions, rewards, contexts=None, shared_seed_hex=None):
        if contexts is None:
            contexts = [None] * len(decisions)

        accepted = 0
        for arm, reward, context in zip(decisions, rewards, contexts):
            reward_value = float(reward)
            if reward_value > 15 or reward_value == 0 or not np.isfinite(reward_value):
                logger.warning(
                    "DROP_TRAIN_SAMPLE arm=%s reward=%s reason=invalid_reward",
                    arm,
                    reward,
                )
                continue
            feature = self._feature_row(context, arm)
            self._X.append(np.asarray(feature, dtype=np.float64))
            self._y.append(reward_value)
            self.arm_counts[arm] = self.arm_counts.get(arm, 0) + 1
            accepted += 1
            logger.info(
                "KERNEL_UCB_TRAIN_SAMPLE idx=%d arm=%s reward=%.6f feature=%s",
                len(self._y),
                arm,
                reward_value,
                feature,
            )

        self._update_count += 1
        if accepted == 0 or self._update_count % self._fit_every != 0:
            return
        self._fit_active_model()

    def save(self, path):
        import joblib

        joblib.dump(
            {
                "algo": "kernel_ucb_continuous_timeout",
                "X": self._X,
                "y": self._y,
                "update_count": self._update_count,
                "arm_counts": self.arm_counts,
                "ucb_alpha": self.ucb_alpha,
                "regularization": self.regularization,
                "length_scale": self.length_scale,
                "replay_window": self._replay_window,
                "positive_timeout_min": self.positive_timeout_min,
                "timeout_bounds": (
                    self.mixed_space.timeout_lo,
                    self.mixed_space.timeout_hi,
                ),
                "optimizer_restarts": self.optimizer_restarts,
            },
            path,
        )

    def load(self, path):
        import joblib

        data = joblib.load(path)
        algo = data.get("algo")
        if algo != "kernel_ucb_continuous_timeout":
            raise ValueError(
                f"Incompatible checkpoint algo={algo!r}; expected continuous KernelUCB"
            )
        saved_bounds = tuple(float(v) for v in data.get("timeout_bounds", ()))
        current_bounds = (
            self.mixed_space.timeout_lo,
            self.mixed_space.timeout_hi,
        )
        if saved_bounds and not np.allclose(saved_bounds, current_bounds):
            raise ValueError(
                f"Checkpoint timeout bounds {saved_bounds} do not match {current_bounds}"
            )
        self._X = [np.asarray(row, dtype=np.float64) for row in data.get("X", [])]
        self._y = [float(value) for value in data.get("y", [])]
        self._update_count = int(data.get("update_count", 0))
        self.arm_counts = dict(data.get("arm_counts") or {})
        self.ucb_alpha = float(data.get("ucb_alpha", self.ucb_alpha))
        self.regularization = float(
            data.get("regularization", self.regularization)
        )
        self.length_scale = float(data.get("length_scale", self.length_scale))
        self.optimizer_restarts = int(
            data.get("optimizer_restarts", self.optimizer_restarts)
        )
        if self._y:
            self._fit_active_model()
