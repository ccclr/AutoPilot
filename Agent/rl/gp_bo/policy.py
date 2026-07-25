"""
Gaussian Process Bayesian Optimization policy with a CMAB-compatible API.

Action space remains the discrete arm catalog (so CMABTrainer / ArmCatalog are
unchanged). Arms are embedded as continuous feature vectors; a GP models
reward as f(context, arm), and the next arm is chosen by maximizing UCB
(mean + kappa * std) over all arms — exact BO on a small discrete set.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import Iterable, Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

logger = logging.getLogger(__name__)


class GPBOPolicy:
    def __init__(
        self,
        arms: Iterable[str],
        feature_dim: int,
        policy_name: str = "gp_bo",
        kappa: float = 2.0,
        min_samples_to_fit: int = 3,
        fit_every: int = 1,
        uses_context: bool = True,
        random_state: int = 0,
        replay_window: int = 200,
        n_restarts_optimizer: int = 2,
    ):
        self._arms = list(arms)
        self.policy_name = policy_name
        self.kappa = float(kappa)
        self._min_samples_to_fit = int(min_samples_to_fit)
        self._fit_every = max(1, int(fit_every))
        self._uses_context = bool(uses_context)
        self.uses_context = self._uses_context
        self._random_state = int(random_state)
        self._replay_window = max(1, int(replay_window))
        self._feature_dim = int(feature_dim)
        self._n_restarts_optimizer = int(n_restarts_optimizer)

        # GP is constructed on first fit once the true input dimension is known.
        self._gp: Optional[GaussianProcessRegressor] = None
        self._is_fitted = False
        self._update_count = 0
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self.arm_counts = {arm: 0 for arm in self._arms}
        self._recent_decisions: deque = deque(maxlen=self._replay_window)
        self._monitor_topk = 5
        self._input_dim: Optional[int] = None

    def _stable_int(self, *parts) -> int:
        payload = "|".join(str(p) for p in parts).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _shared_rng_index(self, size: int, seed_hex: str | None, label: str) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        digest = hashlib.sha256(f"{seed_hex}|{label}".encode("utf-8")).digest()
        seed_u64 = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed_u64)
        return int(rng.integers(size))

    def _arm_to_vector(self, arm) -> np.ndarray:
        if isinstance(arm, str) and "=" in arm:
            values = []
            for part in arm.split(","):
                _, value = part.split("=", 1)
                values.append(float(value))
            return np.asarray(values, dtype=np.float32)
        return np.asarray(arm, dtype=np.float32).flatten()

    def _normalize_arm_vector(self, arm_vec: np.ndarray) -> np.ndarray:
        """Log-scale large magnitudes then min-max normalize using catalog extremes."""
        arm_mat = np.stack([self._arm_to_vector(a) for a in self._arms], axis=0)
        # Prefer log1p for heavy-tailed params like batch_size.
        logged = np.log1p(np.maximum(arm_mat, 0.0))
        lo = logged.min(axis=0)
        hi = logged.max(axis=0)
        denom = np.where(hi > lo, hi - lo, 1.0)
        logged_arm = np.log1p(np.maximum(arm_vec.astype(np.float64), 0.0))
        return ((logged_arm - lo) / denom).astype(np.float32)

    def _feature_row(self, context, arm) -> np.ndarray:
        arm_vec = self._normalize_arm_vector(self._arm_to_vector(arm))
        if self._uses_context and context is not None:
            ctx_vec = np.asarray(context, dtype=np.float32).flatten()
            return np.concatenate([ctx_vec, arm_vec])
        return arm_vec

    def _feature_matrix(self, context) -> np.ndarray:
        return np.asarray(
            [self._feature_row(context, arm) for arm in self._arms],
            dtype=np.float32,
        )

    def _make_gp(self, n_features: int) -> GaussianProcessRegressor:
        length_scale = np.ones(n_features, dtype=np.float64)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self._n_restarts_optimizer,
            random_state=self._random_state,
            alpha=1e-6,
        )

    def _cold_start_arm(self, shared_seed_hex: str | None) -> str:
        counts = [self.arm_counts[a] for a in self._arms]
        min_count = min(counts) if counts else 0
        candidates = [a for a, c in zip(self._arms, counts) if c == min_count]
        idx = self._shared_rng_index(len(candidates), shared_seed_hex, "gp_bo_cold_start")
        chosen = candidates[idx]
        logger.info(
            "GP-BO cold start: selecting least-tried arm idx=%d arm=%s (min_count=%d)",
            idx,
            chosen,
            min_count,
        )
        return chosen

    def select_arm(self, context, shared_seed_hex: str | None = None):
        if (
            not self._is_fitted
            or self._gp is None
            or len(self._y) < self._min_samples_to_fit
        ):
            return self._cold_start_arm(shared_seed_hex)

        features = self._feature_matrix(context)
        logger.info("GPBO_INFERENCE_START arms=%d", len(self._arms))
        try:
            mean, std = self._gp.predict(features, return_std=True)
        except Exception as e:
            logger.warning("GP predict failed (%s); falling back to cold start", e)
            return self._cold_start_arm(shared_seed_hex)
        logger.info("GPBO_INFERENCE_DONE")

        std = np.maximum(np.asarray(std, dtype=np.float64), 1e-9)
        mean = np.asarray(mean, dtype=np.float64)
        ucb = mean + self.kappa * std

        topk = min(self._monitor_topk, len(self._arms))
        top_ucb_idx = np.argsort(ucb)[::-1][:topk]
        logger.info("================================================================================")
        logger.info("GP-BO SELECT ARM (UCB)")
        logger.info("Context: %s", np.asarray(context))
        logger.info("kappa=%.3f samples=%d", self.kappa, len(self._y))
        for rank, idx in enumerate(top_ucb_idx, start=1):
            logger.info(
                "  #%d: arm=%s mean=%.6f std=%.6f ucb=%.6f",
                rank,
                self._arms[idx],
                mean[idx],
                std[idx],
                ucb[idx],
            )

        max_ucb = ucb.max()
        max_indices = np.flatnonzero(np.isclose(ucb, max_ucb))
        if len(max_indices) > 1 and shared_seed_hex:
            pick = self._shared_rng_index(len(max_indices), shared_seed_hex, "gp_bo_tie")
            chosen_idx = int(max_indices[pick])
        else:
            chosen_idx = int(max_indices.min())
        chosen = self._arms[chosen_idx]
        logger.info(
            "✓ SELECTED ARM: %s (ucb=%.6f mean=%.6f std=%.6f)",
            chosen,
            ucb[chosen_idx],
            mean[chosen_idx],
            std[chosen_idx],
        )
        logger.info("================================================================================")
        return chosen

    def update(self, decisions, rewards, contexts=None, shared_seed_hex: str | None = None):
        if contexts is None:
            contexts = [None] * len(decisions)

        for arm, reward, context in zip(decisions, rewards, contexts):
            if float(reward) > 15:
                logger.warning(
                    "DROP_TRAIN_SAMPLE arm=%s reward=%.6f reason=reward_gt_15",
                    arm,
                    float(reward),
                )
                continue
            feature_row = self._feature_row(context, arm)
            self._X.append(np.asarray(feature_row, dtype=np.float32))
            self._y.append(float(reward))
            self.arm_counts[arm] = self.arm_counts.get(arm, 0) + 1
            self._recent_decisions.append(arm)
            logger.info(
                "TRAIN_SAMPLE idx=%d arm=%s reward=%.6f context=%s feature=%s",
                len(self._y),
                arm,
                float(reward),
                np.asarray(context) if context is not None else None,
                np.asarray(feature_row),
            )

        self._update_count += 1
        if len(self._y) < self._min_samples_to_fit:
            logger.info(
                "GP-BO warmup samples=%d/%d; skip fit",
                len(self._y),
                self._min_samples_to_fit,
            )
            return
        if self._update_count % self._fit_every != 0:
            return

        replay_length = min(len(self._y), self._replay_window)
        X = np.asarray(self._X[-replay_length:], dtype=np.float64)
        y = np.asarray(self._y[-replay_length:], dtype=np.float64)
        self._input_dim = X.shape[1]
        self._gp = self._make_gp(X.shape[1])
        try:
            self._gp.fit(X, y)
            self._is_fitted = True
            pred = self._gp.predict(X)
            mse = float(np.mean((pred - y) ** 2))
            logger.info(
                "GP Updated: samples=%d replay=%d MSE=%.6f kernel=%s",
                len(self._y),
                replay_length,
                mse,
                self._gp.kernel_,
            )
        except Exception as e:
            self._is_fitted = False
            logger.warning("GP fit failed: %s", e)

    def save(self, path):
        import joblib

        joblib.dump(
            {
                "algo": "gp_bo",
                "gp": self._gp,
                "X": self._X,
                "y": self._y,
                "is_fitted": self._is_fitted,
                "update_count": self._update_count,
                "arm_counts": self.arm_counts,
                "kappa": self.kappa,
                "input_dim": self._input_dim,
            },
            path,
        )

    def load(self, path):
        import joblib

        data = joblib.load(path)
        if data.get("algo") not in (None, "gp_bo"):
            logger.warning(
                "Loading checkpoint with algo=%s into GPBOPolicy", data.get("algo")
            )
        self._gp = data.get("gp")
        self._X = data.get("X", [])
        self._y = data.get("y", [])
        self._is_fitted = bool(data.get("is_fitted", False))
        self._update_count = int(data.get("update_count", 0))
        saved_counts = data.get("arm_counts") or {}
        for arm in self._arms:
            self.arm_counts[arm] = int(saved_counts.get(arm, 0))
        if "kappa" in data:
            self.kappa = float(data["kappa"])
        self._input_dim = data.get("input_dim")
        if self._gp is not None and self._X:
            self._is_fitted = True
