"""Shared mixed action-space helpers for RL policies.

The four non-timeout Autobahn parameters are enumerated while
``fast_path_timeout`` remains continuous inside the policy.  The timeout is
rounded only when an action is encoded for the integer-millisecond runtime
configuration.
"""

from __future__ import annotations

import itertools
from typing import Any, List, Sequence, Tuple

from actions.action_encode import ActionCodec

Arm = str

ARM_KEYS = [
    "batch_size",
    "header_size",
    "cut_condition_type",
    "fast_path_timeout",
    "k",
]
TIMEOUT_KEY = "fast_path_timeout"
TIMEOUT_KEY_INDEX = ARM_KEYS.index(TIMEOUT_KEY)


def encode_arm_values(values: Sequence[float | int]) -> Arm:
    parts = []
    for key, value in zip(ARM_KEYS, values):
        if key == TIMEOUT_KEY:
            parts.append(f"{key}={int(round(float(value)))}")
        else:
            parts.append(f"{key}={int(value)}")
    return ",".join(parts)


def decode_arm_values(arm_id: Arm) -> Tuple[float, ...]:
    values = {}
    for part in arm_id.split(","):
        key, value = part.split("=", 1)
        values[key] = float(value)
    return tuple(values[key] for key in ARM_KEYS)


def decode_arm_params(arm_id: Arm) -> dict[str, Any]:
    values = decode_arm_values(arm_id)
    params = dict(zip(ARM_KEYS, values))
    params[TIMEOUT_KEY] = int(round(params[TIMEOUT_KEY]))
    for key in ARM_KEYS:
        if key != TIMEOUT_KEY:
            params[key] = int(params[key])
    return params


class MixedActionSpace:
    """Discrete Autobahn bases multiplied by a continuous timeout interval."""

    def __init__(
        self,
        codec: ActionCodec | None = None,
        timeout_bounds: tuple[float, float] | None = None,
    ):
        self.codec = codec or ActionCodec()
        bounds = timeout_bounds
        if bounds is None:
            bounds = getattr(self.codec, "fast_path_timeout_ms_bounds", None)
        if bounds is None:
            values = self.codec.fast_path_timeout_ms_values
            bounds = (min(values), max(values))
        self.timeout_lo = float(bounds[0])
        self.timeout_hi = float(bounds[1])
        if self.timeout_hi < self.timeout_lo:
            raise ValueError("timeout bounds must be ordered as (lo, hi)")

        discrete_sets = [
            self.codec.batch_size_values,
            self.codec.header_size_values,
            self.codec.cut_condition_type_values,
            self.codec.parallel_proposals_values,
        ]
        self._bases: List[Tuple[int, int, int, int]] = list(
            itertools.product(*discrete_sets)
        )
        self._placeholder_arms = [
            encode_arm_values((b, h, c, self.timeout_lo, k))
            for (b, h, c, k) in self._bases
        ]

    def list_bases(self) -> List[Tuple[int, int, int, int]]:
        return list(self._bases)

    def list_placeholder_arms(self) -> List[Arm]:
        return list(self._placeholder_arms)

    def make_arm(self, base: Tuple[int, int, int, int], timeout_ms: float) -> Arm:
        b, h, c, k = base
        timeout = _clip(timeout_ms, self.timeout_lo, self.timeout_hi)
        return encode_arm_values((b, h, c, timeout, k))

    def normalize_timeout(self, timeout_ms: float) -> float:
        span = self.timeout_hi - self.timeout_lo
        if span <= 0:
            return 0.0
        return (
            _clip(timeout_ms, self.timeout_lo, self.timeout_hi) - self.timeout_lo
        ) / span

    def denormalize_timeout(self, unit: float) -> float:
        u = _clip(unit, 0.0, 1.0)
        return self.timeout_lo + u * (self.timeout_hi - self.timeout_lo)

    def timeout_grid(self, n_grid: int) -> List[float]:
        """Return a grid for policies that intentionally use finite candidates."""
        n = max(2, int(n_grid))
        if self.timeout_hi <= self.timeout_lo:
            return [self.timeout_lo]
        return [
            self.timeout_lo
            + i * (self.timeout_hi - self.timeout_lo) / (n - 1)
            for i in range(n)
        ]


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), lo), hi))


class MixedArmCatalog:
    """CMABTrainer-compatible decoder for mixed continuous actions."""

    def __init__(self, space: MixedActionSpace):
        self.space = space
        self.codec = space.codec

    def list_arms(self) -> List[Arm]:
        return self.space.list_placeholder_arms()

    def decode_arm(self, arm: Arm) -> dict[str, Any]:
        return decode_arm_params(arm)
