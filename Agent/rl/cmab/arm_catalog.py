from __future__ import annotations

import itertools
import random
from typing import Any, Iterable, List, Mapping, Tuple

from actions.action_encode import ActionCodec

Arm = str


class ArmCatalog:
    def __init__(self, codec: ActionCodec | None = None, max_arms: int | None = None, seed: int = 0):
        self.codec = codec or ActionCodec()
        self.action_dims = list(self.codec.action_dims)
        self._arm_keys = [
            "batch_size",
            "header_size",
            "cut_condition_type",
            "fast_path_timeout",
            "k",
            # "use_optimistic_tips",
        ]
        # use_optimistic_tips: 1=True, 0=False for arm encoding (int required by _decode_arm)
        self._use_optimistic_tips_arm_values = [1, 0]
        self._value_sets = [
            self.codec.batch_size_values,
            self.codec.header_size_values,
            self.codec.cut_condition_type_values,
            self.codec.fast_path_timeout_ms_values,
            self.codec.parallel_proposals_values,
            # self._use_optimistic_tips_arm_values,
        ]
        self._arms = self._build_arms(max_arms=max_arms, seed=seed)
        self._arm_lookup = {arm_id: arm_tuple for arm_id, arm_tuple in self._arms}
        self._catalog_arm_ids = set(self._arm_lookup)

    def _build_arms(self, max_arms: int | None, seed: int) -> List[Tuple[Arm, Tuple[int, ...]]]:
        all_arms = list(itertools.product(*self._value_sets))
        if max_arms is None or max_arms >= len(all_arms):
            return [(self._encode_arm(arm), arm) for arm in all_arms]
        rng = random.Random(seed)
        sampled = rng.sample(all_arms, k=max_arms)
        return [(self._encode_arm(arm), arm) for arm in sampled]

    def _encode_arm(self, arm: Tuple[int, ...]) -> Arm:
        parts = []
        for key, value in zip(self._arm_keys, arm):
            parts.append(f"{key}={int(value)}")
        return ",".join(parts)

    def _decode_arm(self, arm_id: Arm) -> Tuple[int, ...]:
        values = {}
        for part in arm_id.split(","):
            key, value = part.split("=", 1)
            values[key] = int(value)
        return tuple(values[key] for key in self._arm_keys)

    def list_arms(self) -> List[Arm]:
        return list(self._arm_lookup.keys())

    @property
    def arm_keys(self) -> tuple[str, ...]:
        return tuple(self._arm_keys)

    @property
    def timeout_values(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.codec.fast_path_timeout_ms_values)

    @property
    def cut_values(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.codec.cut_condition_type_values)

    def encode_params(self, params: Mapping[str, Any]) -> Arm:
        values = tuple(int(params[key]) for key in self._arm_keys)
        arm = self._encode_arm(values)
        if arm not in self._catalog_arm_ids:
            raise ValueError(f"Parameters are not in the CMAB arm catalog: {params}")
        return arm

    def contains(self, arm: Arm) -> bool:
        return arm in self._catalog_arm_ids

    def structured_initial_arms(self, base_arm: Arm) -> List[Arm]:
        """Return base plus every one-factor alternative in a stable order."""
        if base_arm not in self._catalog_arm_ids:
            raise ValueError(f"Base arm is not in the CMAB arm catalog: {base_arm}")

        base_values = self._arm_lookup[base_arm]
        result = [base_arm]
        for index, value_set in enumerate(self._value_sets):
            for value in value_set:
                if int(value) == int(base_values[index]):
                    continue
                candidate_values = list(base_values)
                candidate_values[index] = int(value)
                candidate = self._encode_arm(tuple(candidate_values))
                if candidate in self._catalog_arm_ids:
                    result.append(candidate)
        return result

    def one_parameter_neighbors(self, arm: Arm) -> List[Arm]:
        if arm not in self._catalog_arm_ids:
            raise ValueError(f"Arm is not in the CMAB arm catalog: {arm}")
        values = self._arm_lookup[arm]
        return [
            candidate
            for candidate, candidate_values in self._arms
            if sum(a != b for a, b in zip(values, candidate_values)) == 1
        ]

    def filter_by_protocol_values(
        self,
        arms: Iterable[Arm],
        timeout_values: Iterable[int],
        cut_values: Iterable[int],
    ) -> List[Arm]:
        allowed_timeouts = {int(value) for value in timeout_values}
        allowed_cuts = {int(value) for value in cut_values}
        result = []
        for arm in arms:
            params = self.decode_arm(arm)
            if (
                params["fast_path_timeout"] in allowed_timeouts
                and params["cut_condition_type"] in allowed_cuts
            ):
                result.append(arm)
        return result

    def decode_arm(self, arm: Arm) -> dict[str, Any]:
        if arm not in self._arm_lookup:
            self._arm_lookup[arm] = self._decode_arm(arm)
        values = self._arm_lookup[arm]
        return dict(zip(self._arm_keys, values))
