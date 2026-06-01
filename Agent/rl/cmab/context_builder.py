from __future__ import annotations

from typing import Literal

import numpy as np

from actions.state_encode import parse_metrics_with_context


ContextMode = Literal["dynamic", "full"]


class ContextBuilder:
    def __init__(self, mode: ContextMode = "dynamic"):
        if mode not in ("dynamic", "full"):
            raise ValueError(f"Unsupported context mode: {mode}")
        self.mode = mode

    def build_context(self, metrics_path: str) -> np.ndarray:
        context, dynamic_state, _, _ = parse_metrics_with_context(metrics_path)
        if self.mode == "dynamic":
            return np.asarray(dynamic_state, dtype=np.float32)
        return np.concatenate([context, dynamic_state]).astype(np.float32)

