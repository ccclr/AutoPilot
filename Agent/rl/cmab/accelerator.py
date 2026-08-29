"""
Training accelerator: prune the action space with golden rules.

Fast-path timeout
-----------------
After a leader has 2f+1 votes it still waits Δ to gather 3f+1.
  timeout ≥ Δ → fast path;  timeout < Δ → slow path.
The smallest catalog timeout strictly above Δ is the covering value
(e.g. cap=110 → 200). Timeouts larger than that covering value can be
dropped; smaller ones are kept so a feasible fast path remains.

Only the master (node 0) probes the ICMP RTT full matrix.
It publishes a hint {timeout_cap, detect_epoch, apply_epoch} to every node.
Followers only read the hint. The cap is applied at apply_epoch
(detect_epoch + apply_delay), e.g. probe at epoch 100, apply at 105.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_TIMEOUT_KEY = "fast_path_timeout"
_BENCH_DIR = Path(__file__).resolve().parents[3] / "benchmark"
HINT_NAME = ".accelerator.json"
APPLY_DELAY_EPOCHS = 5


def _arm_timeout(arm: str) -> float:
    for part in arm.split(","):
        key, _, value = part.partition("=")
        if key.strip() == _TIMEOUT_KEY:
            try:
                return float(value)
            except ValueError:
                return 0.0
    return 0.0


def max_fast_path_delta_ms(matrix: np.ndarray) -> Optional[float]:
    """Δ_i = last vote - (2f+1)-th vote, using ICMP RTT (ms).

    A vote arrives after propose-out + vote-back, so the delay is the ping RTT,
    not RTT/2.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        return None
    n = int(matrix.shape[0])
    f = (n - 1) // 3
    need = 3 * f + 1
    deltas: list[float] = []
    for i in range(n):
        delays: list[float] = []
        for j in range(n):
            rtt = matrix[i, j]
            if j == i:
                delays.append(0.0)
            elif np.isfinite(rtt):
                delays.append(float(rtt))
        if len(delays) < need:
            continue
        delays.sort()
        deltas.append(delays[-1] - delays[2 * f])
    if not deltas:
        return None
    return float(max(deltas))


def _fabfile():
    os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")
    import site
    for p in (site.getusersitepackages(), "/usr/lib/python3/dist-packages", str(_BENCH_DIR)):
        if p and p not in sys.path:
            sys.path.append(p)
    import fabfile as mod

    return mod


class TrainingAccelerator:
    def __init__(
        self,
        period: int = 100,
        apply_delay: int = APPLY_DELAY_EPOCHS,
        is_master: bool = False,
        hint_path: Optional[str | Path] = None,
    ):
        self.period = max(1, int(period))
        self.apply_delay = max(1, int(apply_delay))
        self.is_master = bool(is_master)
        self.hint_path = Path(hint_path) if hint_path else Path.home() / HINT_NAME
        self._applied_cap: Optional[float] = None
        self._lock = threading.Lock()
        self._last_detect_epoch: Optional[int] = None
        self._probe_thread: Optional[threading.Thread] = None

    @property
    def timeout_cap(self) -> Optional[float]:
        with self._lock:
            return self._applied_cap

    def on_epoch(self, epoch: Optional[int]) -> None:
        if epoch is None:
            return
        epoch = int(epoch)
        if self.is_master:
            self._maybe_probe(epoch)
        self._apply_hint(epoch)

    def stop(self) -> None:
        thread = self._probe_thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._probe_thread = None

    def _maybe_probe(self, epoch: int) -> None:
        if epoch <= 0 or epoch % self.period != 0:
            return
        if self._last_detect_epoch == epoch:
            return
        thread = self._probe_thread
        if thread is not None and thread.is_alive():
            return
        self._last_detect_epoch = epoch
        self._probe_thread = threading.Thread(
            target=self._probe,
            args=(epoch,),
            name="accelerator-probe",
            daemon=True,
        )
        self._probe_thread.start()
        logger.info(
            "ACCELERATOR master probe started detect_epoch=%d apply_epoch=%d",
            epoch,
            epoch + self.apply_delay,
        )

    def _probe(self, detect_epoch: int) -> None:
        cwd = os.getcwd()
        try:
            os.chdir(_BENCH_DIR)
            fab = _fabfile()
            matrix = fab.collect_latency_matrix(quiet=True)
            cap = max_fast_path_delta_ms(matrix)
            if cap is None:
                logger.warning("ACCELERATOR no valid Δ; skip publish")
                return
            hint = {
                "timeout_cap": cap,
                "detect_epoch": detect_epoch,
                "apply_epoch": detect_epoch + self.apply_delay,
            }
            fab.publish_accelerator_hint(hint, str(self.hint_path))
            logger.info("ACCELERATOR published %s", hint)
        except Exception as e:
            logger.warning("ACCELERATOR probe failed: %s", e)
        finally:
            os.chdir(cwd)

    def _apply_hint(self, epoch: int) -> None:
        hint = self._read_hint()
        if hint is None:
            return
        apply_epoch = hint.get("apply_epoch")
        cap = hint.get("timeout_cap")
        if apply_epoch is None or cap is None:
            return
        try:
            apply_epoch = int(apply_epoch)
            cap = float(cap)
        except (TypeError, ValueError):
            return
        if epoch < apply_epoch:
            return
        with self._lock:
            if self._applied_cap != cap:
                logger.info(
                    "ACCELERATOR apply cap=%s at epoch=%d (scheduled=%d)",
                    cap,
                    epoch,
                    apply_epoch,
                )
            self._applied_cap = cap

    def _read_hint(self) -> Optional[dict]:
        if not self.hint_path.exists():
            return None
        try:
            raw = self.hint_path.read_text(encoding="utf-8").strip()
            if not raw:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def covering_timeout(self, timeouts: Iterable[float]) -> Optional[float]:
        """Smallest action timeout strictly larger than cap, or None if none exists."""
        cap = self.timeout_cap
        if cap is None:
            return None
        covering = sorted(t for t in set(float(x) for x in timeouts) if t > cap)
        return covering[0] if covering else None

    def filter_arms(self, arms: Iterable[str]) -> list[str]:
        """Keep timeouts up to the first catalog value > cap; drop anything larger."""
        arms = list(arms)
        timeouts = [_arm_timeout(a) for a in arms]
        limit = self.covering_timeout(timeouts)
        if limit is None:
            return arms
        kept = [a for a, t in zip(arms, timeouts) if t <= limit]
        if len(kept) < len(arms):
            logger.info(
                "ACCELERATOR prune timeout_cap=%.1f limit=%.1f arms %d -> %d",
                self.timeout_cap,
                limit,
                len(arms),
                len(kept),
            )
        return kept
