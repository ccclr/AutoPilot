from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
import hashlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .arm_catalog import ArmCatalog
from .context_builder import ContextBuilder
from actions.state_encode import build_dqn_state
from offline_dataset import AsyncTransitionDatasetWriter
from .protocol_rules import (
    CANDIDATE_CONFIRMATIONS,
    FPR_HIGH_THRESHOLD,
    FPR_LOW_THRESHOLD,
    LANE_HEALTHY_RATIO,
    NO_IMPROVEMENT_EPOCHS,
    PROTOCOL_METRIC_WINDOW,
    REWARD_IMPROVEMENT_THRESHOLD,
    REWARD_ROLLBACK_THRESHOLD,
    STRUCTURED_INIT_EPOCHS,
    allowed_cut_values,
    allowed_timeout_values,
)

logger = logging.getLogger(__name__)


class CMABTrainer:
    def __init__(
        self,
        metrics_dir: str,
        parameters_file: str,
        checkpoint_dir: str,
        policy: Any,
        context_builder: ContextBuilder,
        arm_catalog: ArmCatalog,
        metrics_timeout: int = 300,
        node_index: Optional[int] = None,
        warmup_iterations: int = 5,
        checkpoint_prefix: str = "cmab_checkpoint",
        enable_protocol_rules: bool = False,
        transition_writer: Optional[AsyncTransitionDatasetWriter] = None,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.parameters_file = Path(parameters_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.policy = policy
        self.context_builder = context_builder
        self.arm_catalog = arm_catalog
        self.metrics_timeout = metrics_timeout

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.last_metrics_file: Optional[Path] = None
        self.training_active = True
        self.reward_history = deque(maxlen=20)
        self.arm_counts = {}
        self._param_socket: Optional[socket.socket] = None
        self.node_index = node_index
        self.enable_protocol_rules = bool(enable_protocol_rules)
        self.warmup_iterations = (
            0 if self.enable_protocol_rules else max(0, warmup_iterations)
        )
        self.checkpoint_prefix = checkpoint_prefix or "cmab_checkpoint"
        self.transition_writer = transition_writer

        # Rule-guided CMAB state.  These fields are unused when the feature is
        # disabled, leaving the original selection path unchanged.
        self._structured_arms: list[str] = []
        self._structured_index = 0
        self._incumbent_arm: Optional[str] = None
        self._incumbent_confirmation_remaining = 0
        self._candidate_arm: Optional[str] = None
        self._candidate_rewards: list[float] = []
        self._candidate_baseline_reward: Optional[float] = None
        self._arm_reward_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max(PROTOCOL_METRIC_WINDOW, CANDIDATE_CONFIRMATIONS))
        )
        self._fpr_history: deque[float] = deque(maxlen=PROTOCOL_METRIC_WINDOW)
        self._lane_history: deque[list[float]] = deque(maxlen=PROTOCOL_METRIC_WINDOW)
        self._last_protocol_metrics_epoch: Optional[int] = None
        self._epochs_without_improvement = 0
        self._rules_converged = False

    def run(self, num_iterations: Optional[int], checkpoint_freq: int):
        logger.info("Initializing CMAB training loop...")
        logger.info("Policy: %s", self.policy.policy_name)
        logger.info("Iterations: %s", num_iterations)
        logger.info(
            "Warmup iterations (skip policy update): %s",
            self.warmup_iterations,
        )
        logger.info("CMAB protocol rules enabled: %s", self.enable_protocol_rules)
        if self.enable_protocol_rules:
            logger.info(
                "CMAB_RULES_CONFIG structured=%d metric_window=%d confirmations=%d "
                "reward_improvement=%.2f rollback=%.2f fpr_low=%.2f "
                "fpr_high=%.2f lane_ratio=%.2f no_improvement=%d",
                STRUCTURED_INIT_EPOCHS,
                PROTOCOL_METRIC_WINDOW,
                CANDIDATE_CONFIRMATIONS,
                REWARD_IMPROVEMENT_THRESHOLD,
                REWARD_ROLLBACK_THRESHOLD,
                FPR_LOW_THRESHOLD,
                FPR_HIGH_THRESHOLD,
                LANE_HEALTHY_RATIO,
                NO_IMPROVEMENT_EPOCHS,
            )
        self._connect_param_socket()
        self.last_metrics_file = self._get_latest_metrics_file()
        if self.last_metrics_file is None:
            logger.info("No existing metrics file, waiting for first available global_state...")
            self.last_metrics_file = self._wait_for_first_available_metrics_file(
                timeout=self.metrics_timeout
            )

        iteration = 0
        last_arm = self._load_initial_arm_from_parameters_file()
        if last_arm is not None:
            logger.info("Bootstrapped initial arm from parameters file: %s", last_arm)
        if self.enable_protocol_rules:
            if last_arm is None or not self.arm_catalog.contains(last_arm):
                last_arm = self.arm_catalog.list_arms()[0]
                logger.warning(
                    "Initial parameters are not a legal CMAB arm; using catalog arm: %s",
                    last_arm,
                )
            self._structured_arms = self.arm_catalog.structured_initial_arms(last_arm)
            logger.info(
                "STRUCTURED_INIT_READY count=%d expected=%d base=%s arms=%s",
                len(self._structured_arms),
                STRUCTURED_INIT_EPOCHS,
                last_arm,
                self._structured_arms,
            )
            if len(self._structured_arms) != STRUCTURED_INIT_EPOCHS:
                logger.warning(
                    "Structured initialization contains %d arms instead of %d; "
                    "continuing with the legal catalog values.",
                    len(self._structured_arms),
                    STRUCTURED_INIT_EPOCHS,
                )
        action_by_epoch: dict[int, str] = {}
        while self.training_active:
            if num_iterations is not None and iteration >= num_iterations:
                logger.info("Reached maximum iterations, stopping.")
                break

            logger.info("FEATURIZE_START metrics=%s", self.last_metrics_file.name)
            context = self._build_context_from_global_state(self.last_metrics_file)
            logger.info("FEATURIZE_DONE context_dim=%d", len(context))
            current_epoch = self._get_epoch_from_metrics_file(self.last_metrics_file)
            shared_seed_hex = self._compute_shared_seed_hex(self.last_metrics_file)
            decision_kind = "standard"
            if self.enable_protocol_rules:
                self._record_protocol_metrics(self.last_metrics_file)
                arm, decision_kind = self._select_rule_guided_arm(
                    context,
                    shared_seed_hex,
                )
            else:
                arm = self.policy.select_arm(context, shared_seed_hex=shared_seed_hex)
            params = self.arm_catalog.decode_arm(arm)
            self.arm_counts[arm] = self.arm_counts.get(arm, 0) + 1
            if current_epoch is not None:
                # Action selected from state_t is credited to reward from state_{t+1}.
                action_by_epoch[current_epoch + 1] = arm

            self._write_parameters_to_file(params, current_epoch)
            logger.info("Applied params: %s", params)

            next_metrics = self._wait_for_new_metrics_file(self.last_metrics_file, timeout=self.metrics_timeout)
            if next_metrics is None:
                logger.warning("Timeout waiting for new metrics file, skipping update.")
                continue

            # Load once for the original CMAB reward path and optional passive
            # export. Export must not add another metrics-file read.
            next_metrics_data = self._load_json_with_retry(next_metrics)
            reward = self._extract_reward_from_data(
                next_metrics_data,
                next_metrics,
            )
            reward_epoch = self._get_epoch_from_metrics_file(next_metrics)
            if reward > 15 or reward == 0:
                logger.warning(
                    "Dropping sample due to suspicious high reward (>10): reward=%.6f metrics=%s",
                    reward,
                    next_metrics.name,
                )
                # Move forward to avoid reprocessing the same metrics file.
                self.last_metrics_file = next_metrics
                continue
            if current_epoch is not None and reward_epoch is not None and reward_epoch > current_epoch + 1:
                logger.warning(
                    "Detected non-contiguous reward epoch: current=%s, reward=%s, backfilling credited arm for skipped epochs",
                    current_epoch,
                    reward_epoch,
                )
                # state_t selects action_{t+1}. action_{current+1} is already set above.
                # For jumps, only backfill truly skipped epochs: [current+2, reward_epoch].
                for epoch in range(current_epoch + 2, reward_epoch + 1):
                    action_by_epoch.setdefault(epoch, arm)
            # Credit priority:
            # 1) abandon signal => previous epoch action
            # 2) direct epoch mapping => action selected for reward_epoch
            # 3) fallback => current selected arm
            use_arm = action_by_epoch.get(reward_epoch) if reward_epoch is not None else None
            abandon_signal = None
            abandoned = False
            if reward_epoch is not None:
                abandon_signal = Path(f"/tmp/autopilot_rl_param_abandon_{reward_epoch-1}.signal")
                if abandon_signal.exists():
                    abandoned = True
                    logger.warning(
                        "Detected abandon signal for epoch %s, using previous iteration action for update",
                        reward_epoch,
                    )
                    use_arm = last_arm

            if use_arm is None:
                # Safety fallback when previous action is unavailable.
                use_arm = arm

            in_warmup = iteration < self.warmup_iterations
            if not in_warmup:
                update_contexts = [context] if self.policy.uses_context else None
                self.policy.update(
                    [use_arm],
                    [reward],
                    update_contexts,
                    shared_seed_hex=shared_seed_hex,
                )
            else:
                logger.info(
                    "Warmup iteration %s/%s: skip policy update (reward=%.6f, arm=%s)",
                    iteration + 1,
                    self.warmup_iterations,
                    reward,
                    use_arm,
                )
            if self.enable_protocol_rules:
                self._handle_rule_guided_result(
                    selected_arm=arm,
                    credited_arm=use_arm,
                    reward=reward,
                    decision_kind=decision_kind,
                )
            logger.info(
                "Iteration %s : context=%s arm=%s params=%s",
                iteration + 1,
                context,
                use_arm,
                self.arm_catalog.decode_arm(use_arm),
            )

            iteration += 1
            self.reward_history.append(reward)

            if reward_epoch is not None:
                abandon_signal = Path(f"/tmp/autopilot_rl_param_abandon_{reward_epoch-1}.signal")
                if not abandon_signal.exists():
                    last_arm = use_arm
            
            avg_reward = sum(self.reward_history) / len(self.reward_history)
            top_arm = max(self.arm_counts, key=self.arm_counts.get)
            top_ratio = self.arm_counts[top_arm] / max(1, sum(self.arm_counts.values()))
            logger.info(
                "Iteration %s reward=%.6f avg_reward(20)=%.6f last_metrics=%s top_arm=%s top_ratio=%.2f",
                iteration,
                reward,
                avg_reward,
                next_metrics.name,
                top_arm,
                top_ratio,
            )

            if iteration % checkpoint_freq == 0:
                checkpoint_path = (
                    self.checkpoint_dir / f"{self.checkpoint_prefix}_{iteration}.pkl"
                )
                self.policy.save(str(checkpoint_path))
                logger.info("Saved checkpoint: %s", checkpoint_path)

            self.last_metrics_file = next_metrics
            # Export is deliberately last and fail-open: all CMAB state,
            # checkpoint, and protocol-rule work for this iteration is complete.
            self._export_transition_best_effort(
                current_epoch=current_epoch,
                reward_epoch=reward_epoch,
                context=context,
                arm=use_arm,
                reward=reward,
                next_metrics_data=next_metrics_data,
                abandoned=abandoned,
            )

        if self.transition_writer is not None:
            self._close_transition_writer()

    def stop(self):
        self.training_active = False
        if self.transition_writer is not None:
            self._close_transition_writer()
        if self._param_socket is not None:
            try:
                self._param_socket.close()
            except OSError:
                pass
            self._param_socket = None

    def _export_transition_best_effort(
        self,
        *,
        current_epoch: Optional[int],
        reward_epoch: Optional[int],
        context: np.ndarray,
        arm: str,
        reward: float,
        next_metrics_data: dict,
        abandoned: bool,
    ) -> None:
        writer = self.transition_writer
        if writer is None:
            return
        contiguous = (
            current_epoch is not None
            and reward_epoch == current_epoch + 1
        )
        valid_reward = np.isfinite(reward) and 0 < reward <= 15
        if not contiguous or abandoned or not valid_reward:
            logger.warning(
                "CMAB_OFFLINE_TRANSITION_DROPPED source_epoch=%s "
                "reward_epoch=%s contiguous=%s abandoned=%s reward=%s",
                current_epoch,
                reward_epoch,
                contiguous,
                abandoned,
                reward,
            )
            return
        try:
            next_state = self._build_context_from_data(next_metrics_data)
            writer.write(
                source_epoch=current_epoch,
                reward_epoch=reward_epoch,
                state=context,
                arm=arm,
                reward=reward,
                next_state=next_state,
                done=False,
                truncated=False,
            )
        except Exception:
            # This also protects CMAB if a future writer implementation stops
            # being asynchronous or validates records on the caller thread.
            logger.exception(
                "CMAB_OFFLINE_TRANSITION_EXPORT_DISABLED run=%s; "
                "CMAB will continue normally",
                getattr(writer, "run_id", "unknown"),
            )
            self._close_transition_writer()

    def _close_transition_writer(self) -> None:
        writer = self.transition_writer
        self.transition_writer = None
        if writer is None:
            return
        try:
            writer.close()
        except Exception:
            logger.exception(
                "Failed to close CMAB transition writer; CMAB will continue"
            )

    def _get_latest_metrics_file(self) -> Optional[Path]:
        files = list(self.metrics_dir.glob("global_state_epoch_*.json"))
        if not files:
            return None

        def parse_file_key(file_path: Path):
            try:
                epoch = int(file_path.stem.split("_")[-1])
                return (epoch,)
            except (ValueError, IndexError, AttributeError):
                return (-1,)

        files.sort(key=parse_file_key, reverse=True)
        return files[0]

    def _wait_for_new_metrics_file(self, last_metrics_file: Path, timeout: int) -> Optional[Path]:
        logger.info("Waiting for newer metrics file (latest-wins)...")
        start_time = time.time()

        current_epoch = self._get_epoch_from_metrics_file(last_metrics_file)
        if current_epoch is None:
            current_epoch = self._get_max_epoch()

        while time.time() - start_time < timeout:
            newer_files = []
            for file_path in self.metrics_dir.glob("global_state_epoch_*.json"):
                epoch = self._get_epoch_from_metrics_file(file_path)
                if epoch is None:
                    continue
                if epoch > current_epoch:
                    newer_files.append((epoch, file_path))

            if newer_files:
                newer_files.sort(key=lambda item: item[0])
                newest_epoch, newest_file = newer_files[-1]
                if newest_epoch > current_epoch + 1:
                    logger.warning(
                        "Detected epoch gap: current=%s, newest=%s, skipping missing epochs",
                        current_epoch,
                        newest_epoch,
                    )
                return newest_file
            time.sleep(0.1)
        return None

    def _wait_for_epoch(self, epoch: int, timeout: int) -> str:
        start = time.time()
        pattern = f"global_state_epoch_{epoch}.json"
        while time.time() - start < timeout:
            files = list(self.metrics_dir.glob(pattern))
            if files:
                return str(files[0])
            time.sleep(0.1)
        raise TimeoutError(f"Timeout waiting for epoch {epoch}")

    def _wait_for_first_available_metrics_file(self, timeout: int) -> Path:
        start = time.time()
        while time.time() - start < timeout:
            latest = self._get_latest_metrics_file()
            if latest is not None:
                epoch = self._get_epoch_from_metrics_file(latest)
                if epoch is not None and epoch > 0:
                    logger.warning(
                        "First observed global state is epoch %s (epoch 0 missing), continuing with latest-wins",
                        epoch,
                    )
                return latest
            time.sleep(0.1)
        raise TimeoutError("Timeout waiting for any global_state_epoch_*.json")

    def _load_initial_arm_from_parameters_file(self) -> Optional[str]:
        if not self.parameters_file.exists():
            return None
        try:
            with open(self.parameters_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            keys = ("batch_size", "header_size", "cut_condition_type", "fast_path_timeout", "k")
            if not all(k in data for k in keys):
                return None
            parts = []
            for k in keys:
                # if k == "use_optimistic_tips":
                #     parts.append(f"{k}={1 if data[k] else 0}")
                # else:
                parts.append(f"{k}={int(data[k])}")
            arm = ",".join(parts)
            self.arm_catalog.decode_arm(arm)
            return arm
        except Exception as e:
            logger.warning(
                "Failed to bootstrap arm from parameters file %s: %s",
                self.parameters_file,
                e,
            )
            return None

    def _record_protocol_metrics(self, metrics_path: Path) -> None:
        epoch = self._get_epoch_from_metrics_file(metrics_path)
        if epoch is not None and epoch == self._last_protocol_metrics_epoch:
            return

        data = self._load_json_with_retry(metrics_path)
        try:
            fast_path_ratio = float(data.get("global_fast_path_ratio", 0.0))
        except (TypeError, ValueError):
            fast_path_ratio = 0.0

        lane_values: list[float] = []
        growth_rates = (
            data.get("state_4_lane_vector", {})
            .get("growth_rates", {})
        )
        if isinstance(growth_rates, dict):
            for _, value in sorted(growth_rates.items()):
                try:
                    lane_values.append(float(value))
                except (TypeError, ValueError):
                    continue

        self._fpr_history.append(fast_path_ratio)
        if lane_values:
            self._lane_history.append(lane_values)
        self._last_protocol_metrics_epoch = epoch
        logger.info(
            "PROTOCOL_METRICS epoch=%s fpr=%.6f lanes=%s",
            epoch,
            fast_path_ratio,
            lane_values,
        )

    def _mean_arm_reward(self, arm: str, limit: int) -> Optional[float]:
        values = list(self._arm_reward_history.get(arm, ()))
        if not values:
            return None
        recent = values[-max(1, int(limit)):]
        return sum(recent) / len(recent)

    def _choose_initial_incumbent(self) -> str:
        ranked = []
        for index, arm in enumerate(self._structured_arms):
            mean_reward = self._mean_arm_reward(arm, limit=1)
            if mean_reward is not None:
                ranked.append((mean_reward, -index, arm))
        if ranked:
            _, _, incumbent = max(ranked)
        else:
            incumbent = self._structured_arms[0]

        self._incumbent_arm = incumbent
        self._incumbent_confirmation_remaining = CANDIDATE_CONFIRMATIONS
        logger.info(
            "INITIAL_INCUMBENT arm=%s observed_reward=%s confirmations=%d",
            incumbent,
            self._mean_arm_reward(incumbent, limit=1),
            self._incumbent_confirmation_remaining,
        )
        return incumbent

    def _select_rule_guided_arm(
        self,
        context: np.ndarray,
        shared_seed_hex: str,
    ) -> tuple[str, str]:
        if self._structured_index < len(self._structured_arms):
            arm = self._structured_arms[self._structured_index]
            logger.info(
                "STRUCTURED_INIT epoch=%d/%d arm=%s",
                self._structured_index + 1,
                len(self._structured_arms),
                arm,
            )
            return arm, "structured"

        if self._incumbent_arm is None:
            self._choose_initial_incumbent()

        if self._incumbent_confirmation_remaining > 0:
            logger.info(
                "INCUMBENT_CONFIRM arm=%s remaining=%d/%d",
                self._incumbent_arm,
                self._incumbent_confirmation_remaining,
                CANDIDATE_CONFIRMATIONS,
            )
            return self._incumbent_arm, "incumbent_confirm"

        if self._candidate_arm is not None:
            logger.info(
                "CANDIDATE_HOLD arm=%s collected=%d/%d",
                self._candidate_arm,
                len(self._candidate_rewards),
                CANDIDATE_CONFIRMATIONS,
            )
            return self._candidate_arm, "candidate"

        if self._rules_converged:
            logger.info(
                "RULE_GUIDED_EXPLOIT incumbent=%s reason=no_improvement",
                self._incumbent_arm,
            )
            return self._incumbent_arm, "exploit"

        neighbors = self.arm_catalog.one_parameter_neighbors(self._incumbent_arm)
        incumbent_params = self.arm_catalog.decode_arm(self._incumbent_arm)
        allowed_timeouts, mean_fpr = allowed_timeout_values(
            incumbent_params["fast_path_timeout"],
            list(self._fpr_history),
            self.arm_catalog.timeout_values,
        )
        allowed_cuts, averaged_lanes, healthy_lanes = allowed_cut_values(
            list(self._lane_history),
            self.arm_catalog.cut_values,
        )

        current_timeout = incumbent_params["fast_path_timeout"]
        current_cut = incumbent_params["cut_condition_type"]
        filter_reason = "normal"
        if current_timeout not in allowed_timeouts:
            # Repair one protocol dimension at a time.  Timeout has priority
            # because it directly controls the fast/slow-path wait.
            filtered = [
                arm
                for arm in neighbors
                if self.arm_catalog.decode_arm(arm)["fast_path_timeout"]
                in allowed_timeouts
            ]
            filter_reason = "timeout_repair"
        elif current_cut not in allowed_cuts:
            filtered = [
                arm
                for arm in neighbors
                if self.arm_catalog.decode_arm(arm)["cut_condition_type"]
                in allowed_cuts
            ]
            filter_reason = "cut_repair"
        else:
            filtered = self.arm_catalog.filter_by_protocol_values(
                neighbors,
                timeout_values=allowed_timeouts,
                cut_values=allowed_cuts,
            )

        logger.info(
            "TIMEOUT_FILTER current=%d mean_fpr=%s allowed=%s",
            current_timeout,
            f"{mean_fpr:.6f}" if mean_fpr is not None else "n/a",
            sorted(allowed_timeouts),
        )
        logger.info(
            "CUT_FILTER current=%d averaged_lanes=%s healthy=%s allowed=%s",
            current_cut,
            averaged_lanes,
            healthy_lanes,
            sorted(allowed_cuts),
        )
        logger.info(
            "NEIGHBOR_FILTER incumbent=%s total=%d remaining=%d reason=%s arms=%s",
            self._incumbent_arm,
            len(neighbors),
            len(filtered),
            filter_reason,
            filtered,
        )

        if not filtered:
            self._rules_converged = True
            logger.info(
                "CMAB_RULES_CONVERGED incumbent=%s reason=no_legal_neighbor",
                self._incumbent_arm,
            )
            return self._incumbent_arm, "exploit"

        candidate = self.policy.select_arm(
            context,
            shared_seed_hex=shared_seed_hex,
            allowed_arms=filtered,
        )
        baseline = self._mean_arm_reward(
            self._incumbent_arm,
            limit=CANDIDATE_CONFIRMATIONS,
        )
        self._candidate_arm = candidate
        self._candidate_rewards = []
        self._candidate_baseline_reward = baseline
        logger.info(
            "CANDIDATE_START incumbent=%s candidate=%s baseline_reward=%s",
            self._incumbent_arm,
            candidate,
            f"{baseline:.6f}" if baseline is not None else "n/a",
        )
        return candidate, "candidate"

    def _clear_candidate(self) -> None:
        self._candidate_arm = None
        self._candidate_rewards = []
        self._candidate_baseline_reward = None

    def _handle_rule_guided_result(
        self,
        selected_arm: str,
        credited_arm: str,
        reward: float,
        decision_kind: str,
    ) -> None:
        self._arm_reward_history[credited_arm].append(float(reward))

        if credited_arm != selected_arm:
            logger.warning(
                "RULE_RESULT_DEFER selected=%s credited=%s kind=%s reward=%.6f",
                selected_arm,
                credited_arm,
                decision_kind,
                reward,
            )
            return

        if decision_kind == "structured":
            expected = self._structured_arms[self._structured_index]
            if selected_arm == expected:
                self._structured_index += 1
            logger.info(
                "STRUCTURED_INIT_RESULT completed=%d/%d arm=%s reward=%.6f",
                self._structured_index,
                len(self._structured_arms),
                selected_arm,
                reward,
            )
            return

        if decision_kind == "incumbent_confirm":
            self._incumbent_confirmation_remaining = max(
                0,
                self._incumbent_confirmation_remaining - 1,
            )
            logger.info(
                "INCUMBENT_CONFIRM_RESULT arm=%s reward=%.6f remaining=%d",
                selected_arm,
                reward,
                self._incumbent_confirmation_remaining,
            )
            return

        if decision_kind != "candidate" or selected_arm != self._candidate_arm:
            return

        self._candidate_rewards.append(float(reward))
        self._epochs_without_improvement += 1
        baseline = self._candidate_baseline_reward

        if (
            baseline is not None
            and baseline > 0
            and reward < baseline * (1.0 - REWARD_ROLLBACK_THRESHOLD)
        ):
            logger.info(
                "CANDIDATE_ROLLBACK incumbent=%s candidate=%s reward=%.6f "
                "baseline=%.6f reason=severe_drop",
                self._incumbent_arm,
                selected_arm,
                reward,
                baseline,
            )
            # Re-apply the incumbent on the next valid epoch before trying
            # another neighbour. This makes the rollback real at the protocol
            # level and keeps subsequent exploration one parameter away from
            # the action that is actually running.
            self._incumbent_confirmation_remaining = 1
            self._clear_candidate()
        elif len(self._candidate_rewards) < CANDIDATE_CONFIRMATIONS:
            logger.info(
                "CANDIDATE_RESULT arm=%s reward=%.6f collected=%d/%d",
                selected_arm,
                reward,
                len(self._candidate_rewards),
                CANDIDATE_CONFIRMATIONS,
            )
            return
        else:
            candidate_mean = sum(self._candidate_rewards) / len(self._candidate_rewards)
            gain = (
                (candidate_mean - baseline) / baseline
                if baseline is not None and baseline > 0
                else float("-inf")
            )
            if gain >= REWARD_IMPROVEMENT_THRESHOLD:
                previous = self._incumbent_arm
                self._incumbent_arm = selected_arm
                self._epochs_without_improvement = 0
                self._rules_converged = False
                logger.info(
                    "CANDIDATE_ACCEPT previous=%s incumbent=%s mean_reward=%.6f "
                    "baseline=%.6f gain=%.6f",
                    previous,
                    selected_arm,
                    candidate_mean,
                    baseline,
                    gain,
                )
            else:
                logger.info(
                    "CANDIDATE_REJECT incumbent=%s candidate=%s mean_reward=%.6f "
                    "baseline=%s gain=%s",
                    self._incumbent_arm,
                    selected_arm,
                    candidate_mean,
                    f"{baseline:.6f}" if baseline is not None else "n/a",
                    f"{gain:.6f}" if np.isfinite(gain) else "n/a",
                )
                self._incumbent_confirmation_remaining = 1
            self._clear_candidate()

        if self._epochs_without_improvement >= NO_IMPROVEMENT_EPOCHS:
            self._rules_converged = True
            logger.info(
                "CMAB_RULES_CONVERGED incumbent=%s no_improvement_epochs=%d",
                self._incumbent_arm,
                self._epochs_without_improvement,
            )

    def _get_max_epoch(self) -> int:
        files = list(self.metrics_dir.glob("global_state_epoch_*.json"))
        max_epoch = -1
        for file_path in files:
            try:
                epoch_str = file_path.stem.split("_")[-1]
                epoch_num = int(epoch_str)
                max_epoch = max(max_epoch, epoch_num)
            except (ValueError, IndexError, AttributeError):
                continue
        return max_epoch

    def _get_epoch_from_metrics_file(self, metrics_file: Path) -> Optional[int]:
        try:
            return int(metrics_file.stem.split("_")[-1])
        except (ValueError, IndexError, AttributeError):
            return None

    def _load_json_with_retry(self, json_path: Path) -> dict:
        data = None
        for attempt in range(5):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                if not raw.strip():
                    raise json.JSONDecodeError("Empty file", raw, 0)
                data = json.loads(raw)
                break
            except json.JSONDecodeError:
                if attempt == 4:
                    raise
                time.sleep(0.1)
        if data is None:
            raise ValueError(f"Failed to parse metrics file: {json_path}")
        return data

    def _build_context_from_global_state(self, metrics_path: Path) -> np.ndarray:
        data = self._load_json_with_retry(metrics_path)
        return self._build_context_from_data(data)

    def _build_context_from_data(self, data: dict) -> np.ndarray:
        dynamic_state = build_dqn_state(data)
        if self.context_builder.mode == "dynamic":
            return dynamic_state
        # Full mode historically concatenates context + dynamic; context is empty in current setup.
        return dynamic_state

    def _extract_reward_from_global_state(self, metrics_path: Path) -> float:
        data = self._load_json_with_retry(metrics_path)
        return self._extract_reward_from_data(data, metrics_path)

    def _extract_reward_from_data(self, data: dict, metrics_path: Path) -> float:
        reward = data.get("global_reward")
        if reward is None:
            logger.warning("global_reward missing in %s, defaulting to 0.0", metrics_path)
            return 0.0
        try:
            return float(reward)
        except (TypeError, ValueError):
            logger.warning("Invalid global_reward in %s: %s", metrics_path, reward)
            return 0.0

    def _compute_shared_seed_hex(self, metrics_path: Path) -> str:
        """
        seed_t = sha256(global_state_t || block_height || commit_hash)
        """
        data = self._load_json_with_retry(metrics_path)
        global_state_t = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        block_height = str(data.get("epoch", -1))
        commit_hash = str(data.get("cc_digest", ""))
        payload = f"{global_state_t}|{block_height}|{commit_hash}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _write_parameters_to_file(self, params, epoch: Optional[int]):
        import fcntl
        import tempfile
        import shutil

        params_file = self.parameters_file
        params_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            try:
                with open(params_file, "r") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    current_params = json.load(f)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (FileNotFoundError, json.JSONDecodeError):
                current_params = {}

            current_params.update({
                "batch_size": params["batch_size"],
                "header_size": params["header_size"],
                "cut_condition_type": params["cut_condition_type"],
                "fast_path_timeout": params["fast_path_timeout"],
                "k": params["k"],
                # "use_optimistic_tips": bool(params["use_optimistic_tips"]),
            })

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=os.path.dirname(params_file)) as temp_file:
                json.dump(current_params, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            shutil.move(temp_file.name, params_file)
            logger.info("Updated parameters file: %s", params_file)

            self._send_param_update_socket(epoch)

        except Exception as e:
            logger.error("Failed to write parameters to file: %s", e)
            if "temp_file" in locals():
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass


    def _get_param_socket_path(self) -> str:
        node_index = self.node_index
        return f"/tmp/autopilot_rl_param_{node_index}.sock"

    def _send_param_update_socket(self, epoch: Optional[int]) -> bool:
        socket_path = self._get_param_socket_path()
        payload = json.dumps({"epoch": epoch})
        for attempt in range(1, 11):
            try:
                if self._param_socket is None:
                    self._connect_param_socket()
                self._param_socket.sendall((payload + "\n").encode("utf-8"))
                logger.info("🚩 Sent RL param update via socket: %s", socket_path)
                return True
            except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
                if self._param_socket is not None:
                    try:
                        self._param_socket.close()
                    except OSError:
                        pass
                    self._param_socket = None
                if attempt < 10:
                    time.sleep(min(0.2 * attempt, 2.0))
                    continue
                logger.warning("Failed to send RL param update via socket %s: %s", socket_path, e)
                return False

    def _connect_param_socket(self) -> None:
        if self._param_socket is not None:
            return
        socket_path = self._get_param_socket_path()
        max_retries = 20
        retry_delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "🔌 Attempting to connect RL param socket: %s (attempt %s/%s)",
                    socket_path,
                    attempt,
                    max_retries,
                )
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(0.5)
                client.connect(socket_path)
                self._param_socket = client
                logger.info("🔌 Connected RL param socket: %s", socket_path)
                return
            except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
                try:
                    client.close()
                except OSError:
                    pass
                if attempt < max_retries:
                    logger.warning(
                        "⚠️  RL param socket connect attempt %s failed: %s, retrying in %.1fs...",
                        attempt,
                        e,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay = 2.0
                    continue
                logger.warning("Failed to connect RL param socket %s after %s attempts: %s", socket_path, max_retries, e)
                return
