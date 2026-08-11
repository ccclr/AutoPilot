#!/usr/bin/env python3
"""Continuous KernelUCB training entry point for Autopilot."""

import argparse
import logging
import math
from pathlib import Path

from actions.action_encode import ActionCodec
from actions.mixed_action_space import MixedActionSpace, MixedArmCatalog
from cmab import CMABTrainer, ContextBuilder
from kernel_ucb import ContinuousKernelUCBPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Continuous-timeout KernelUCB training for Autopilot"
    )
    parser.add_argument("--metrics-dir", default=str(home / "autopilot" / "metrics"))
    parser.add_argument("--parameters-file", default=str(home / ".parameters.json"))
    parser.add_argument("--checkpoint-dir", default="/tmp/kernel_ucb_checkpoints")
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--context-mode", choices=["dynamic", "full"], default="dynamic")
    parser.add_argument("--ucb-alpha", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--timeout-min", type=float, default=1.0)
    parser.add_argument("--timeout-max", type=float, default=300.0)
    parser.add_argument("--optimizer-restarts", type=int, default=5)
    parser.add_argument("--replay-window", type=int, default=200)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=0,
        help="Number of cold-start samples retained before the first kernel fit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-timeout", type=int, default=300)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--node-index", type=int, default=0)
    args = parser.parse_args()

    if args.timeout_min <= 0 or args.timeout_max < args.timeout_min:
        parser.error("timeout bounds must satisfy 0 < timeout-min <= timeout-max")
    if math.ceil(args.timeout_min) > math.floor(args.timeout_max):
        parser.error("timeout bounds must contain at least one integer millisecond")

    logger.info(
        "Starting continuous KernelUCB: timeout=0 or [%.3f, %.3f]ms "
        "alpha=%.3f lambda=%.6f length_scale=%.3f restarts=%d",
        args.timeout_min,
        args.timeout_max,
        args.ucb_alpha,
        args.regularization,
        args.length_scale,
        args.optimizer_restarts,
    )

    codec = ActionCodec(policy="rf_ts")
    mixed_space = MixedActionSpace(
        codec=codec,
        timeout_bounds=(0.0, args.timeout_max),
    )
    arm_catalog = MixedArmCatalog(mixed_space)
    policy = ContinuousKernelUCBPolicy(
        mixed_space=mixed_space,
        ucb_alpha=args.ucb_alpha,
        regularization=args.regularization,
        length_scale=args.length_scale,
        replay_window=args.replay_window,
        min_samples_to_fit=max(1, args.warmup_iterations),
        random_state=args.seed,
        positive_timeout_min=args.timeout_min,
        optimizer_restarts=args.optimizer_restarts,
    )
    if args.resume_from:
        policy.load(args.resume_from)
        logger.info("Loaded KernelUCB policy from %s", args.resume_from)

    trainer = CMABTrainer(
        metrics_dir=args.metrics_dir,
        parameters_file=args.parameters_file,
        checkpoint_dir=args.checkpoint_dir,
        policy=policy,
        context_builder=ContextBuilder(mode=args.context_mode),
        arm_catalog=arm_catalog,
        metrics_timeout=args.metrics_timeout,
        node_index=args.node_index,
        # KernelUCB warmup samples are retained by the policy.
        warmup_iterations=0,
        checkpoint_prefix="kernel_ucb_checkpoint",
    )
    trainer.run(
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()
