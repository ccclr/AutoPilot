#!/usr/bin/env python3
"""
Continuous GP-BO Training Script for Autopilot System.

Reuses CMABTrainer (metrics wait / reward credit / parameter write). Only the
policy kernel is GP-UCB Bayesian Optimization.
"""

import argparse
import logging

from actions.action_encode import ActionCodec
from cmab import ArmCatalog, CMABTrainer, ContextBuilder
from gp_bo import GPBOPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Continuous GP-BO Training for Autopilot")
    parser.add_argument("--metrics-dir", type=str, default="/home/ccclr0302/autopilot/metrics")
    parser.add_argument("--parameters-file", type=str, default="/home/ccclr0302/.parameters.json")
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/gp_bo_checkpoints")
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--policy", type=str, default="gp_bo", choices=["gp_bo", "default"])
    parser.add_argument("--context-mode", type=str, default="dynamic", choices=["dynamic", "full"])
    parser.add_argument("--kappa", type=float, default=2.0, help="UCB exploration coefficient")
    parser.add_argument("--max-arms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-timeout", type=int, default=300)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--node-index", type=int, default=0)
    args = parser.parse_args()

    logger.info("Starting Autopilot Continuous GP-BO Training")
    logger.info(
        "metrics_dir=%s parameters_file=%s kappa=%.3f",
        args.metrics_dir,
        args.parameters_file,
        args.kappa,
    )

    # Reuse discrete value sets from ActionCodec; "default" shrinks to one arm.
    codec_policy = "default" if args.policy == "default" else "rf_ts"
    codec = ActionCodec(policy=codec_policy)
    arm_catalog = ArmCatalog(codec=codec, max_arms=args.max_arms, seed=args.seed)
    arms = arm_catalog.list_arms()
    feature_dim = len(arm_catalog.decode_arm(arms[0])) if arms else 0

    policy = GPBOPolicy(
        arms,
        feature_dim=feature_dim,
        policy_name="gp_bo",
        kappa=args.kappa,
        random_state=args.seed,
    )
    if args.resume_from:
        policy.load(args.resume_from)
        logger.info("Loaded GP-BO policy from %s", args.resume_from)

    context_builder = ContextBuilder(mode=args.context_mode)
    trainer = CMABTrainer(
        metrics_dir=args.metrics_dir,
        parameters_file=args.parameters_file,
        checkpoint_dir=args.checkpoint_dir,
        policy=policy,
        context_builder=context_builder,
        arm_catalog=arm_catalog,
        metrics_timeout=args.metrics_timeout,
        node_index=args.node_index,
        checkpoint_prefix="gp_bo_checkpoint",
    )

    trainer.run(num_iterations=args.num_iterations, checkpoint_freq=args.checkpoint_freq)


if __name__ == "__main__":
    main()
