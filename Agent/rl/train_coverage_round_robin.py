#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from actions.action_encode import ActionCodec
from cmab import ArmCatalog
from controllers.action_transport import ActionBroadcaster
from coverage_round_robin import CoverageRoundRobinTrainer
from offline_dataset import TransitionDatasetWriter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centralized shuffled round-robin coverage collection"
    )
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--parameters-file", required=True)
    # Accepted for the common controller interface; coverage creates no checkpoint.
    parser.add_argument("--checkpoint-dir", default=str(Path.home() / "checkpoints"))
    parser.add_argument("--node-index", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--metrics-timeout", type=int, default=300)
    parser.add_argument("--action-endpoints", required=True)
    parser.add_argument("--action-timeout", type=float, default=2.0)
    parser.add_argument("--action-retries", type=int, default=2)
    parser.add_argument("--transition-export-dir", required=True)
    parser.add_argument("--environment-label", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    if args.node_index != 0:
        parser.error("coverage round-robin collector must run on node 0")
    if args.seed < 0:
        parser.error("coverage seed must be non-negative")

    catalog = ArmCatalog(codec=ActionCodec(policy="rf_ts"))
    arms = tuple(catalog.list_arms())
    broadcaster = ActionBroadcaster.from_csv(
        args.action_endpoints,
        timeout=args.action_timeout,
        retries=args.action_retries,
    )
    writer = TransitionDatasetWriter(
        root_dir=args.transition_export_dir,
        environment=args.environment_label,
        run_id=args.run_id,
        arms=arms,
        node_index=0,
        behavior_policy="coverage_round_robin",
        metadata={
            "policy": "coverage_round_robin",
            "seed": args.seed,
            "action_order": "shuffle_without_replacement_per_cycle",
            "warmup_iterations_ignored": args.warmup_iterations,
        },
    )
    logger.info(
        "COVERAGE_CONFIG actions=%d seed=%d endpoints=%s dataset=%s",
        len(arms),
        args.seed,
        args.action_endpoints,
        writer.transition_path,
    )
    trainer = CoverageRoundRobinTrainer(
        metrics_dir=args.metrics_dir,
        arms=arms,
        decode_arm=catalog.decode_arm,
        broadcaster=broadcaster,
        transition_writer=writer,
        seed=args.seed,
        metrics_timeout=args.metrics_timeout,
    )
    trainer.run(args.num_iterations)


if __name__ == "__main__":
    main()
