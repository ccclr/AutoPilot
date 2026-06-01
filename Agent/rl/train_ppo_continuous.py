#!/usr/bin/env python3
"""
Continuous PPO Training Script for Autobahn System

This script runs a continuous PPO training process that:
1. Initializes the PPO algorithm
2. Runs standard RL training loop with the Autobahn environment
3. The environment handles waiting for metrics and computing training data
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import time

from envs.env import AutobahnEnv
from actions.state_encode import get_state_dim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousPPOTrainer:
    """Continuous PPO training manager"""

    def __init__(self, metrics_dir: str, parameters_file: str, 
                 checkpoint_dir: str, checkpoint_freq: int,
                 num_iterations: int = None):
        self.metrics_dir = Path(metrics_dir)
        self.parameters_file = Path(parameters_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_iterations = num_iterations
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.algo = None
        self.state_dim = None
        self.training_active = True
        self.iteration_count = 0
        self.initialized = False

        # Training data buffer

    def initialize_training(self):
        """Initialize the PPO training system"""
        logger.info("Initializing PPO training system...")

        # Initialize Ray
        try:
            ray.init(logging_level=logging.INFO)
        except Exception as e:
            logger.error(f"Ray initialization failed: {e}")
            raise

        # Calculate state dimension
        self.state_dim = self._calculate_state_dim()

        # Register environment
        register_env("autobahn-env", lambda cfg: AutobahnEnv(cfg))

        # Configure PPO algorithm
        # Parameters from stable-baselines3 PPO defaults (lines 84-108)
        config = (
            PPOConfig()
            .environment(
                env="autobahn-env",
                env_config={
                    "metrics_dir": str(self.metrics_dir),
                    "state_dim": self.state_dim,
                    "parameters_file": str(self.parameters_file),
                },
            )
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .env_runners(num_env_runners=1)
            .training(
                # From stable-baselines3 PPO defaults
                lr=3e-4,  # learning_rate: 3e-4
                train_batch_size=32,  # n_steps: 2048
                minibatch_size=16,  # batch_size: 64
                num_sgd_iter=10,  # n_epochs: 10
                gamma=0.99,  # gamma: 0.99
                lambda_=0.95,  # gae_lambda: 0.95
                clip_param=0.2,  # clip_range: 0.2
                # Note: stable-baselines3 default is 0.0, but for discrete action spaces
                # (MultiDiscrete), non-zero entropy encourages exploration
                entropy_coeff_schedule=[
                    [0, 0.5], [40, 0.2], [80, 0.1]
                ],  # Start with 0.5, decay to 0.1 over training
                vf_loss_coeff=0.5,  # vf_coef: 0.5
                grad_clip=0.5,  # max_grad_norm: 0.5
            )
            .resources(num_gpus=0)
            .framework("torch")
            .experimental(_validate_config=False)
        )

        # Model configuration with explicit dimensions
        config.model.update({
            "fcnet_hiddens": [32, 16],
            "fcnet_activation": "tanh",
        })

        # Log the state dimension for debugging
        print(f"PPO Configuration: state_dim={self.state_dim}")

        # Build algorithm
        try:
            self.algo = config.build_algo()
        except ImportError as e:
            logger.warning(f"RL framework unavailable: {e}")
            self.algo = None
            return
        except Exception as e:
            logger.error(f"Failed to build algorithm: {e}")
            self.algo = None
            return

        logger.info("🤖 PPO training system initialized successfully")
        self.initialized = True

    def _calculate_state_dim(self) -> int:
        """Calculate dynamic state dimension (without context)"""
        try:
            # For observation space: return only dynamic state dimension
            state_dim = get_state_dim(str(self.metrics_dir), include_context=False)
            logger.info(f"Calculated dynamic state dimension: {state_dim}")
            return state_dim
        except Exception as e:
            logger.warning(f"Unable to calculate state dimension, using default: {e}")
            # Lane growth rates dimension - typically 4 lanes
            return 4  # Default dynamic state dimension for lane growth rates


    def run_continuous_training(self):
        """Run the continuous PPO training loop - let PPO control everything"""
        logger.info("Starting continuous PPO training - PPO controls all sampling")

        try:
            self.initialize_training()

            logger.info(f"Algorithm available: {self.algo is not None}")
            if self.algo is None:
                logger.error("PPO algorithm not available, exiting")
                return

            # Let PPO handle everything: reset -> step -> step -> ... -> done -> reset -> ...
            # The environment will wait for real Autobahn epochs, PPO handles all the sampling
            iteration = 0
            logger.info("About to enter training loop...")

            while self.training_active:
                iteration += 1

                # Check iteration limit
                if self.num_iterations is not None and iteration >= self.num_iterations:
                    logger.info(f"Reached maximum iterations ({self.num_iterations})")
                    break

                try:
                    logger.info(f"[{iteration}] Starting PPO train() call...")
                    # PPO will automatically call env.reset() and env.step() as needed
                    result = self.algo.train()
                    logger.info(f"[{iteration}] PPO train() call completed, result keys: {list(result.keys())}")

                    # Log training progress
                    num_steps = result.get('num_env_steps_sampled_this_iter', 0)
                    logger.info(f"[{iteration}] PPO training iteration completed - Steps: {num_steps}")

                    # Save checkpoint periodically
                    if iteration % self.checkpoint_freq == 0:
                        try:
                            checkpoint_path = self.algo.save(str(self.checkpoint_dir / f"checkpoint_{iteration}"))
                            logger.info(f"Checkpoint saved: {checkpoint_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save checkpoint: {e}")

                except Exception as e:
                    logger.error(f"PPO training iteration {iteration} failed: {e}")
                    break

        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal, shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources (based on train_ppo.py)"""
        logger.info("🧹 Cleaning up continuous training resources...")
        self.training_active = False

        if self.algo:
            try:
                final_checkpoint = self.algo.save(str(self.checkpoint_dir / "final_model"))
                logger.info(f"  Final model saved: {final_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to save final model: {e}")

        try:
            ray.shutdown()
            logger.info("🔌 Ray framework shut down")
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Continuous PPO Training for Autobahn")
    parser.add_argument("--metrics-dir", type=str, default="/home/ccclr0302/autobahn/metrics",
                       help="Metrics file directory")
    parser.add_argument("--parameters-file", type=str, default="/home/ccclr0302/.parameters.json",
                       help="Parameter file path")
    parser.add_argument("--slots-per-epoch", type=int, default=20,
                       help="Number of slots per epoch")
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/ppo_continuous_checkpoints",
                       help="Checkpoint save directory")
    parser.add_argument("--num-iterations", type=int, default=None,
                       help="Number of training iterations to run (None for continuous)")
    parser.add_argument("--checkpoint-freq", type=int, default=1,
                       help="Frequency to save checkpoints (every N iterations)")

    args = parser.parse_args()

    logger.info("  Starting Autobahn Continuous PPO Training System")
    logger.info(f"  Configuration: metrics_dir={args.metrics_dir}, parameters_file={args.parameters_file}")

    # Create and run continuous trainer
    trainer = ContinuousPPOTrainer(
        metrics_dir=args.metrics_dir,
        parameters_file=args.parameters_file,
        checkpoint_dir=args.checkpoint_dir,
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq
    )

    trainer.run_continuous_training()


if __name__ == "__main__":
    main()
