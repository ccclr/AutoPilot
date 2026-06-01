import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from pathlib import Path
import logging
import json
import os
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from actions.action_encode import ActionCodec
from actions.state_encode import parse_metrics_with_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutobahnEnv(gym.Env):
    # Global state to track where to continue from
    _global_start_epoch = 0
    # Global state to track current progress across all environment instances
    _global_current_epoch = 0
    def __init__(self, config):
        self.metrics_dir = Path(config["metrics_dir"])
        self.parameters_file = Path(config["parameters_file"])
        self.codec = ActionCodec()

        self.action_space = spaces.MultiDiscrete(self.codec.action_dims)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.get("dynamic_state_dim", 4),),  # Default to 4 for lane growth rates, will be updated
            dtype=np.float32,
        )
        self._observation_space_initialized = False

        # Track current epoch and previous training data
        # Use start_epoch from config if provided, otherwise use global
        self.start_epoch = config.get("start_epoch", AutobahnEnv._global_start_epoch)
        self.current_epoch = self.start_epoch
        self.previous_state = None  # State from previous epoch (for training)
        self.previous_action = None  # Action from previous epoch (for training)
        self.last_metrics_file = None  # Track last read metrics file

        # Prediction model training tracking
        # self.previous_predicted_state = None  # Predicted state from previous step (for training prediction model)
        # self.previous_action_for_prediction = None  # Action used for previous prediction (for training prediction model)

        # Context (system configuration) - set once per episode
        self.episode_context = None  # Hardware + network config, relatively static
        self.context_dim = None  # Will be determined on first reset

        # Per-dimension EMA Normalization for dynamic state and context
        self.ema_alpha = 0.5  # EMA smoothing factor (50% new, 50% old)

        # Dynamic state normalization
        self.state_mean = None  # Running mean for dynamic state dimensions
        self.state_var = None   # Running variance for dynamic state dimensions
        self.state_std = None   # Running std for dynamic state dimensions
        self.last_raw_dynamic_state = None  # Store last raw dynamic state for debugging

        # Context normalization (less frequent updates since context is relatively static)
        self.context_mean = None  # Running mean for context dimensions
        self.context_var = None   # Running variance for context dimensions
        self.context_std = None   # Running std for context dimensions
        self.last_raw_context = None  # Store last raw context for debugging

        self.normalization_steps = 0  # Number of steps used for normalization
        self.context_normalization_steps = 0  # Number of context updates
        self.current_raw_dynamic_state = None  # Store current raw dynamic state for debugging

        # State prediction neural network
        self.state_dim = config.get("dynamic_state_dim", 4)
        self._build_prediction_model()

        # Prediction model training
        self.prediction_optimizer = torch.optim.Adam(self.prediction_model.parameters(), lr=1e-4)
        self.prediction_loss_fn = nn.MSELoss()
        self.prediction_training_enabled = config.get("train_prediction_model", False)

        # Training data buffer for prediction model
        self.prediction_buffer = []
        self.max_buffer_size = config.get("prediction_buffer_size", 4)

    def latest_metrics_file(self) -> Path:
        """
        Get the latest metrics file by modification time
        Returns:
            Path: latest metrics file path
        Raises:
            IndexError: if no metrics file is found
        """
        files = list(self.metrics_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No metrics file found in directory {self.metrics_dir}")

        # Sort by modification time (newest first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return files[0]
    
    def reset(self, *, seed=None, options=None):
        """
        Reset environment, return initial state

        In this contextual approach:
        - Set context (system config) once per episode
        - Return initial dynamic state

        Returns:
            tuple: (observation, info) - observation is dynamic_state only
        """
        super().reset(seed=seed)

        try:
            # Find the latest available metrics file to resume from
            latest_metrics = self._get_latest_metrics_file()
            if latest_metrics:
                metrics = str(latest_metrics)
                logger.info(f"Resuming from latest metrics file: {latest_metrics.name}")
            else:
                # No existing metrics, wait for initial epoch
                start_from_epoch = max(AutobahnEnv._global_current_epoch, self.start_epoch)
                logger.info(f"No existing metrics, waiting for epoch {start_from_epoch}")
                metrics = self._wait_for_epoch(start_from_epoch)

            # Update last metrics file tracking on reset
            self.last_metrics_file = Path(metrics)

            # Parse metrics with context separation
            context, dynamic_state, _, _ = parse_metrics_with_context(metrics)

            # Update context normalization statistics
            self.update_context_normalization_stats(context)

            # Apply EMA normalization to context
            # normalized_context = self.normalize_context_ema(context)
            normalized_context = context

            # Set context for this episode (only done once per episode)
            self.episode_context = normalized_context
            if self.context_dim is None:
                self.context_dim = len(context)

            # Initialize or update observation space based on actual dimensions
            actual_dynamic_dim = len(dynamic_state)
            if not self._observation_space_initialized:
                # First time: set observation space to match actual dynamic state dimension
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(actual_dynamic_dim,),
                    dtype=np.float32,
                )
                self._observation_space_initialized = True
                logger.info(f"Initialized observation space with dynamic state dimension: {actual_dynamic_dim}")
            elif self.observation_space.shape[0] != actual_dynamic_dim:
                # Dimension mismatch: update observation space
                logger.warning(f"Updating observation space dimension: {self.observation_space.shape[0]} -> {actual_dynamic_dim}")
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(actual_dynamic_dim,),
                    dtype=np.float32,
                )

            # Update EMA normalization statistics with raw dynamic state
            # self.update_normalization_stats(dynamic_state)
            # normalized_dynamic_state = self.normalize_state_ema(dynamic_state)
            normalized_dynamic_state = dynamic_state

            # Initialize tracking variables
            # Set current_epoch based on the epoch we're starting from
            if self.last_metrics_file is not None:
                try:
                    epoch_str = self.last_metrics_file.name.split('_')[1]
                    self.current_epoch = int(epoch_str)
                    logger.info(f"Resuming from epoch {self.current_epoch}")
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse epoch from {self.last_metrics_file.name}, starting from 0")
                    self.current_epoch = 0
            else:
                self.current_epoch = 0
                logger.info("Starting from epoch 0")

            # 不要重置previous_state和previous_action，因为我们需要它们来创建下一个step的training_data
            # 这些值会在step方法中更新，应该跨episode保留
            # self.previous_state = None
            # self.previous_action = None

            # Reset prediction training tracking
            # 同样，不要重置previous_predicted_state，因为我们需要它来训练预测模型
            # self.previous_predicted_state = None
            # self.previous_action_for_prediction = None

        except (FileNotFoundError, IndexError, KeyError) as e:
            # If metrics file doesn't exist or format is wrong, use zero states
            import warnings
            warnings.warn(f"Unable to find or parse metrics file, using zero states: {e}")

            # Initialize with zeros using current observation space dimensions
            if self.context_dim is None:
                # Assume context takes roughly half the dimensions, but at least 4
                estimated_context_dim = max(4, self.observation_space.shape[0] // 2)
                self.context_dim = estimated_context_dim

            self.episode_context = np.zeros(self.context_dim, dtype=np.float32)
            dynamic_state = np.zeros(self.observation_space.shape[0], dtype=np.float32)

            # Update and normalize zero state
            # self.update_normalization_stats(dynamic_state)
            # normalized_dynamic_state = self.normalize_state_ema(dynamic_state)
            normalized_dynamic_state = dynamic_state

            # Initialize tracking variables
            # Set current_epoch based on the epoch we're starting from
            if self.last_metrics_file is not None:
                try:
                    epoch_str = self.last_metrics_file.name.split('_')[1]
                    self.current_epoch = int(epoch_str)
                    logger.info(f"Resuming from epoch {self.current_epoch} (fallback)")
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse epoch from {self.last_metrics_file.name}, starting from 0 (fallback)")
                    self.current_epoch = 0
            else:
                self.current_epoch = 0
                logger.info("Starting from epoch 0 (fallback)")

            # 不要重置previous_state和previous_action，因为我们需要它们来创建下一个step的training_data
            # 这些值会在step方法中更新，应该跨episode保留
            # self.previous_state = None
            # self.previous_action = None

            # Reset prediction training tracking
            # 同样，不要重置previous_predicted_state，因为我们需要它来训练预测模型
            # self.previous_predicted_state = None
            # self.previous_action_for_prediction = None

        return normalized_dynamic_state, {}

    def step(self, action):
        """
        Execute one action step: wait for new metrics file and compute training data

        The logic is:
        - Apply the action (parameters) to the system
        - Wait for a new metrics file to appear
        - Parse the new metrics file to get current state and reward
        - Use previous state/action + current reward to create training data
        - Predict next state for seamless transitions
        - Return the predicted next state as observation

        Args:
            action: MultiDiscrete action vector

        Returns:
            tuple: (next_observation, reward, done, truncated, info)
        """
        # Decode action to parameter values
        params = self.codec.decode(action)

        # Apply parameters to system
        self._write_parameters_to_file(params)
        logger.info(f"Applied parameters to system: {params}")

        # Wait for new metrics file
        metrics_file = self._wait_for_new_metrics_file()
        if metrics_file is None:
            logger.warning("Timeout waiting for new metrics file, using fallback")
            next_state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            reward = -50.0
            done = True
            truncated = False
            info = {"params": params, "epoch": self.current_epoch, "training_data": None}
            return next_state, reward, done, truncated, info

        # Parse new metrics file
        try:
            context, dynamic_state, reward, _ = parse_metrics_with_context(str(metrics_file))
            self.current_epoch += 1
            # Update global progress to ensure next episode continues from here
            AutobahnEnv._global_current_epoch = max(AutobahnEnv._global_current_epoch, self.current_epoch)

            # Update context normalization (less frequent)
            # self.update_context_normalization_stats(context)
            # normalized_context = self.normalize_context_ema(context)
            self.episode_context = context

            # Handle dynamic state dimension
            expected_dynamic_dim = self.observation_space.shape[0]
            if len(dynamic_state) != expected_dynamic_dim:
                if len(dynamic_state) < expected_dynamic_dim:
                    dynamic_state = np.pad(dynamic_state, (0, expected_dynamic_dim - len(dynamic_state)))
                else:
                    dynamic_state = dynamic_state[:expected_dynamic_dim]

            # Update normalization stats and normalize
            # self.update_normalization_stats(dynamic_state)
            # current_normalized_state = self.normalize_state_ema(dynamic_state)
            current_normalized_state = dynamic_state

            # logger.info(f"Parsed metrics from epoch {self.current_epoch}: reward={reward:.3f}")
            # if action_from_metrics is not None:
            #     logger.info(f"Action from metrics file: {action_from_metrics['batch_size']}")

        except Exception as e:
            logger.warning(f"Failed to parse metrics file: {e}, using fallback")
            current_normalized_state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            reward = -50.0

        # Train prediction model using previous epoch's prediction vs current epoch's actual state
        # if self.previous_predicted_state is not None and self.previous_action_for_prediction is not None:
        #     self.add_prediction_training_data(
        #         self.previous_predicted_state, 
        #         self.previous_action_for_prediction, 
        #         current_normalized_state
        #     )
        #     logger.info(f"Added training data for prediction model: (predicted_state_from_epoch_{self.current_epoch-1}, actual_state_in_epoch_{self.current_epoch})")


        next_observation = current_normalized_state

        # batch_size_action = {'batch_size': action_from_metrics['batch_size']} if action_from_metrics is not None and 'batch_size' in action_from_metrics else params
        # predicted_next_state = self._predict_next_state(current_normalized_state, batch_size_action)

        training_data = None
        if self.previous_state is not None and self.previous_action is not None:
            full_previous_state = np.concatenate([self.episode_context, self.previous_state])
            # full_predicted_next_state = np.concatenate([self.episode_context, predicted_next_state])

            training_data = {
                'state': full_previous_state,
                'action': params,
                'reward': reward,
                'next_state': next_observation, 
                'epoch': self.current_epoch - 1
            }
            logger.info(f"Created PPO training data: state(action={params.get('batch_size')}) + reward({reward:.3f}) -> next_state(predicted_by_StatePredictor)")

        # Update for next step
        self.previous_state = current_normalized_state
        self.previous_action = params
        self.last_metrics_file = metrics_file

        done = False  # Each step is a complete episodee
        truncated = False
        info = {
            "params": params,
            "epoch": self.current_epoch,
            "training_data": training_data
        }

        logger.info(f"Current epoch {self.current_epoch} actual state: {current_normalized_state}")
        # logger.info(f"Next epoch {self.current_epoch+1} state: {predicted_next_state}")
        logger.info(f"Action: {params}")
        logger.info(f"Training data: {training_data}")

        # self.previous_predicted_state = predicted_next_state.copy()
        # self.previous_action_for_prediction = batch_size_action.copy()

        logger.info(f"Step completed - Epoch: {self.current_epoch}, Reward: {reward:.3f}, Next obs is actual state")

        return next_observation, reward, done, truncated, info


    def _wait_for_new_metrics_file(self, timeout: int = 60) -> Optional[Path]:
        """
        Wait for a new metrics file to appear, determined by epoch number

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Path to new metrics file, or None if timeout
        """
        logger.info("Waiting for new metrics file...")
        start_time = time.time()

        # Determine the next expected epoch number
        if self.last_metrics_file is not None:
            # Extract epoch from last processed file
            try:
                last_epoch_str = self.last_metrics_file.name.split('_')[1]
                current_epoch = int(last_epoch_str)
                next_epoch = current_epoch + 1
                logger.info(f"Last processed epoch: {current_epoch}, waiting for epoch {next_epoch}...")
            except (ValueError, IndexError):
                logger.warning("Could not parse epoch from last file, using max available")
                next_epoch = self._get_max_epoch() + 1
                logger.info(f"Could not parse last epoch, waiting for epoch {next_epoch}...")
        else:
            # No last file, start from epoch 0
            next_epoch = 0
            logger.info("No previous metrics file, waiting for epoch 0...")

        while time.time() - start_time < timeout:
            # Check for any file with the next epoch number
            epoch_files = list(self.metrics_dir.glob(f"epoch_{next_epoch}_slot_*.json"))

            if epoch_files:
                # Take the first matching file (should only be one)
                found_file = epoch_files[0]
                logger.info(f"Found expected metrics file: {found_file.name}")
                return found_file

            time.sleep(0.1)  # Small sleep to avoid busy-waiting

        logger.warning(f"Timeout waiting for epoch {next_epoch} metrics file after {timeout} seconds")
        return None

    def _wait_for_epoch(self, epoch, timeout=300):
        start = time.time()
        pattern = f"epoch_{epoch}_slot_*.json"

        while time.time() - start < timeout:
            files = list(self.metrics_dir.glob(pattern))
            if files:
                return str(files[0])
            time.sleep(0.1)

        raise TimeoutError(f"Timeout waiting for epoch {epoch}")

    def _get_latest_metrics_file(self) -> Optional[Path]:
        """Get the latest metrics file by epoch and slot number"""
        files = list(self.metrics_dir.glob("epoch_*_slot_*.json"))
        if not files:
            return None

        # Sort by epoch first, then by slot number
        def parse_file_key(file_path):
            try:
                parts = file_path.name.split('_')
                epoch = int(parts[1])
                slot = int(parts[3].split('.')[0])  # Remove .json extension
                return (epoch, slot)
            except (ValueError, IndexError):
                return (-1, -1)

        files.sort(key=parse_file_key, reverse=True)
        return files[0]

    def _get_max_epoch(self) -> int:
        """Get the maximum epoch number from existing files"""
        files = list(self.metrics_dir.glob("epoch_*_slot_*.json"))
        max_epoch = -1

        for file_path in files:
            try:
                epoch_str = file_path.name.split('_')[1]
                epoch_num = int(epoch_str)
                max_epoch = max(max_epoch, epoch_num)
            except (ValueError, IndexError):
                continue

        return max_epoch

    def _predict_next_state(self, current_state: np.ndarray, action: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Predict the next state using a neural network

        Args:
            current_state: Current normalized state
            action: Action that led to current state

        Returns:
            Predicted next state, or None if prediction fails
        """
        try:
            # Encode action to one-hot vector
            action_encoded = self.codec.encode(action)
            if action_encoded is None:
                logger.warning("Failed to encode action for prediction")
                return None

            # 准备输入：分别处理state和action（不再简单拼接）
            current_state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(self.device)  # [1, state_dim]
            action_encoded_tensor = torch.from_numpy(action_encoded.astype(np.float32)).float().unsqueeze(0).to(self.device)  # [1, action_dim]

            # 前向传播：使用明确的状态转移语义
            # next_state = current_state + delta(current_state, action)
            with torch.no_grad():
                predicted_tensor = self.prediction_model(current_state_tensor, action_encoded_tensor)
                predicted_state = predicted_tensor.squeeze(0).cpu().numpy()

            # Keep within reasonable bounds
            predicted_state = np.clip(predicted_state, -100, 100)

            # Ensure correct dtype
            return predicted_state.astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to predict next state with neural network: {e}")
            # Fallback to simple noise-based prediction
            try:
                noise_scale = 0.02  # 2% noise
                noise = np.random.normal(0, noise_scale * np.abs(current_state))
                predicted_state = current_state + noise
                predicted_state = np.clip(predicted_state, -5, 5)
                return predicted_state
            except Exception as fallback_e:
                logger.warning(f"Fallback prediction also failed: {fallback_e}")
                return None

    def _write_parameters_to_file(self, params: Dict[str, Any]):
        """
        Write the learned parameters to the parameters file so metrics_collector can use them.

        Args:
            params: Dictionary of parameters learned by RL
        """
        import fcntl
        import tempfile
        import shutil

        params_file = self.parameters_file

        try:
            # Read current parameters file
            try:
                with open(params_file, "r") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    current_params = json.load(f)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (FileNotFoundError, json.JSONDecodeError):
                current_params = {}

            # Update parameters
            current_params.update({
                'batch_size': params['batch_size'],
                # 'max_batch_delay': params['max_batch_delay'],
                # 'header_size': params['header_size'],
                # 'cut_condition_type': params['cut_condition_type'],
                # 'fast_path_timeout': params['fast_path_timeout_ms'],
                # 'k': params['parallel_proposals']
            })

            # Atomic write
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=os.path.dirname(params_file)) as temp_file:
                json.dump(current_params, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            shutil.move(temp_file.name, params_file)
            logger.info(f"✅ Updated parameters file: {params_file}")

            # Create signal file to notify core about parameter update (RL agent -> core)
            signal_file = "/tmp/autobahn_rl_param_update.signal"
            try:
                Path(signal_file).touch()
                logger.info(f"🚩 Created RL parameter update signal file: {signal_file}")
            except Exception as signal_error:
                logger.warning(f"Failed to create RL parameter update signal file: {signal_error}")

        except Exception as e:
            logger.error(f"❌ Failed to write parameters to file: {e}")
            # Clean up temp file if it exists
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file.name)
                except:
                    pass


    def update_normalization_stats(self, state: np.ndarray):
        """
        Update per-dimension EMA normalization statistics for dynamic state.

        Args:
            state: Raw dynamic state vector (before normalization)
        """
        try:
            # Store raw state for debugging
            self.last_raw_dynamic_state = state.copy()

            if self.state_mean is None:
                # Initialize with first state
                self.state_mean = state.copy().astype(np.float32)
                self.state_var = np.zeros_like(state, dtype=np.float32)
                self.state_std = np.ones_like(state, dtype=np.float32)  # Start with std=1 to avoid division by zero
                self.normalization_steps = 1
                return

            # Update running statistics using EMA
            self.normalization_steps += 1

            # EMA update for mean: mean = alpha * new + (1-alpha) * mean
            delta = state - self.state_mean
            self.state_mean = self.state_mean + self.ema_alpha * delta

            # EMA update for variance (using Welford's online algorithm adapted for EMA)
            delta2 = state - self.state_mean
            self.state_var = (1 - self.ema_alpha) * (self.state_var + self.ema_alpha * delta * delta2)

            # Update standard deviation
            self.state_std = np.sqrt(np.maximum(self.state_var, 1e-8))  # Avoid zero division

        except Exception as e:
            logger.warning(f"Failed to update dynamic state normalization stats: {e}")

    def update_context_normalization_stats(self, context: np.ndarray):
        """
        Update per-dimension EMA normalization statistics for context.
        Context changes less frequently, so we use a slower EMA (smaller alpha).

        Args:
            context: Raw context vector (before normalization)
        """
        try:
            # Store raw context for debugging
            self.last_raw_context = context.copy()

            context_alpha = self.ema_alpha * 0.1  # Slower adaptation for context (10x slower)

            if self.context_mean is None:
                # Initialize with first context
                self.context_mean = context.copy().astype(np.float32)
                self.context_var = np.zeros_like(context, dtype=np.float32)
                self.context_std = np.ones_like(context, dtype=np.float32)
                self.context_normalization_steps = 1
                return

            # Update running statistics using EMA
            self.context_normalization_steps += 1

            # EMA update for mean
            delta = context - self.context_mean
            self.context_mean = self.context_mean + context_alpha * delta

            # EMA update for variance
            delta2 = context - self.context_mean
            self.context_var = (1 - context_alpha) * (self.context_var + context_alpha * delta * delta2)

            # Update standard deviation
            self.context_std = np.sqrt(np.maximum(self.context_var, 1e-8))

        except Exception as e:
            logger.warning(f"Failed to update context normalization stats: {e}")

    def normalize_context_ema(self, context: np.ndarray) -> np.ndarray:
        """
        Apply per-dimension EMA normalization to context.

        Args:
            context: Raw context vector

        Returns:
            np.ndarray: Normalized context vector
        """
        try:
            if self.context_mean is None or self.context_normalization_steps < 2:
                # Not enough data for reliable normalization, return as-is
                # (assuming context is pre-normalized from state_encode.py)
                return context

            # Apply z-score normalization
            normalized = (context - self.context_mean) / self.context_std

            # Clip extreme values
            normalized = np.clip(normalized, -5.0, 5.0)

            return normalized.astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to normalize context with EMA: {e}")
            return context  # Return original context as fallback

    def normalize_state_ema(self, state: np.ndarray) -> np.ndarray:
        """
        Apply per-dimension EMA normalization to state.

        Args:
            state: Raw state vector

        Returns:
            np.ndarray: Normalized state vector
        """
        try:
            if self.state_mean is None or self.normalization_steps < 2:
                # Not enough data for reliable normalization, return zero-centered with unit variance
                # This prevents instability during early training
                return (state - np.mean(state)) / (np.std(state) + 1e-8)

            # Apply z-score normalization: (x - mean) / std
            normalized = (state - self.state_mean) / self.state_std

            # Clip extreme values to prevent gradient explosion
            normalized = np.clip(normalized, -5.0, 5.0)

            return normalized.astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to normalize state with EMA: {e}")
            # Fallback to simple normalization
            return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def _build_prediction_model(self):
        """
        Build a simple neural network for state prediction
        """
        class StatePredictor(nn.Module):
            """
            State Transition Model: 给定当前state和action，预测下一个state
            
            语义：next_state = current_state + delta(state, action)
            这种设计更明确地表达了"状态转移"的概念
            """
            def __init__(self, state_dim, action_dim, hidden_dim=32):
                super(StatePredictor, self).__init__()
                self.state_dim = state_dim
                self.action_dim = action_dim
                
                logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")

                # State编码器：处理当前状态
                self.state_encoder = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2)
                )
                
                # Action编码器：处理动作
                self.action_encoder = nn.Sequential(
                    nn.Linear(action_dim, hidden_dim // 2),
                    nn.ReLU()
                )
                
                # 融合层：将state和action的编码融合，预测状态变化delta
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, state_dim)  # 输出状态变化量delta
                )

            def forward(self, current_state, action_encoded):
                """
                前向传播：预测下一个状态
                
                Args:
                    current_state: 当前状态 [batch_size, state_dim]
                    action_encoded: 编码后的动作 [batch_size, action_dim]
                
                Returns:
                    next_state: 预测的下一个状态 [batch_size, state_dim]
                """
                # 分别编码state和action
                state_features = self.state_encoder(current_state)  # [batch_size, hidden_dim//2]
                action_features = self.action_encoder(action_encoded)  # [batch_size, hidden_dim//2]
                
                # 融合state和action的特征
                combined = torch.cat([state_features, action_features], dim=1)  # [batch_size, hidden_dim]
                
                # 预测状态变化量delta
                delta = self.fusion(combined)  # [batch_size, state_dim]
                
                # 使用残差连接：next_state = current_state + delta
                # 这明确表达了"状态转移"的语义
                next_state = current_state + delta
                
                return next_state

        # Get action dimensions from codec
        action_dim = len(self.codec.action_dims)
        self.prediction_model = StatePredictor(self.state_dim, action_dim)
        self.prediction_model.eval()  # Set to evaluation mode

        # Move to CPU (since we don't have GPU in this setup)
        self.device = torch.device('cpu')
        self.prediction_model.to(self.device)

        logger.info(f"Built state transition model: state_dim={self.state_dim}, action_dim={action_dim}")
        logger.info(f"Model semantics: next_state = current_state + delta(current_state, action)")

    def update_prediction_model(self, current_state: np.ndarray, action: Dict[str, Any], next_state: np.ndarray):
        """
        Update the prediction model using supervised learning

        Args:
            current_state: Current state before action
            action: Action taken
            next_state: Actual next state observed
        """
        if not self.prediction_training_enabled or len(self.prediction_buffer) < 10:
            # Not enough data or training disabled
            return

        try:
            # Encode action
            action_encoded = self.codec.encode(action)
            if action_encoded is None:
                return

            # Prepare training data：分别处理state和action
            current_state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(self.device)
            action_encoded_tensor = torch.from_numpy(action_encoded.astype(np.float32)).float().unsqueeze(0).to(self.device)
            target_tensor = torch.from_numpy(next_state).float().to(self.device)

            # Forward pass：使用新的模型接口
            self.prediction_model.train()
            prediction = self.prediction_model(current_state_tensor, action_encoded_tensor)
            loss = self.prediction_loss_fn(prediction.squeeze(0), target_tensor)

            # Backward pass
            self.prediction_optimizer.zero_grad()
            loss.backward()
            self.prediction_optimizer.step()

            self.prediction_model.eval()

            # Log training progress occasionally
            if len(self.prediction_buffer) % 100 == 0:
                logger.info(f"Prediction model loss: {loss.item():.6f}")

        except Exception as e:
            logger.warning(f"Failed to update prediction model: {e}")

    def add_prediction_training_data(self, current_state: np.ndarray, action: Dict[str, Any], next_state: np.ndarray):
        """
        Add training data to the prediction buffer

        Args:
            current_state: Current state
            action: Action taken
            next_state: Resulting next state
        """
        try:
            action_encoded = self.codec.encode(action)
            if action_encoded is None:
                return

            training_sample = {
                'current_state': current_state.copy(),
                'action_encoded': action_encoded.copy(),
                'next_state': next_state.copy()
            }

            self.prediction_buffer.append(training_sample)

            logger.info(f"Added prediction training data #{len(self.prediction_buffer)}: "
                       f"current_state={current_state}, "
                       f"action={action}, "
                       f"next_state={next_state}")

            # Keep buffer size limited
            if len(self.prediction_buffer) > self.max_buffer_size:
                self.prediction_buffer.pop(0)

            # Train model periodically: 只有当buffer有足够样本时才训练

                if len(self.prediction_buffer) % 2 == 0 or len(self.prediction_buffer) >= self.max_buffer_size:
                    self._train_prediction_model_batch()

        except Exception as e:
            logger.warning(f"Failed to add prediction training data: {e}")

    def _train_prediction_model_batch(self):
        """Train prediction model on a batch of data from buffer"""
        if len(self.prediction_buffer) < 1:
            return

        try:
            # Sample recent data: 使用较小的batch size，但至少要有几个样本
            batch_size = min(16, max(4, len(self.prediction_buffer) // 2))  # 使用buffer的一半，但不超过16，至少4个
            if len(self.prediction_buffer) < batch_size:
                batch_size = len(self.prediction_buffer)
            
            # 如果有足够样本，使用随机采样；否则使用所有样本
            if len(self.prediction_buffer) >= batch_size:
                batch_indices = np.random.choice(len(self.prediction_buffer), batch_size, replace=False)
            else:
                batch_indices = np.arange(len(self.prediction_buffer))
            
            batch_data = [self.prediction_buffer[i] for i in batch_indices]

            # Prepare batch tensors：分别准备state和action（不再拼接）
            current_states = []
            action_encodings = []
            targets = []

            logger.info(f"=== Training StateTransitionModel with batch_size={batch_size} ===")
            for idx, sample_idx in enumerate(batch_indices):
                sample = batch_data[idx]
                current_states.append(sample['current_state'])
                action_encodings.append(sample['action_encoded'].astype(np.float32))
                targets.append(sample['next_state'])
                
                try:
                    decoded_action = self.codec.decode(sample['action_encoded'])
                except:
                    decoded_action = "decode_failed"
                
                actual_delta = sample['next_state'] - sample['current_state']
                
                logger.info(f"  Sample {idx+1}/{batch_size} (buffer_idx={sample_idx}): "
                           f"current_state={sample['current_state']}, "
                           f"action={decoded_action}, "
                           f"target_next_state={sample['next_state']}, "
                           f"actual_delta={actual_delta}")

            # 转换为tensor
            current_states_tensor = torch.from_numpy(np.array(current_states)).float().to(self.device)  # [batch_size, state_dim]
            action_encodings_tensor = torch.from_numpy(np.array(action_encodings)).float().to(self.device)  # [batch_size, action_dim]
            targets_tensor = torch.from_numpy(np.array(targets)).float().to(self.device)  # [batch_size, state_dim]

            # Training step：使用新的模型接口
            self.prediction_model.train()
            predictions = self.prediction_model(current_states_tensor, action_encodings_tensor)
            loss = self.prediction_loss_fn(predictions, targets_tensor)

            self.prediction_optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪：防止梯度爆炸，提高训练稳定性
            torch.nn.utils.clip_grad_norm_(self.prediction_model.parameters(), max_norm=1.0)
            
            self.prediction_optimizer.step()

            self.prediction_model.eval()
            
            predictions_np = predictions.detach().cpu().numpy()
            targets_np = targets_tensor.cpu().numpy()
            
            logger.info(f"Training completed - Loss: {loss.item():.6f}, Buffer size: {len(self.prediction_buffer)}")
            logger.info(f"  Predictions stats: mean={predictions_np.mean():.3f}, std={predictions_np.std():.3f}, "
                       f"min={predictions_np.min():.3f}, max={predictions_np.max():.3f}")
            logger.info(f"  Targets stats: mean={targets_np.mean():.3f}, std={targets_np.std():.3f}, "
                       f"min={targets_np.min():.3f}, max={targets_np.max():.3f}")
            logger.info(f"  Sample predictions vs targets:")
            for i in range(min(3, len(predictions_np))):  
                logger.info(f"    Sample {i+1}: pred={predictions_np[i]}, target={targets_np[i]}, "
                           f"diff={np.abs(predictions_np[i] - targets_np[i])}")

        except Exception as e:
            logger.warning(f"Failed to train prediction model batch: {e}")
