import time
import json
import threading
import subprocess
import socket
import struct
import logging
import os
import sys
import queue
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class ControllerLogger:
    """Dedicated logger for controller events"""

    def __init__(self, log_dir: Optional[str] = None, node_index: Optional[int] = None):
        self.log_dir = Path(log_dir)
        self.node_index = node_index
        # Use node-indexed filename if node_index is provided
        filename = f"controller-{node_index}.log" if node_index is not None else "controller.log"
        self.controller_log_path = self.log_dir / filename
        self._buffer: list[str] = []
        self._buffer_size = 1  # Flush every event for immediate visibility
        self._ensure_log_dir()
        self._ensure_controller_log_file()

    def _ensure_log_dir(self):
        """Ensure log directory exists"""
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def _ensure_controller_log_file(self):
        """Ensure controller log file exists"""
        try:
            if not self.controller_log_path.exists():
                self.controller_log_path.touch()
                print(f"Created controller log file: {self.controller_log_path}")
        except Exception as e:
            logger.warning(f"Failed to create controller log file: {e}")

    def log_event(self, event_type: str, details: Dict[str, Any], timestamp: Optional[str] = None):
        """Log a controller event"""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + 'Z'

        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'details': details
        }

        # Convert to JSON line
        json_line = json.dumps(event)
        self._buffer.append(json_line)

        # Flush if buffer is full
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self):
        """Flush buffered events to file"""
        if not self._buffer:
            return

        try:
            self._ensure_controller_log_file()
            with open(self.controller_log_path, 'a', encoding='utf-8') as f:
                for line in self._buffer:
                    f.write(line + '\n')
            self._buffer.clear()
        except Exception as e:
            logger.warning(f"Failed to flush controller log: {e}")

    def flush(self):
        """Force flush remaining events"""
        self._flush()

    def __del__(self):
        """Ensure buffer is flushed on destruction"""
        self.flush()


def get_controller_logger(node_index: Optional[int] = None, log_dir: Optional[str] = None) -> ControllerLogger:
    """Create a new controller logger instance"""
    return ControllerLogger(log_dir=log_dir, node_index=node_index)

class AutopilotController:
    """
    Autopilot system controller, responsible for applying parameters and retrieving metrics
    """

    def __init__(
        self,
        metrics_dir: str,
        parameters_file: str,
        node_index: Optional[int] = None,
        log_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        rl_algo: str = "cmab",
    ):
        """
        Initialize controller

        Args:
            metrics_dir: metrics file directory
            parameters_file: parameter file path (.parameters.json)
            node_index: node index for logging
            log_dir: log directory
            resume_from: optional policy checkpoint path
            rl_algo: "cmab" (RF-TS) or "gp_bo" (GP-UCB Bayesian Optimization)
        """
        self.metrics_dir = Path(metrics_dir)
        self.parameters_file = Path(parameters_file)
        self.node_index = node_index
        self.log_dir = Path(log_dir)
        self.resume_from = resume_from
        self.rl_algo = (rl_algo or "cmab").lower()
        if self.rl_algo not in ("cmab", "gp_bo"):
            raise ValueError(f"Unsupported rl_algo: {self.rl_algo}")
        # Agent/rl root (parent of controllers/)
        self._rl_root = Path(__file__).resolve().parent.parent

        # Initialize controller logger
        self.logger = get_controller_logger(node_index=node_index, log_dir=log_dir)
        self.embedded_trainer = None
        self.training_process = None
        self.training_thread = None
        self.training_data_queue = None
        self.training_active = False
        self.training_initialized = False
        # Step counting for training frequency control (match env.py behavior)
        self.training_step_count = 0
        self.training_update_frequency = 1  # Update training every epoch/step

        # Ensure metrics directory exists
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self._start_continuous_training()

    def _is_training_active(self) -> bool:
        """Check if training is currently active"""
        return self.training_active and (self.training_process is not None or self.embedded_trainer is not None)

    def _set_training_inactive(self):
        """Set training as inactive"""
        self.training_active = False

    def _start_continuous_training(self):
        """Start embedded continuous RL training that runs throughout the Autopilot execution"""
        if self._is_training_active():
            logger.info("Continuous training already active")
            return

        logger.info("Starting continuous RL training (algo=%s)...", self.rl_algo)

        self._start_continuous_training_subprocess()

    def _training_script_name(self) -> str:
        if self.rl_algo == "gp_bo":
            return "train_gp_bo.py"
        return "train_cmab_continuous.py"

    def _checkpoint_dir(self) -> Path:
        if self.rl_algo == "gp_bo":
            return Path("/home/ccclr0302/gp_bo_checkpoints")
        return Path("/home/ccclr0302/checkpoints")

    def _start_continuous_training_subprocess(self):
        logger.info("Starting continuous training subprocess (fallback mode)...")

        # Create training log file
        training_log_file = self.log_dir / f"continuous_training_{self.node_index}.log"
        logger.info(f"Continuous training output will be logged to: {training_log_file}")

        try:
            train_script = self._rl_root / self._training_script_name()
            checkpoint_dir = self._checkpoint_dir()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Start training script in continuous mode
            cmd = [
                sys.executable,
                str(train_script),
                "--node-index", str(self.node_index),
                "--metrics-dir", str(self.metrics_dir),
                "--parameters-file", str(self.parameters_file),
                "--checkpoint-dir", str(checkpoint_dir),
            ]
            if self.resume_from:
                cmd.extend(["--resume-from", str(self.resume_from)])

            logger.info(f"Starting continuous training with command: {' '.join(cmd)}")

            # Start training process (non-blocking, continuous)
            # Note: We don't use start_new_session=True to avoid zombie processes
            with open(training_log_file, 'a', buffering=1) as logfile:
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=str(self._rl_root),
                    # Don't use start_new_session=True - monitor thread will wait for it
                )

            logger.info(f"Continuous training subprocess started with PID: {self.training_process.pid}")
            self.training_active = True

            # Start background thread to monitor training process
            self.training_thread = threading.Thread(target=self._monitor_continuous_training, daemon=True)
            self.training_thread.start()

            logger.info("Continuous training subprocess started successfully")

        except Exception as e:
            logger.error(f"Failed to start continuous training subprocess: {e}")
            self.training_active = False

    def _monitor_continuous_training(self):
        """Monitor the continuous training process"""
        try:
            # Wait for process to complete (which should be never in continuous mode)
            return_code = self.training_process.wait()

            self._set_training_inactive()

            if return_code == 0:
                logger.info("  Continuous training completed normally")
            else:
                logger.error(f"  Continuous training failed with return code {return_code}")

        except Exception as e:
            logger.error(f"  Failed to monitor continuous training: {e}")
            self._set_training_inactive()

    def cleanup_training(self):
        """Cleanup training resources"""
        logger.info("Cleaning up controller training resources...")
        self.logger.log_event('controller_cleanup_start', {
            'has_embedded_trainer': self.embedded_trainer is not None,
            'has_training_process': self.training_process is not None
        })

        self.training_active = False

        if self.embedded_trainer:
            try:
                self.embedded_trainer.cleanup()
                logger.info("Embedded trainer cleaned up")
                self.logger.log_event('embedded_trainer_cleanup', {'status': 'success'})
            except Exception as e:
                logger.warning(f"Failed to cleanup embedded trainer: {e}")
                self.logger.log_event('embedded_trainer_cleanup', {
                    'status': 'failed',
                    'error': str(e)
                })

        if self.training_process:
            try:
                self.training_process.terminate()
                self.training_process.wait(timeout=5)
                logger.info("Training subprocess terminated")
                self.logger.log_event('training_process_cleanup', {'status': 'success'})
            except Exception as e:
                logger.warning(f"Failed to terminate training subprocess: {e}")
                self.logger.log_event('training_process_cleanup', {
                    'status': 'failed',
                    'error': str(e)
                })
                try:
                    self.training_process.kill()
                except:
                    pass

    def __del__(self):
        """Destructor - cleanup resources"""
        self.cleanup_training()


def main():
    """Main function to run controller as standalone server"""
    import argparse
    import signal
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Autopilot RL Controller Server')
    parser.add_argument('--metrics-dir', type=str, required=True,
                       help='Directory for metrics files')
    parser.add_argument('--parameters-file', type=str, required=True,
                       help='Path to parameters JSON file')
    parser.add_argument('--slots-per-epoch', type=int, default=20,
                       help='Number of slots per epoch')
    parser.add_argument('--node-index', type=int, default=None,
                       help='Node index for logging')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Log directory')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume RL policy from checkpoint path')
    parser.add_argument('--rl-algo', type=str, default='cmab',
                       choices=['cmab', 'gp_bo'],
                       help='RL algorithm: cmab (RF-TS) or gp_bo (GP-UCB)')

    args = parser.parse_args()

    print("🚀 Starting Autopilot RL Controller Server")
    print(f"📊 Metrics dir: {args.metrics_dir}")
    print(f"⚙️  Parameters file: {args.parameters_file}")
    print(f"🏷️  Node index: {args.node_index}")
    print(f"📝 Log dir: {args.log_dir}")
    print(f"🧠 RL algo: {args.rl_algo}")
    print(f"🔁 Resume from: {args.resume_from}")

    # Logger will be initialized by AutopilotController

    # Initialize controller
    try:
        controller = AutopilotController(
            metrics_dir=args.metrics_dir,
            parameters_file=args.parameters_file,
            node_index=args.node_index,
            log_dir=args.log_dir,
            resume_from=args.resume_from,
            rl_algo=args.rl_algo,
        )
        print("✅ Controller initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize controller: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Signal handler for graceful shutdown
    shutdown_requested = False
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        print("\n🛑 Shutting down controller server...")
        shutdown_requested = True
        controller.cleanup_training()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep server running
    print("🔄 Controller server running (press Ctrl+C to stop)...")
    try:
        while not shutdown_requested:
            time.sleep(1)  # Keep alive
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt, shutting down...")
    finally:
        print("🛑 Cleaning up controller resources...")
        controller.cleanup_training()
        print("✅ Controller server stopped.")


if __name__ == "__main__":
    main()