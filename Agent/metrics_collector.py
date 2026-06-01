#!/usr/bin/env python3

"""
Metrics Collector with Optimized Log Reading

This module implements a metrics collector for blockchain consensus systems with
optimized log file reading to handle large, continuously growing log files efficiently.
"""

import os
import re
import json
import time
import math
import psutil
import numpy as np
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
import sys
import os

# Add the project root to sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from benchmark.benchmark.logs import LogParser
except ImportError:
    # Fallback for when running as a script
    sys.path.insert(0, os.path.join(project_root, 'benchmark'))
    from benchmark.logs import LogParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def _sanitize_floats_for_json(value: Any) -> Any:
    """Recursively sanitize non-finite floats to 0.0 for JSON serialization."""
    if isinstance(value, dict):
        return {k: _sanitize_floats_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_floats_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_floats_for_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        normalized = float(value)
        return normalized if math.isfinite(normalized) else 0.0
    return value

class MetricsLogger:
    """Dedicated logger for metrics events to reduce I/O overhead"""

    def __init__(self, log_dir: str, node_index: Optional[int] = None):
        self.log_dir = Path(log_dir)
        # Use provided node_index, or try to detect it, or default to 0
        self.node_index = node_index
        # Use node-indexed filename if node_index is provided
        filename = f"metrics-0.log" if node_index is not None else "metrics.log"
        self.metrics_log_path = self.log_dir / filename
        self._buffer: List[str] = []
        self._buffer_size = 1  # Flush every event for immediate visibility
        self._ensure_log_dir()
        self._ensure_metrics_log_file()

    def _ensure_log_dir(self):
        """Ensure log directory exists"""
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def _ensure_metrics_log_file(self):
        """Ensure metrics log file exists"""
        try:
            # Create the file if it doesn't exist
            if not self.metrics_log_path.exists():
                self.metrics_log_path.touch()
                print(f"Created metrics log file: {self.metrics_log_path}")
        except Exception as e:
            logger.warning(f"Failed to create metrics log file: {e}")

    def log_event(self, event_type: str, details: Dict[str, Any], timestamp: Optional[str] = None):
        """Log a metrics event"""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.utcnow().isoformat() + 'Z'

        event = MetricsEvent(
            timestamp=timestamp,
            event_type=event_type,
            details=details
        )

        # Convert to JSON line
        json_line = json.dumps(_sanitize_floats_for_json(asdict(event)))
        self._buffer.append(json_line)

        # Flush if buffer is full
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self):
        """Flush buffered events to file"""
        if not self._buffer:
            return

        try:
            # Ensure the metrics log file exists before writing
            self._ensure_metrics_log_file()
            with open(self.metrics_log_path, 'a', encoding='utf-8') as f:
                for line in self._buffer:
                    f.write(line + '\n')
            self._buffer.clear()
        except Exception as e:
            logger.warning(f"Failed to flush metrics log: {e}")

    def flush(self):
        """Force flush remaining events"""
        self._flush()

    def __del__(self):
        """Ensure buffer is flushed on destruction"""
        self.flush()

# Global metrics logger instance
_metrics_logger: Optional[MetricsLogger] = None
_metrics_logger_node_index: Optional[int] = None

def get_metrics_logger(log_dir: str, node_index: int) -> MetricsLogger:
    """Get or create global metrics logger"""
    global _metrics_logger, _metrics_logger_node_index
    if _metrics_logger is None or (_metrics_logger_node_index != node_index and node_index is not None):
        _metrics_logger = MetricsLogger(log_dir=log_dir, node_index=node_index)
        _metrics_logger_node_index = node_index
    return _metrics_logger

@dataclass
class HardwareCapacity:
    """Layer 1: Hardware capacity metrics"""
    cpu_cores: int
    memory_gb: float
    network_bandwidth_mbps: float
    workers_per_node: int

@dataclass
class NetworkCondition:
    """State 2: Network condition - 1D vector of latencies from self to other nodes"""
    latency_vector: List[float]  # latency from self to each other node in ms
    # bandwidth_vector: List[float]  # bandwidth from self to each other node in Mbps

@dataclass
class FastPathMetrics:
    """Layer 2 Part II: Fast path metrics"""
    fast_path_ratio: float
    slow_path_ratio: float

@dataclass
class Workload:
    """State 3: Workload characteristics"""
    tx_size_bytes: int
    tx_arrival_rate_tps: float

    @property
    def ingress_bps(self) -> float:
        return float(self.tx_size_bytes) * float(self.tx_arrival_rate_tps) * 8.0

@dataclass
class LaneVector:
    """State 4: Lane vector - height growth rates for each lane"""
    growth_rates: Dict[str, float]  # {validator_pk: height_per_second}

@dataclass
@dataclass
class ConsensusMetrics:
    """End-to-end throughput and latency metrics for the previous epoch"""
    end_to_end_tps: float
    end_to_end_bps: float
    end_to_end_latency_ms: float

@dataclass
class MetricsEvent:
    """Metrics event for dedicated logging"""
    timestamp: str  # ISO format
    event_type: str  # 'committee', 'prepare', 'commit', 'workload', 'fast_path', 'slow_path'
    details: Dict[str, Any]

@dataclass
class SystemState:
    """Complete system state with 5 components"""
    timestamp: float
    window_duration: float
    # State 1: Hardware configurations
    hardware: HardwareCapacity
    # State 2: Network condition (1D vector of latencies)
    network: NetworkCondition
    # State 3: Workload
    workload: Workload
    # State 4: Lane vector
    lane_vector: LaneVector
    # State 5: Fast path ratio
    fast_path_ratio: float
    # Additional metrics: End-to-end throughput and latency for previous epoch
    consensus_metrics: ConsensusMetrics

class MetricsCollector:
    """Main class for metrics/parameter collection"""

    def __init__(self, log_dir: str, node_index: Optional[int] = None, parameters_file: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.node_index = node_index
        logger.info(f"📁 Using log directory: {self.log_dir}")
        logger.info(f"🔢 Using node index: {self.node_index}")

        # Keep metrics output path consistent with controller's --metrics-dir.
        # Use home directory to avoid repo directory ownership/permission issues.
        self.output_dir = Path.home() / f"metrics-{self.node_index}"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.parameters_file = Path(parameters_file) if parameters_file else None

        # Log caching mechanism to improve performance
        self._log_cache: Dict[str, List[str]] = {}  # {filename: lines}
        self._log_file_stats: Dict[str, Dict[str, Any]] = {}  # {filename: {'mtime': float, 'size': int, 'inode': int}}
        self._log_file_positions: Dict[str, int] = {}  # {filename: last_read_position}
        self._cache_max_age_seconds = 30  # Cache expires after 30 seconds
        self._cache_enabled = True
        self._incremental_reading_enabled = True
        
        # Track previous epoch's slot_end for reward calculation
        self.previous_epoch_window: Optional[int] = None  # slot_end from previous epoch

    def _read_metrics_events(self, event_types: Optional[List[str]] = None,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None,
                           slot_start: Optional[int] = None,
                           slot_end: Optional[int] = None,
                           max_lines: Optional[int] = None) -> List[MetricsEvent]:
        """Read metrics events from dedicated metrics log file with optimized IO

        Args:
            event_types: Filter by event types, None for all
            start_time: Filter events after this timestamp
            end_time: Filter events before this timestamp
            slot_start: Filter events with slot >= slot_start (inclusive)
            slot_end: Filter events with slot < slot_end (exclusive, right-open)
            max_lines: Maximum number of lines to read from end of file (for IO optimization)

        Returns:
            List of MetricsEvent objects
        """
        events = []

        # Read from dedicated metrics log only (no fallback)
        # Determine the correct metrics log path based on node_index
        if hasattr(self, 'node_index') and self.node_index is not None:
            filename = f"metrics-{self.node_index}.log"
        else:
            filename = "metrics.log"
        metrics_log_path = self.log_dir / filename
        if metrics_log_path.exists():
            try:
                # Optimize: If slot range is specified, use reverse reading to get recent data first
                # But only if we have a reasonable max_lines limit (to avoid reading entire file)
                use_reverse = slot_start is not None and max_lines is not None and max_lines < 50000
                if use_reverse:
                    lines = self._read_log_file_reverse(metrics_log_path, max_lines)
                    # Don't reverse - we want to process from newest to oldest
                    # But we need to collect all events in range, so process normally
                    lines_to_process = lines
                else:
                    # Use cached reading for full file or when max_lines is too large
                    lines = self._read_log_file(metrics_log_path)
                    lines_to_process = lines
                
                for line in lines_to_process:
                    try:
                        event_dict = json.loads(line.strip())
                        # Remove node_index if present (for backward compatibility with old logs)
                        event_dict.pop('node_index', None)
                        event = MetricsEvent(**event_dict)

                        # Early filter by slot range (before parsing timestamp for performance)
                        if slot_start is not None or slot_end is not None:
                            event_slot = event.details.get('slot')
                            if event_slot is not None:
                                # Check if slot is in range
                                if slot_start is not None and event_slot < slot_start:
                                    continue
                                if slot_end is not None and event_slot >= slot_end:
                                    continue

                        # Apply event type filter
                        if event_types and event.event_type not in event_types:
                            continue

                        # Convert timestamp to float for comparison
                        from datetime import datetime
                        # Handle both ISO format with Z and +00:00 timezone
                        ts_str = event.timestamp
                        if ts_str.endswith('Z'):
                            ts_str = ts_str[:-1] + '+00:00'

                        # Truncate microseconds to 6 digits (Python datetime limit)
                        if '.' in ts_str:
                            parts = ts_str.split('.')
                            if len(parts) == 2 and len(parts[1]) > 6:
                                # Find timezone part
                                tz_start = -1
                                for i, c in enumerate(parts[1]):
                                    if c in '+-':
                                        tz_start = i
                                        break
                                if tz_start > 0:
                                    microseconds = parts[1][:tz_start]
                                    timezone = parts[1][tz_start:]
                                    if len(microseconds) > 6:
                                        microseconds = microseconds[:6]
                                    ts_str = f"{parts[0]}.{microseconds}{timezone}"
                                else:
                                    # No timezone, truncate microseconds
                                    microseconds = parts[1][:6]
                                    ts_str = f"{parts[0]}.{microseconds}"

                        event_ts = datetime.fromisoformat(ts_str).timestamp()

                        if start_time and event_ts < start_time:
                            continue
                        if end_time and event_ts > end_time:
                            continue

                        events.append(event)
                    except (json.JSONDecodeError, ValueError):
                        continue  # Skip malformed lines

            except Exception as e:
                logger.warning(f"Failed to read metrics log: {e}")

        return events

    def _read_log_file(self, filepath: Path) -> List[str]:
        """Read log file directly without caching

        Args:
            filepath: Path to the log file

        Returns:
            List of lines from the log file
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        except (OSError, IOError) as e:
            logger.warning(f"Failed to read log file {filepath}: {e}")
            return []

    def _read_log_file_reverse(self, filepath: Path, max_lines: int = 10000) -> List[str]:
        """Read log file from end, returning only the last N lines (for IO optimization)
        
        This is much faster for large files when we only need recent data.
        
        Args:
            filepath: Path to the log file
            max_lines: Maximum number of lines to read from end
            
        Returns:
            List of lines from the end of the file
        """
        try:
            with open(filepath, 'rb') as f:
                # Seek to end
                f.seek(0, 2)
                file_size = f.tell()
                
                # Estimate bytes per line (rough estimate: 200 bytes per line)
                # Read a larger chunk to ensure we get enough lines
                chunk_size = min(max_lines * 500, file_size)
                
                # Read from end
                f.seek(max(0, file_size - chunk_size))
                content = f.read()
                
            # Decode and split
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                content_str = content.decode('latin-1')
            
            lines = content_str.splitlines()
            
            # Return last max_lines
            return lines[-max_lines:] if len(lines) > max_lines else lines
            
        except (OSError, IOError) as e:
            logger.warning(f"Failed to read log file in reverse {filepath}: {e}")
            return []

    def _get_node_count(self) -> int:
        """Get the total number of nodes from primary logs"""
        node_mapping = self._parse_committee_from_logs()
        return len(node_mapping) if node_mapping else 0
        
    def _measure_interface_bandwidth(self) -> float:
        """Measure network interface bandwidth capacity using system tools"""
        try:
            # Install ethtool if not available
            ethtool_path = '/usr/sbin/ethtool'
            ip_path = '/usr/sbin/ip'
            try:
                import subprocess
                subprocess.run([ethtool_path, '--version'], capture_output=True, text=True, timeout=10)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info("Installing ethtool...")
                subprocess.run(['sudo', 'apt-get', 'update'], capture_output=True, text=True, timeout=30)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ethtool'], capture_output=True, text=True, timeout=60)
                logger.info("ethtool installed successfully")

            # # Method 1: Try different interface names (eth0, ens4, enp0s3, etc.)
            # interfaces_to_try = ['eth0', 'ens4', 'enp0s3', 'enp0s4', 'eno1']
            # for interface in interfaces_to_try:
            #     try:
            #         result = subprocess.run([ethtool_path, interface], capture_output=True, text=True, timeout=10)
            #         if result.returncode == 0:
            #             # Parse speed from ethtool output
            #             for line in result.stdout.split('\n'):
            #                 if 'Speed:' in line:
            #                     speed_str = line.split('Speed:')[1].strip()
            #                     logger.info(f"Found interface {interface} with speed: {speed_str}")
            #                     if 'Mb/s' in speed_str:
            #                         speed_mbps = float(speed_str.replace('Mb/s', '').strip())
            #                         return speed_mbps
            #                     elif 'Gb/s' in speed_str:
            #                         speed_gbps = float(speed_str.replace('Gb/s', '').strip())
            #                         return speed_gbps * 1000
            #     except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            #         continue

            # # Method 2: Use ip link show to find active interfaces and their speeds
            # try:
            #     result = subprocess.run([ip_path, 'link', 'show'], capture_output=True, text=True, timeout=10)
            #     if result.returncode == 0:
            #         lines = result.stdout.split('\n')
            #         for i, line in enumerate(lines):
            #             if 'state UP' in line:
            #                 # Extract interface name
            #                 if ':' in line:
            #                     interface = line.split(':')[1].strip()
            #                     # Look for speed info in next few lines
            #                     for j in range(i, min(i+5, len(lines))):
            #                         if 'speed' in lines[j].lower():
            #                             speed_line = lines[j].strip()
            #                             # Parse speed (might be in format like "1000Mb/s" or "1Gb/s")
            #                             import re
            #                             speed_match = re.search(r'speed\s+(\d+)(Mb/s|Gb/s)', speed_line)
            #                             if speed_match:
            #                                 speed_val = float(speed_match.group(1))
            #                                 speed_unit = speed_match.group(2)
            #                                 if speed_unit == 'Gb/s':
            #                                     return speed_val * 1000
            #                                 else:
            #                                     return speed_val
            # except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            #     pass

            # # Method 3: Check /sys/class/net for interface speeds
            # try:
            #     import glob
            #     for interface_path in glob.glob('/sys/class/net/*/speed'):
            #         try:
            #             with open(interface_path, 'r') as f:
            #                 speed_mbps = int(f.read().strip())
            #                 if speed_mbps > 0:
            #                     logger.info(f"Found interface speed from sysfs: {speed_mbps} Mbps")
            #                     return float(speed_mbps)
            #         except (IOError, ValueError):
            #             continue
            # except Exception:
            #     pass

            # Method 4: For cloud environments, use known bandwidth values based on instance type
            try:
                # Check for GCP and get instance type
                result = subprocess.run(['curl', '-s', '-H', 'Metadata-Flavor: Google',
                                       'http://metadata.google.internal/computeMetadata/v1/instance/machine-type'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and 'machineTypes' in result.stdout:
                    machine_type = result.stdout.strip().split('/')[-1]
                    logger.info(f"Detected GCP instance type: {machine_type}")

                    # GCP machine type to bandwidth mapping (approximate)
                    gcp_bandwidth_map = {
                        't2d-standard': 10000,  # T2D instances can burst to 10Gbps
                        'n2-standard': 10000,
                        'c2-standard': 10000,
                        'e2-standard': 2000,
                    }

                    for prefix, bandwidth in gcp_bandwidth_map.items():
                        if machine_type.startswith(prefix):
                            logger.info(f"Using GCP {prefix} series bandwidth: {bandwidth} Mbps")
                            return float(bandwidth)

                    # Default for unknown GCP types
                    return 10000.0
            except:
                pass

            # Method 5: Try iperf3 to localhost as last resort (but this measures loopback, not real NIC)
            try:
                # Check if iperf3 is available
                subprocess.run(['iperf3', '--version'], capture_output=True, text=True, timeout=5)

                # Start iperf3 server in background
                server_proc = subprocess.Popen(['iperf3', '-s', '-D'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)  # Wait for server to start

                # Run client against localhost
                result = subprocess.run(['iperf3', '-c', '127.0.0.1', '-t', '3', '-J'],
                                      capture_output=True, text=True, timeout=10)
                server_proc.terminate()
                server_proc.wait()

                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    try:
                        bps = data["end"]["sum_received"]["bits_per_second"]
                        mbps = float(bps) / 1e6
                        logger.info(f"iperf3 localhost test gave: {mbps} Mbps (loopback speed)")
                        # In cloud environments, loopback can be very fast, but cap it at reasonable values
                        if 1000 < mbps < 50000:  # Between 1Gbps and 50Gbps
                            return mbps
                    except Exception:
                        pass
            except Exception:
                pass

            # Method 5: Check for AWS (GCP is handled in Method 4)
            try:
                result = subprocess.run(['curl', '-s', 'http://169.254.169.254/latest/meta-data/instance-type'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    instance_type = result.stdout.strip()
                    logger.info(f"Detected AWS instance type: {instance_type}")
                    # AWS instance types have varying network speeds, use conservative estimate
                    return 5000.0  # 5Gbps default for most AWS instances
            except:
                pass

            logger.warning("Could not determine network interface bandwidth, using default 1Gbps")
            return 1000.0  # Default 1Gbps

        except Exception as e:
            logger.warning(f"Failed to measure interface bandwidth: {e}, using default")
            return 1000.0  # Default 1Gbps

    def collect_hardware_capacity(self) -> HardwareCapacity:
        """Collect hardware capacity metrics"""
        try:
            # CPU core count
            cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
            
            # Memory capacity
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # Network bandwidth (interface speed via system tools)
            network_bandwidth_mbps = 10000
            
           
            
            # Workers per node (from parameter file)
            workers_per_node = 1
            if self.parameters_file.exists():
                with open(self.parameters_file, 'r') as f:
                    params = json.load(f)
                    workers_per_node = params.get('workers', 1)
            
            return HardwareCapacity(
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                network_bandwidth_mbps=network_bandwidth_mbps,
                workers_per_node=workers_per_node
            )
        except Exception as e:
            logger.error(f"Failed to collect hardware metrics: {e}")
            return HardwareCapacity(0, 0.0, 0.0, 1)
    
    def collect_network_condition(self) -> NetworkCondition:
        """Collect network condition as 1D vector - latencies and bandwidths from self to all other nodes"""
        try:
            node_count = self._get_node_count()

            # Get latency vector
            latency_vector_from_fab = self._call_fabfile_latency()
            if latency_vector_from_fab is None:
                logger.warning("No latency vector available, using default values")
                default_latency = [50.0] * (node_count - 1)
                default_bandwidth = [100.0] * (node_count - 1)  # Default 100 Mbps
                return NetworkCondition(latency_vector=default_latency)

            logger.info(f"Received latency vector of size {len(latency_vector_from_fab)}")

            # Process the latency vector: handle NaN values and ensure proper formatting
            latency_vector = []
            for latency in latency_vector_from_fab:
                # Handle edge cases: NaN, 0 (which might indicate measurement failure), or negative values
                if np.isnan(latency) or latency <= 0:
                    latency = 50.0  # Use reasonable default for failed measurements
                latency_vector.append(float(latency))
            return NetworkCondition(latency_vector=latency_vector)

        except Exception as e:
            logger.error(f"Failed to collect network condition: {e}")
            node_count = self._get_node_count()
            return NetworkCondition(
                latency_vector=[50.0] * (node_count - 1)
            )

            
    def _call_fabfile_latency(self) -> Optional[np.ndarray]:
        """Measure SSH latency between nodes directly (integrated version of fab latency)"""
        try:
            import time
            from fabric import Connection
            from paramiko import RSAKey, SSHException
            from invoke.exceptions import UnexpectedExit

            # Load SSH key
            try:
                ssh_key = RSAKey.from_private_key_file("/home/ccclr0302/.ssh/google_compute_engine")
                connect_kwargs = {'pkey': ssh_key}
            except (IOError, SSHException) as e:
                logger.warning(f"Failed to load SSH key: {e}")
                logger.warning("Falling back to default latency values (50ms)")
                node_count = self._get_node_count()
                return np.full(node_count - 1, 50.0)

            # Auto-detect current node from primary logs
            source_node, current_node_ip = self._parse_current_node_from_logs()
            if source_node is None:
                logger.warning("Could not determine current node from primary logs")
                logger.warning("Falling back to default latency values (50ms)")
                node_count = self._get_node_count()
                return np.full(node_count - 1, 50.0)

            logger.info(f"Auto-detected current node: {source_node} at {current_node_ip}")

            # Read committee info from logs
            node_mapping = self._parse_committee_from_logs()
            if not node_mapping:
                logger.warning("Failed to parse committee info from logs")
                logger.warning("Falling back to default latency values (50ms)")
                node_count = self._get_node_count()
                return np.full(node_count - 1, 50.0)

            logger.info(f"Total nodes in committee: {len(node_mapping)}")

            if current_node_ip not in node_mapping.values():
                logger.warning(f"Source node {source_node} not found in committee")
                logger.warning("Falling back to default latency values (50ms)")
                node_count = self._get_node_count()
                return np.full(node_count - 1, 50.0)

            # Get all target nodes (excluding self)
            target_nodes = [(name, ip) for name, ip in node_mapping.items() if ip != current_node_ip]
            if len(target_nodes) == 0:
                logger.warning("No target nodes found, falling back to default latency values (50ms)")
                node_count = self._get_node_count()
                return np.full(node_count - 1, 50.0)    
            logger.info(f"Will measure latency to {len(target_nodes)} other nodes")

            def ssh_latency(src_ip, dst_ip, repeat=1):
                """Measure SSH latency between two nodes"""
                if src_ip == dst_ip:
                    return 0.0
                results = []
                for _ in range(repeat):
                    try:
                        # Connect to destination node
                        conn = Connection(host=dst_ip, user="ccclr0302", connect_kwargs=connect_kwargs)
                        conn.run("echo warmup", hide=True, timeout=5)
                        start = time.time()
                        conn.run("echo hello", hide=True, timeout=5)
                        end = time.time()
                        conn.close()
                        results.append((end - start) * 1000)  # Convert to ms
                    except Exception as e:
                        logger.debug(f"SSH {src_ip} → {dst_ip} failed: {e}")
                        results.append(np.nan)
                valid = [x for x in results if not np.isnan(x)]
                return np.mean(valid) if valid else np.nan

            # Measure latency vector from source node to all others
            logger.info("=== Measuring Latency from Current Node ===")
            latency_vector = []
            measured_count = 0

            for target_name, target_ip in target_nodes:
                latency = ssh_latency(current_node_ip, target_ip)
                latency_vector.append(latency if latency else np.nan)

                if latency:
                    logger.info(f"  {source_node} → {target_name}: {latency:.2f} ms")
                    measured_count += 1
                else:
                    logger.info(f"  {source_node} → {target_name}: Failed")

            logger.info(f"Successfully measured latency to {measured_count}/{len(target_nodes)} nodes")

            # Calculate statistics
            valid_latencies = [lat for lat in latency_vector if not np.isnan(lat)]
            if valid_latencies:
                mean_lat = np.mean(valid_latencies)
                std_lat = np.std(valid_latencies)
                min_lat = np.min(valid_latencies)
                max_lat = np.max(valid_latencies)

                logger.info("=== Latency Statistics ===")
                logger.info(f"Mean Latency: {mean_lat:.2f} ms")
                logger.info(f"Std Deviation: {std_lat:.2f} ms")
                logger.info(f"Min Latency: {min_lat:.2f} ms")
                logger.info(f"Max Latency: {max_lat:.2f} ms")
                logger.info(f"Success Rate: {len(valid_latencies)}/{len(latency_vector)}")

            # Return latency vector as numpy array
            result_vector = np.array([x if not np.isnan(x) else np.nan for x in latency_vector])
            logger.info(f"Successfully measured latency vector: {len(result_vector)} values")
            return result_vector

        except Exception as e:
            logger.error(f"Failed to measure latency directly: {e}")
            logger.warning("Falling back to default latency values (50ms)")
            node_count = self._get_node_count()
            return np.full(node_count - 1, 50.0)

    def _parse_committee_from_logs(self) -> Dict[str, str]:
        """Parse committee information from metrics log files

        Returns:
            Dict mapping node names to IP addresses
        """
        import glob
        import json

        node_mapping = {}  # name -> ip

        # First try to read from metrics log files (new format)
        metrics_files = glob.glob(f'{self.log_dir}/metrics-{self.node_index}.log')
        if metrics_files:
            # Use the first metrics file
            metrics_file = metrics_files[0]
            try:
                with open(metrics_file, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            if event.get('event_type') == 'committee':
                                details = event.get('details', {})
                                pk = details.get('pk')
                                consensus_addr = details.get('consensus_addr')
                                if pk and consensus_addr:
                                    ip = consensus_addr.split(':')[0]  # Extract IP from "ip:port"
                                    node_mapping[pk] = ip
                        except json.JSONDecodeError:
                            continue
                if node_mapping:
                    logger.info(f"Parsed committee from metrics log: {len(node_mapping)} nodes")
                    return node_mapping
            except Exception as e:
                logger.warning(f"Failed to parse committee from metrics log {metrics_file}: {e}")

        return node_mapping

    def _parse_current_node_from_logs(self) -> Tuple[Optional[str], Optional[str]]:
        """Parse current node information from metrics log files

        Reads the (node_count + 1)th line from metrics-{node_index}.log to get node info.

        Returns:
            Tuple of (node_name, node_ip) or (None, None) if not found
        """
        import json

        try:
            # Read the metrics file for this node and find the node_info event
            metrics_file = self.log_dir / f"metrics-{self.node_index}.log"
            if not metrics_file.exists():
                logger.warning(f"Metrics file not found: {metrics_file}")
                return None, None

            with open(metrics_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                        if event.get('event_type') == 'node_info':
                            details = event.get('details', {})
                            node_name = details.get('public_key')
                            node_ip = details.get('ip_address')

                            if node_name and node_ip:
                                logger.info(f"Parsed current node info from line {line_num}: name={node_name[:16]}..., ip={node_ip}")
                                return node_name, node_ip
                            else:
                                logger.warning(f"Missing public_key or ip_address in node_info event on line {line_num}")

                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

            logger.warning("No valid node_info event found in metrics file")

        except Exception as e:
            logger.warning(f"Failed to parse current node from metrics log: {e}")

        return None, None

    def collect_workload_for_window(self, window_start_time: float, window_end_time: float, slot_start: Optional[int] = None, slot_end: Optional[int] = None) -> Workload:
        """Collect workload for a specific time window by analyzing client transaction sending logs.
        
        Strategy: Match committed transactions in [slot_start, slot_end] with client sending logs
        1. Get transaction IDs that were committed in this slot range from metrics log
        2. Find these transaction IDs in client log to get their sending timestamps
        3. Calculate the sending rate based on actual sending times

        Args:
            window_start_time: Start timestamp of the window (fallback if slot range not available)
            window_end_time: End timestamp of the window (fallback if slot range not available)
            slot_start: Start slot for transaction matching (preferred method)
            slot_end: End slot for transaction matching (preferred method)
        """
        try:
            # Read all client log files for the current node (client-<node_index>-*.log)
            import glob
            client_log_pattern = str(self.log_dir / f"client-{self.node_index}-*.log")
            client_log_files = glob.glob(client_log_pattern)

            logger.info(f"Found {len(client_log_files)} client log files: {[Path(f).name for f in client_log_files]}")

            if not client_log_files:
                logger.warning("No client log files found, returning default workload")
                return Workload(tx_size_bytes=512, tx_arrival_rate_tps=0.0)

            tx_size_bytes = 512  # Default
            actual_rate_tps = 0.0

            # NEW STRATEGY: Use slot range to get committed transaction IDs from metrics log
            if slot_start is not None and slot_end is not None:
                logger.info(f"🎯 Using slot-based transaction matching for workload calculation")
                logger.info(f"   Slot range: [{slot_start}, {slot_end}]")
                
                # Step 1: Get transaction IDs that were committed in this slot range from metrics log
                committed_tx_ids = set()
                try:
                    tx_commit_events = self._read_metrics_events(
                        event_types=["transaction_commit"],
                        slot_start=slot_start,
                        slot_end=slot_end,
                        max_lines=None
                    )
                    
                    for event in tx_commit_events:
                        tx_id = event.details.get('transaction_id')
                        if tx_id is not None:
                            committed_tx_ids.add(tx_id)
                    
                    logger.info(f"   Found {len(committed_tx_ids)} committed transactions in slot range")
                    if committed_tx_ids:
                        logger.info(f"   Transaction ID range: [{min(committed_tx_ids)}, {max(committed_tx_ids)}]")
                
                except Exception as e:
                    logger.warning(f"Failed to read transaction_commit events from metrics log: {e}")
                    committed_tx_ids = set()
                
                # Step 2: Find these transaction IDs in client log and get their sending times
                if committed_tx_ids:
                    matched_transactions = []  # (timestamp, tx_id)
                    
                    for client_log_file in client_log_files:
                        try:
                            with open(client_log_file, 'r') as f:
                                for line in f:
                                    # Parse transaction size
                                    if "Transactions size" in line:
                                        tx_size_bytes = int(line.split("Transactions size: ")[1].split(" ")[0])
                                        continue
                                    
                                    # Find transactions that match our committed IDs
                                    if "Sending sample transaction" in line:
                                        try:
                                            # Extract transaction ID
                                            tx_match = re.search(r'Sending sample transaction (\d+)', line)
                                            if tx_match:
                                                tx_id = int(tx_match.group(1))
                                                
                                                # Only process if this transaction was committed in our slot range
                                                if tx_id in committed_tx_ids:
                                                    # Extract timestamp
                                                    timestamp_match = re.search(r'\[([^\]]+)Z', line)
                                                    if timestamp_match:
                                                        timestamp_str = timestamp_match.group(1) + '+00:00'
                                                        from datetime import datetime
                                                        dt = datetime.fromisoformat(timestamp_str)
                                                        timestamp = dt.timestamp()
                                                        matched_transactions.append((timestamp, tx_id))
                                        except Exception as e:
                                            logger.debug(f"Failed to parse client log line: {line.strip()} - {e}")
                                            continue
                        except Exception as e:
                            logger.warning(f"Failed to read client log file {client_log_file}: {e}")
                            continue
                    
                    # Step 3: Calculate sending rate based on matched transactions
                    if len(matched_transactions) >= 2:
                        # Sort by timestamp
                        matched_transactions.sort(key=lambda x: x[0])
                        
                        # Calculate rate: total transactions / time span
                        first_time = matched_transactions[0][0]
                        last_time = matched_transactions[-1][0]
                        time_span = last_time - first_time
                        
                        if time_span > 0:
                            tx_ids = [tx_id for _, tx_id in matched_transactions]
                            first_tx_id = min(tx_ids)
                            last_tx_id = max(tx_ids)
                            total_transactions = last_tx_id - first_tx_id + 1
                            actual_rate_tps = total_transactions / time_span
                            
                            logger.info(f"  ✅ Matched {len(matched_transactions)} transactions from client log")
                            logger.info(f"  Transaction ID range: {first_tx_id} to {last_tx_id}")
                            logger.info(f"  Total transactions: {total_transactions}")
                            logger.info(f"  Sending time span: {time_span:.3f}s")
                            logger.info(f"  Calculated rate: {actual_rate_tps:.1f} tx/s")
                        else:
                            logger.warning(f"Time span is 0, cannot calculate rate")
                    elif len(matched_transactions) == 1:
                        logger.warning(f"Only 1 matched transaction found, cannot calculate rate")
                    else:
                        logger.warning(f"No matched transactions found in client log")
                    
                    # If we successfully calculated rate, return it
                    if actual_rate_tps > 0:
                        return Workload(
                            tx_size_bytes=max(1, tx_size_bytes),
                            tx_arrival_rate_tps=actual_rate_tps
                        )
            
            # FALLBACK: Calculate overall rate from all transactions in client log
            logger.info(f"📊 Fallback: Calculating overall rate from all client transactions")
            all_transactions = []
            
            for client_log_file in client_log_files:
                try:
                    with open(client_log_file, 'r') as f:
                        for line in f:
                            # Parse transaction size
                            if "Transactions size" in line:
                                tx_size_bytes = int(line.split("Transactions size: ")[1].split(" ")[0])
                                continue
                            
                            # Collect all transaction sending events
                            if "Sending sample transaction" in line:
                                try:
                                    # Extract timestamp
                                    timestamp_match = re.search(r'\[([^\]]+)Z', line)
                                    if timestamp_match:
                                        timestamp_str = timestamp_match.group(1) + '+00:00'
                                        from datetime import datetime
                                        dt = datetime.fromisoformat(timestamp_str)
                                        timestamp = dt.timestamp()

                                        # Extract transaction ID
                                        tx_match = re.search(r'Sending sample transaction (\d+)', line)
                                        if tx_match:
                                            tx_id = int(tx_match.group(1))
                                            all_transactions.append((timestamp, tx_id))

                                except Exception as e:
                                    logger.debug(f"Failed to parse client log line: {line.strip()} - {e}")
                                    continue
                except Exception as e:
                    logger.warning(f"Failed to read client log file {client_log_file}: {e}")
                    continue

            # Calculate overall transaction rate
            if len(all_transactions) >= 2:
                # Sort by timestamp
                all_transactions.sort(key=lambda x: x[0])

                # Calculate overall rate
                overall_start_time = all_transactions[0][0]
                overall_end_time = all_transactions[-1][0]
                overall_duration = overall_end_time - overall_start_time

                if overall_duration > 0:
                    tx_ids = [tx_id for _, tx_id in all_transactions]
                    first_tx_id = min(tx_ids)
                    last_tx_id = max(tx_ids)
                    total_transactions = last_tx_id - first_tx_id + 1
                    actual_rate_tps = total_transactions / overall_duration
                    
                    logger.info(f"  Overall transactions found: {len(all_transactions)}")
                    logger.info(f"  Transaction ID range: {first_tx_id} to {last_tx_id}")
                    logger.info(f"  Total transactions: {total_transactions}")
                    logger.info(f"  Overall duration: {overall_duration:.2f}s")
                    logger.info(f"  Calculated rate: {actual_rate_tps:.1f} tx/s")
                else:
                    logger.warning(f"Overall duration is 0, cannot calculate rate")
            else:
                logger.warning(f"Not enough transactions found in client log (found {len(all_transactions)})")

            return Workload(
                tx_size_bytes=max(1, tx_size_bytes),
                tx_arrival_rate_tps=max(0.0, actual_rate_tps)
            )

        except Exception as e:
            logger.warning(f"Parsing client logs for workload failed: {e}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")

        # Fallback
        return Workload(tx_size_bytes=512, tx_arrival_rate_tps=0.0)

    def collect_fast_path_metrics(self, slot_start: int, slot_end: int) -> FastPathMetrics:
        """Collect fast/slow path metrics for a slot range"""
        try:
            # Read fast path and slow path events from metrics log only
            fast_path_events = self._read_metrics_events(
                event_types=["fast_path"],
                slot_start=slot_start,
                slot_end=slot_end,
            )
            slow_path_events = self._read_metrics_events(
                event_types=["slow_path"],
                slot_start=slot_start,
                slot_end=slot_end,
            )

            fast_path_count = len(fast_path_events)
            slow_path_count = len(slow_path_events)

            logger.info(
                "Fast/slow path events in slots [%s, %s]: fast=%d slow=%d",
                slot_start,
                slot_end,
                fast_path_count,
                slow_path_count,
            )

            total_paths = fast_path_count + slow_path_count
            fast_path_ratio = fast_path_count / total_paths if total_paths > 0 else 0.0
            slow_path_ratio = slow_path_count / total_paths if total_paths > 0 else 0.0

            return FastPathMetrics(
                fast_path_ratio=float(fast_path_ratio),
                slow_path_ratio=float(slow_path_ratio)
            )
        except Exception as e:
            logger.error(f"Failed to collect fast path metrics: {e}")
            return FastPathMetrics(0.0, 0.0)


    def collect_epoch_throughput_latency(self, slot_start: int, slot_end: int) -> ConsensusMetrics:
        """Collect throughput and latency metrics for a specific slot range

        Uses new metrics log events first, falls back to LogParser if needed.

        Args:
            slot_start: Start slot number (inclusive)
            slot_end: End slot number (exclusive, right-open)

        Returns:
            ConsensusMetrics with throughput and latency for the specified slot range
        """
        try:
            logger.info("=" * 80)
            logger.info(f"=== Collecting epoch throughput and latency metrics ===")
            logger.info(f"📊 Slot Range: [{slot_start}, {slot_end}) (half-open, right-exclusive)")
            logger.info("=" * 80)

            # First try to collect from new metrics logs
            metrics_from_logs = self._collect_throughput_latency_from_metrics_logs(slot_start, slot_end)
            if metrics_from_logs:
                logger.info("Successfully collected metrics from dedicated metrics logs")
                return metrics_from_logs
            
            # If no metrics found, return default values
            logger.info("No metrics found, returning default values")
            return ConsensusMetrics(0.0, 0.0, 0.0)

        except Exception as e:
            logger.error(f"Failed to collect epoch throughput and latency: {e}")
            return ConsensusMetrics(0.0, 0.0, 0.0)

    def _collect_throughput_latency_from_metrics_logs(self, slot_start: int, slot_end: int) -> Optional[ConsensusMetrics]:
        """Collect throughput and latency metrics from dedicated metrics logs

        Uses the same calculation logic as benchmark/logs.py collect_realtime_metrics:
        1. Parse batch_commit events to get commits (batch_digest -> timestamp) and batch_sizes
        2. Parse transaction_commit events to get sample metrics (latencies by author)
        3. Compute metrics using traditional logic: duration from commit times, TPS = BPS / avg_transaction_size

        Args:
            slot_start: Start slot number (inclusive)
            slot_end: End slot number (inclusive)

        Returns:
            ConsensusMetrics if successful, None otherwise
        """
        try:
            # Step 1: Parse batch_commit events to get commits and batch_sizes (same as _parse_local_primary_commits and _parse_sample_transaction_metrics)
            # Optimize IO: Only read recent lines if slot range is near the end of the file
            # For slots 80-99, we need to read more data, so use cached reading for now
            # TODO: Implement smarter reverse reading that can handle mid-file ranges
            batch_commit_events = self._read_metrics_events(
                event_types=["batch_commit"],
                start_time=None,  # We'll filter by slot
                end_time=None,
                slot_start=slot_start,
                slot_end=slot_end,
                max_lines=None  # Use full cached read for now to ensure correctness
            )

            # Filter by slot range and build commits dict (batch_digest -> timestamp) and batch_sizes dict
            commits = {}  # batch_digest -> commit timestamp
            batch_sizes = {}  # batch_digest -> {'batch_size': int, 'avg_transaction_size': int}

            for event in batch_commit_events:
                slot = event.details.get('slot')
                if slot is not None and slot_start <= slot < slot_end:
                    # Use digest as batch_digest (same as benchmark: "Committed B<height>(...) -> <digest>=")
                    # This matches the batch digest in primary log format
                    batch_digest = event.details.get('digest')
                    if not batch_digest:
                        # Fallback to header_digest if digest not available
                        batch_digest = event.details.get('header_digest')
                    
                    if batch_digest:
                        # Parse commit timestamp
                        from datetime import datetime
                        try:
                            ts_str = event.timestamp
                            if ts_str.endswith('Z'):
                                ts_str = ts_str[:-1] + '+00:00'
                            elif ts_str.endswith('+00:00'):
                                pass
                            else:
                                # Try to parse as-is
                                pass
                            
                            # Truncate microseconds to 6 digits if needed
                            if '.' in ts_str:
                                parts = ts_str.split('.')
                                if len(parts) == 2:
                                    tz_start = -1
                                    for i, c in enumerate(parts[1]):
                                        if c in '+-':
                                            tz_start = i
                                            break
                                    if tz_start > 0:
                                        microseconds = parts[1][:tz_start]
                                        timezone = parts[1][tz_start:]
                                        if len(microseconds) > 6:
                                            microseconds = microseconds[:6]
                                        ts_str = f"{parts[0]}.{microseconds}{timezone}"
                                    elif len(parts[1]) > 6:
                                        microseconds = parts[1][:6]
                                        ts_str = f"{parts[0]}.{microseconds}"
                            
                            dt = datetime.fromisoformat(ts_str)
                            commit_timestamp = dt.timestamp()
                            commits[batch_digest] = commit_timestamp
                        except Exception as e:
                            logger.debug(f"Failed to parse timestamp for batch {batch_digest}: {e}")
                            continue

                        # Extract batch size information
                        batch_size = event.details.get('batch_size', 0)
                        transaction_count = event.details.get('transaction_count', 0)
                        avg_transaction_size = event.details.get('avg_transaction_size', 0)
                        
                        # Calculate avg_transaction_size if not provided
                        if avg_transaction_size == 0 and transaction_count > 0 and batch_size > 0:
                            avg_transaction_size = batch_size // transaction_count
                        elif avg_transaction_size == 0:
                            avg_transaction_size = 512  # Default

                        batch_sizes[batch_digest] = {
                            'batch_size': batch_size,
                            'avg_transaction_size': avg_transaction_size
                        }

            if not commits:
                print(f"No batch commit events found in slot range [{slot_start}, {slot_end}]")
                return None

            # Step 2: Parse transaction_commit events to get sample metrics (same as _parse_sample_transaction_metrics)
            # Optimize IO: Use cached reading for now
            tx_commit_events = self._read_metrics_events(
                event_types=["transaction_commit"],
                start_time=None,  # We'll filter by slot
                end_time=None,
                slot_start=slot_start,
                slot_end=slot_end,
                max_lines=None  # Use full cached read for now to ensure correctness
            )

            # Group sample metrics by author (same structure as benchmark)
            sample_metrics = {}  # author -> {'latencies': [int], 'sizes': [int], ...}

            for event in tx_commit_events:
                slot = event.details.get('slot')
                if slot is not None and slot_start <= slot < slot_end:
                    author = event.details.get('author')
                    if not author:
                        continue

                    if author not in sample_metrics:
                        sample_metrics[author] = {
                            'latencies': [],
                            'sizes': [],
                            'sample_count': 0
                        }

                    latency_us = event.details.get('e2e_latency_us', 0)
                    tx_size = event.details.get('tx_size_bytes', 512)

                    if latency_us > 0:
                        sample_metrics[author]['latencies'].append(latency_us)
                        sample_metrics[author]['sizes'].append(tx_size)
                        sample_metrics[author]['sample_count'] += 1

            # Step 3: Compute metrics using same logic as _compute_realtime_metrics_traditional_logic
            commit_times = list(commits.values())
            if len(commit_times) > 1:
                duration = max(commit_times) - min(commit_times)
            else:
                duration = 1.0  # fallback

            # Calculate total bytes from batch_sizes for committed batches only
            committed_batch_data = {digest: data for digest, data in batch_sizes.items() if digest in commits}
            total_bytes = sum(data['batch_size'] for data in committed_batch_data.values()) if committed_batch_data else 0

            # Calculate average transaction size from committed batches
            if committed_batch_data:
                avg_transaction_sizes = [data['avg_transaction_size'] for data in committed_batch_data.values() if data['avg_transaction_size'] > 0]
                transaction_size = sum(avg_transaction_sizes) // len(avg_transaction_sizes) if avg_transaction_sizes else 512
            else:
                transaction_size = 512

            # Calculate BPS and TPS (same as benchmark logic)
            bps = total_bytes / duration if duration > 0 else 0
            tps = bps / transaction_size if transaction_size > 0 else 0
            total_transactions = int(total_bytes / transaction_size) if transaction_size > 0 else 0

            # Calculate latency from sample metrics (same as benchmark logic)
            all_latencies = []
            for author_metrics in sample_metrics.values():
                all_latencies.extend(author_metrics['latencies'])

            avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
            avg_latency_ms = avg_latency / 1000.0  # Convert microseconds to milliseconds

            print(f"Metrics from dedicated logs (benchmark logic):")
            print(f"  Slot range: [{slot_start}, {slot_end}]")
            print(f"  Commits: {len(commits)}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Total bytes: {total_bytes:,}")
            print(f"  Avg transaction size: {transaction_size}")
            print(f"  Total transactions: {total_transactions:,}")
            print(f"  End-to-end TPS: {tps:.2f}")
            print(f"  End-to-end BPS: {bps:.2f}")
            print(f"  Average latency: {avg_latency_ms:.2f}ms")

            return ConsensusMetrics(
                end_to_end_tps=tps,
                end_to_end_bps=bps,
                end_to_end_latency_ms=avg_latency_ms
            )

        except Exception as e:
            logger.warning(f"Failed to collect from metrics logs: {e}")
            import traceback
            logger.warning(f"Exception details: {traceback.format_exc()}")
            return None

    def _collect_system_state_for_slot_window(self, slot_start: int, slot_end: int, end_to_end_epoch_idx: Optional[int] = None, epoch_slots: int = 50) -> Optional[SystemState]:
        """Collect system state for a specific slot window"""
        try:
            # Determine time window for this slot window
            # Use batch_commit events from metrics log to get accurate transaction timing
            window_start_time = float('inf')
            window_end_time = 0

            # Try to get time window from batch_commit events in metrics log (more accurate)
            try:
                batch_commit_events = self._read_metrics_events(
                    event_types=["batch_commit"],
                    slot_start=slot_start,
                    slot_end=slot_end,
                    max_lines=None
                )
                
                if batch_commit_events:
                    from datetime import datetime
                    for event in batch_commit_events:
                        try:
                            # Parse timestamp
                            ts_str = event.timestamp
                            if not ts_str.endswith('Z'):
                                ts_str = ts_str.replace('+00:00', 'Z')
                            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                            ts = dt.timestamp()
                            window_start_time = min(window_start_time, ts)
                            window_end_time = max(window_end_time, ts)
                        except Exception as e:
                            continue
                    
                    if window_start_time != float('inf'):
                        logger.info(f"📅 Time window from batch_commit events: [{window_start_time:.3f}, {window_end_time:.3f}] ({window_end_time - window_start_time:.3f}s)")
            except Exception as e:
                logger.debug(f"Failed to get time window from batch_commit events: {e}")

            # State 1: Hardware configurations (fixed)
            hardware = self.collect_hardware_capacity()

            # State 2: Network condition (fixed for simplicity)
            # network = self.collect_network_condition()
            network = NetworkCondition(latency_vector=[50.0, 50.0, 50.0])

            # State 3: Workload for this specific time window
            # Pass slot_start and slot_end for precise transaction matching
            workload = self.collect_workload_for_window(window_start_time, window_end_time, slot_start, slot_end)

            # State 4: Lane vector (height growth rates within this slot window)
            lane_vector_data = self._get_lane_vector_for_slot_window(slot_start, slot_end)
            lane_vector = LaneVector(growth_rates=lane_vector_data)

            # State 5: Fast path ratio (fixed for now)
            fast_path_metrics = self.collect_fast_path_metrics(slot_start, slot_end)
            fast_path_ratio = fast_path_metrics.fast_path_ratio

            # Additional metrics: End-to-end throughput and latency (reward)
            # Strategy: Use (previous_epoch_window + 3, current_slot_end] for reward calculation
            # This represents the performance after applying the configuration from previous epoch
            if self.previous_epoch_window is None:
                # For epoch 0 or when no previous data exists, use empty metrics
                epoch_consensus_metrics = ConsensusMetrics(0.0, 0.0, 0.0)
                logger.info(f"Epoch {end_to_end_epoch_idx}: Using empty end-to-end metrics (no previous epoch window)")
            else:
                reward_start_slot = slot_start
                reward_end_slot = slot_end
                logger.info(f"Epoch {end_to_end_epoch_idx}: Computing end-to-end metrics (reward) from slots [{reward_start_slot}, {reward_end_slot}]")
                logger.info(f"   Previous epoch window ended at slot {self.previous_epoch_window}, current window ends at slot {slot_end}")
                
                epoch_consensus_metrics = self.collect_epoch_throughput_latency(reward_start_slot, reward_end_slot)
                # Ensure we have a valid ConsensusMetrics instance
                if epoch_consensus_metrics is None:
                    logger.warning(f"Failed to collect end-to-end metrics, using default values")
                    epoch_consensus_metrics = ConsensusMetrics(0.0, 0.0, 0.0)

            # others
            window_duration = window_end_time - window_start_time

            return SystemState(
                timestamp=time.time(),
                window_duration=window_duration,
                hardware=hardware,
                network=network,
                workload=workload,
                lane_vector=lane_vector,
                fast_path_ratio=fast_path_ratio,
                consensus_metrics=epoch_consensus_metrics
            )
        except Exception as e:
            logger.error(f"Failed to collect state for slot window [{slot_start}, {slot_end}]: {e}")
            return None

    def _get_lane_vector_for_slot_window(self, slot_start: int, slot_end: int) -> Dict[str, float]:
        """Get lane vector for a specific slot window"""
        try:
            # Read prepare events from ALL metrics log files (not just one node)
            all_prepare_events = []

            # Find all metrics files
            metrics_files = list(self.log_dir.glob("metrics-*.log"))
            if not metrics_files:
                # Fallback to metrics.log if no numbered files exist
                metrics_files = [self.log_dir / "metrics.log"] if (self.log_dir / "metrics.log").exists() else []

            for metrics_file in metrics_files:
                try:
                    # Extract node index from filename for logging
                    filename = metrics_file.name
                    if filename.startswith("metrics-") and filename.endswith(".log"):
                        file_node_index = filename[8:-4]  # Remove "metrics-" and ".log"
                    else:
                        file_node_index = "unknown"

                    print(f"Reading prepare events from {filename} (node {file_node_index})")

                    # Read events from this specific file
                    lines = self._read_log_file(metrics_file)

                    for line in lines:
                        try:
                            event_dict = json.loads(line.strip())
                            # Remove node_index if present (for backward compatibility)
                            event_dict.pop('node_index', None)
                            event = MetricsEvent(**event_dict)

                            # Filter by event type and slot range
                            if event.event_type == "prepare":
                                slot = event.details.get('slot')
                                if slot is not None and slot_start <= slot < slot_end:
                                    all_prepare_events.append(event)

                        except (json.JSONDecodeError, ValueError):
                            continue  # Skip malformed lines

                except Exception as e:
                    print(f"Warning: Failed to read from {metrics_file}: {e}")
                    continue

            # Events are already filtered by slot range in _read_metrics_events
            # But we still need to check in case slot filtering wasn't applied
            prepare_events = []
            window_start_time = float('inf')
            window_end_time = 0

            for event in all_prepare_events:
                slot = event.details.get('slot')
                if slot is not None and slot_start <= slot < slot_end:
                    prepare_events.append(event)

                    # Also track time boundaries for this slot window
                    from datetime import datetime
                    try:
                        # Handle different timestamp formats
                        ts_str = event.timestamp
                        if ts_str.endswith('Z'):
                            ts_str = ts_str[:-1] + '+00:00'

                        # Truncate microseconds to 6 digits (Python datetime limit)
                        if '.' in ts_str:
                            parts = ts_str.split('.')
                            if len(parts) == 2 and len(parts[1]) > 6:
                                # Find timezone part
                                tz_start = -1
                                for i, c in enumerate(parts[1]):
                                    if c in '+-':
                                        tz_start = i
                                        break
                                if tz_start > 0:
                                    microseconds = parts[1][:tz_start]
                                    timezone = parts[1][tz_start:]
                                    if len(microseconds) > 6:
                                        microseconds = microseconds[:6]
                                    ts_str = f"{parts[0]}.{microseconds}{timezone}"

                        dt = datetime.fromisoformat(ts_str)
                        ts = dt.timestamp()
                        window_start_time = min(window_start_time, ts)
                        window_end_time = max(window_end_time, ts)
                    except Exception as e:
                        # Debug: print problematic timestamps
                        print(f"Warning: Failed to parse timestamp '{event.timestamp}': {e}")
                        continue

            if not prepare_events:
                print(f"No prepare events found in slot range [{slot_start}, {slot_end}]")
                print(f"Total prepare events read: {len(all_prepare_events)}")
                if all_prepare_events:
                    sample_slots = [e.details.get('slot') for e in all_prepare_events[:5] if e.details.get('slot') is not None]
                    print(f"Sample slots from events: {sample_slots}")
                return {}

            if window_start_time == float('inf'):
                print(f"No valid timestamps found for prepare events in slot range [{slot_start}, {slot_end}]")
                return {}

            # Group events by validator first, then sort by timestamp
            validator_events = {}
            for event in prepare_events:
                details = event.details
                validator_pk = details['validator_pk']

                # Parse timestamp for this event
                try:
                    ts_str = event.timestamp
                    if ts_str.endswith('Z'):
                        ts_str = ts_str[:-1] + '+00:00'

                    # Truncate microseconds to 6 digits (Python datetime limit)
                    if '.' in ts_str:
                        parts = ts_str.split('.')
                        if len(parts) == 2 and len(parts[1]) > 6:
                            # Find timezone part
                            tz_start = -1
                            for i, c in enumerate(parts[1]):
                                if c in '+-':
                                    tz_start = i
                                    break
                            if tz_start > 0:
                                microseconds = parts[1][:tz_start]
                                timezone = parts[1][tz_start:]
                                if len(microseconds) > 6:
                                    microseconds = microseconds[:6]
                                ts_str = f"{parts[0]}.{microseconds}{timezone}"

                    dt = datetime.fromisoformat(ts_str)
                    timestamp = dt.timestamp()
                except Exception as e:
                    print(f"Warning: Failed to parse timestamp '{event.timestamp}' for validator {validator_pk}: {e}")
                    continue

                if validator_pk not in validator_events:
                    validator_events[validator_pk] = []
                validator_events[validator_pk].append({
                    'height': details['proposal_height'],
                    'timestamp': timestamp,
                    'slot': details['slot']
                })

            # Calculate height growth for each validator
            validator_height_data = {}
            for validator_pk, events in validator_events.items():
                # Sort events by timestamp
                events.sort(key=lambda x: x['timestamp'])

                if events:
                    start_event = events[0]
                    end_event = events[-1]

                    validator_height_data[validator_pk] = {
                        'start_height': start_event['height'],
                        'end_height': end_event['height'],
                        'start_time': start_event['timestamp'],
                        'end_time': end_event['timestamp']
                    }

            # Calculate lane vector based on proposal height growth rates in this slot window
            lane_vector = {}

            # Group events by slot first, then by validator to calculate height growth
            slot_validator_heights = {}  # slot -> validator -> [(height, timestamp)]

            for validator_pk, events in validator_events.items():
                for event in events:
                    slot = event['slot']  # Use actual slot from event
                    height = event['height']
                    timestamp = event['timestamp']
                    if slot not in slot_validator_heights:
                        slot_validator_heights[slot] = {}
                    if validator_pk not in slot_validator_heights[slot]:
                        slot_validator_heights[slot][validator_pk] = []
                    slot_validator_heights[slot][validator_pk].append((height, timestamp))

            # Calculate height growth rate for each validator across the slot range
            for validator_pk in validator_events.keys():
                # Get all (height, timestamp) pairs for this validator across all slots in range
                validator_height_times = []
                for slot in sorted(slot_validator_heights.keys()):
                    if validator_pk in slot_validator_heights[slot]:
                        validator_height_times.extend(slot_validator_heights[slot][validator_pk])

                # Sort by timestamp to ensure chronological order
                validator_height_times.sort(key=lambda x: x[1])

                if len(validator_height_times) >= 2:
                    # Calculate growth rate: (final_height - initial_height) / time_span
                    height_growth = validator_height_times[-1][0] - validator_height_times[0][0]
                    time_span = max(0.1, window_end_time - window_start_time)
                    growth_rate = height_growth / time_span
                    lane_vector[validator_pk] = max(0.0, growth_rate)  # Ensure non-negative
                else:
                    # Not enough data points, use minimal activity
                    lane_vector[validator_pk] = 0.001

            # Ensure all validators from committee are represented
            # If some validators have no events, give them minimal activity
            try:
                committee_events = self._read_metrics_events(event_types=["committee"], max_lines=1)
                if committee_events:
                    committee_details = committee_events[0].details
                    # Extract all validator PKs from committee
                    all_validators = set()
                    for i in range(len(committee_details)):
                        pk_key = f"pk_{i}" if f"pk_{i}" in committee_details else "pk"
                        if pk_key in committee_details:
                            all_validators.add(committee_details[pk_key])
                        elif isinstance(committee_details, list) and i < len(committee_details):
                            if "pk" in committee_details[i]:
                                all_validators.add(committee_details[i]["pk"])

                    # Add missing validators with minimal activity
                    for validator_pk in all_validators:
                        if validator_pk not in lane_vector:
                            lane_vector[validator_pk] = 0.001  # Minimal activity
            except Exception as e:
                print(f"Warning: Could not get committee info for lane vector completion: {e}")

            print(f"Calculated lane vector with {len(lane_vector)} validators in slots [{slot_start}, {slot_end}]")
            if lane_vector:
                # Print some stats for debugging
                rates = list(lane_vector.values())
                print(f"  Activity rates - min: {min(rates):.3f}, max: {max(rates):.3f}, avg: {sum(rates)/len(rates):.3f}")

            return lane_vector

        except Exception as e:
            logger.error(f"Failed to get lane vector for slot window: {e}")
            return {}

    def save_state(self, state: SystemState, filename: Optional[str] = None,
                   include_current_action: bool = True) -> str:
        """Save system state to file, optionally including current action from parameters"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"system_state_{timestamp}.json"

        # Use output_dir for saving files (user-writable)
        filepath = self.output_dir / filename

        # Convert to serializable dict
        # Ensure consensus_metrics is a valid ConsensusMetrics instance
        if state.consensus_metrics is None:
            consensus_metrics_dict = asdict(ConsensusMetrics(0.0, 0.0, 0.0))
        else:
            consensus_metrics_dict = asdict(state.consensus_metrics)

        state_dict = {
            'timestamp': state.timestamp,
            'window_duration': state.window_duration,
            'state_1_hardware': asdict(state.hardware),
            # 'state_2_network': asdict(state.network),  # Disabled per current requirement
            'state_3_workload': asdict(state.workload),
            'state_4_lane_vector': asdict(state.lane_vector),
            'state_5_fast_path_ratio': state.fast_path_ratio,
            'end_to_end_metrics': consensus_metrics_dict
        }

        # Optionally include current action from parameters file
        if include_current_action:
            try:
                current_action = self._get_current_action_for_state()
                if current_action:
                    state_dict['current_action'] = current_action
                    print(f"📋 Included current action in state file: batch_size={current_action.get('batch_size', 'unknown')}")
            except Exception as e:
                print(f"⚠️ Failed to include current action in state file: {e}")

        state_dict = _sanitize_floats_for_json(state_dict)
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)

        print(f"✅ System state saved to: {filepath}")
        return str(filepath)

    def save_epoch_states(self, states: List[SystemState], base_filename: Optional[str] = None) -> List[str]:
        """Save multiple epoch states to separate files

        Args:
            states: List of SystemState objects
            base_filename: Base filename prefix

        Returns:
            List of saved file paths
        """
        if base_filename is None:
            timestamp = int(time.time())
            base_filename = f"epoch_state_{timestamp}"

        saved_files = []
        for i, state in enumerate(states):
            filename = f"{base_filename}_epoch_{i}.json"
            filepath = self.save_state(state, filename)
            saved_files.append(filepath)

        logger.info(f"Saved {len(states)} epoch states")
        return saved_files
    
    def get_state_vector(self, state: SystemState) -> np.ndarray:
        """Convert system state to RL state vector with 5 components"""

        # State 1: Hardware configurations
        hardware_vec = [
            float(state.hardware.cpu_cores),
            float(state.hardware.memory_gb),
            float(state.hardware.network_bandwidth_mbps),
            float(state.hardware.workers_per_node)
        ]

        # State 2: Network condition (1D vector of latencies and bandwidths)
        network_vec = state.network.latency_vector

        # State 3: Workload
        workload_vec = [
            float(state.workload.tx_size_bytes),
            float(state.workload.tx_arrival_rate_tps)
        ]

        # State 4: Lane vector (height growth rates)
        # Sort by validator key for consistent ordering
        lane_rates = [rate for _, rate in sorted(state.lane_vector.growth_rates.items())]
        if not lane_rates:  # fallback if no lanes
            lane_rates = [0.0]

        # State 5: Fast path ratio
        fast_path_vec = [float(state.fast_path_ratio)]

        # Concatenate all state vectors
        state_vector = np.concatenate([
            hardware_vec,      # State 1
            network_vec,       # State 2
            workload_vec,      # State 3
            lane_rates,        # State 4
            fast_path_vec      # State 5
        ])

        # Normalization (optional)
        state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=0.0)

        return state_vector

    def _get_current_action_for_state(self) -> Dict[str, Any]:
        """Get current action from parameters file for state saving (includes all relevant parameters)"""
        try:
            # Read current parameters from file
            with open(self.parameters_file, 'r') as f:
                params = json.load(f)

            # Map parameter file fields to action fields (matching controller's mapping)
            action = {
                "batch_size": params.get("batch_size", 500000),
                "max_batch_delay_ms": params.get("max_batch_delay", 1000),
                "header_size": params.get("header_size", 32),
                "max_header_delay_ms": params.get("max_header_delay", 1000),
                "enable_uncertified_tip": params.get("use_optimistic_tips", False),
                "sync_retry_delay_ms": params.get("sync_retry_delay", 1000),
                "sync_retry_nodes": params.get("sync_retry_nodes", 4),
                "cut_condition_type": params.get("cut_condition_type", 3),
                "fast_path_timeout_ms": params.get("fast_path_timeout", 40),
                "parallel_proposals": params.get("k", 1)
            }

            print(f"📋 Current action from parameters: batch_size={action['batch_size']}, "
                  f"fast_path_timeout={action['fast_path_timeout_ms']}ms, "
                  f"parallel_proposals={action['parallel_proposals']}")

            return action

        except Exception as e:
            logger.warning(f"Failed to read current action from parameters file, using defaults: {e}")
            # Fallback to default action
            return {
                "batch_size": 500000,
                "max_batch_delay_ms": 1000,
                "header_size": 32,
                "max_header_delay_ms": 1000,
                "enable_uncertified_tip": False,
                "sync_retry_delay_ms": 1000,
                "sync_retry_nodes": 4,
                "cut_condition_type": 3,
                "fast_path_timeout_ms": 40,
                "parallel_proposals": 1
            }

    def _send_state_to_rust(self, epoch: int, state: SystemState):
        """Send state directly to Rust via Unix socket (similar to BFTBrain's message passing)"""
        import socket
        import tempfile
        import os

        try:
            # Use fixed socket path (no node index needed)
            # First try environment variable (set by Rust), then fallback to fixed path
            socket_path = os.environ.get('RUST_STATE_SOCKET_PATH', '/tmp/autobahn_state_server.sock')
            
            print(f"🔌 Using socket path: {socket_path}")
            
            # Verify socket file exists
            if not os.path.exists(socket_path):
                print(f"⚠️ Socket file does not exist: {socket_path}")
                try:
                    available = [f for f in os.listdir('/tmp') if 'autobahn_state_server' in f]
                    print(f"   Available sockets in /tmp: {available}")
                except Exception:
                    pass
                return
            
            # Retry connection with exponential backoff (socket server may not be ready yet)
            max_retries = 5
            retry_delay = 0.1  # Start with 100ms
            sock = None
            
            for attempt in range(max_retries):
                try:
                    print(f"🔌 Connecting to Rust state server: {socket_path} (attempt {attempt + 1}/{max_retries})")
                    
                    # Create Unix socket client
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.settimeout(2.0)  # 2 second timeout per attempt
                    
                    sock.connect(socket_path)
                    print(f"✅ Connected to Rust state server")
                    break  # Successfully connected, exit retry loop
                except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
                    # Close socket if it was created but connection failed
                    if sock is not None:
                        try:
                            sock.close()
                        except Exception:
                            pass
                        sock = None
                    
                    if attempt < max_retries - 1:
                        print(f"⚠️ Connection attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"❌ Failed to connect after {max_retries} attempts: {e}")
                        return
                except Exception as e:
                    # Close socket if it was created
                    if sock is not None:
                        try:
                            sock.close()
                        except Exception:
                            pass
                    print(f"❌ Unexpected error connecting to socket: {e}")
                    return
            
            if sock is None:
                print(f"❌ Could not connect to Rust state server after {max_retries} attempts")
                return
            
            # Now we have a connected socket, send the state data
            try:
                # Convert state to JSON dict (same structure as saved file)
                state_dict = {
                    'timestamp': state.timestamp,
                    'window_duration': state.window_duration,
                    'state_1_hardware': asdict(state.hardware),
                    # 'state_2_network': asdict(state.network),  # Disabled per current requirement
                    'state_3_workload': asdict(state.workload),
                    'state_4_lane_vector': asdict(state.lane_vector),
                    'state_5_fast_path_ratio': state.fast_path_ratio,
                    'end_to_end_metrics': asdict(state.consensus_metrics) if state.consensus_metrics else {
                        'end_to_end_tps': 0.0,
                        'end_to_end_bps': 0.0,
                        'end_to_end_latency_ms': 0.0
                    }
                }

                # Extract end-to-end metrics
                tps = state.consensus_metrics.end_to_end_tps if state.consensus_metrics else 0.0
                latency_ms = state.consensus_metrics.end_to_end_latency_ms if state.consensus_metrics else 0.0

                # Create message for Rust
                message = {
                    'epoch': epoch,
                    'state_json': state_dict,
                    'tps': tps,
                    'latency_ms': latency_ms,
                    'timestamp': state.timestamp
                }

                # Send message as JSON (same format as Rust expects)
                message_json = json.dumps(_sanitize_floats_for_json(message))
                message_bytes = message_json.encode('utf-8')

                # Send message length first (4 bytes), then message
                length_bytes = len(message_bytes).to_bytes(4, byteorder='big')
                sock.sendall(length_bytes)
                sock.sendall(message_bytes)

                # Receive ACK
                ack = sock.recv(1024)
                if ack == b'ACK':
                    print(f"✅ State sent to Rust for epoch {epoch} (in-memory storage)")
                else:
                    print(f"⚠️ Unexpected ACK from Rust: {ack}")

            except Exception as e:
                print(f"❌ Error sending data to Rust socket: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Close socket if it exists
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass

        except Exception as e:
            print(f"❌ Failed to send state to Rust: {e}")
            import traceback
            traceback.print_exc()

def main(epoch_slots: int = 20, window_size: int = 5, committed_slot: Optional[int] = None,
         node_index: Optional[int] = None, log_dir: Optional[str] = None,
         parameters_file: Optional[str] = None, clear_metrics_every: Optional[int] = None,
         clear_metrics_log_path: Optional[str] = None):

    # Create collector and configure cache settings
    collector = MetricsCollector(log_dir=log_dir, node_index=node_index, parameters_file=parameters_file)

    # Optional: clear metrics log every N epochs
    clear_every = clear_metrics_every if clear_metrics_every and clear_metrics_every > 0 else None
    if clear_metrics_log_path:
        clear_target = Path(clear_metrics_log_path)
    elif log_dir:
        filename = f"metrics-{node_index}.log" if node_index is not None else "metrics.log"
        clear_target = Path(log_dir) / filename
    else:
        clear_target = None
    def _maybe_clear_metrics_log(epoch_idx: int):
        if not clear_every or not clear_target:
            return
        if (epoch_idx + 1) % clear_every != 0:
            return
        try:
            clear_target.parent.mkdir(parents=True, exist_ok=True)
            with open(clear_target, "w", encoding="utf-8"):
                pass
            logger.info("🧹 Cleared metrics log: %s (every %d epochs)", clear_target, clear_every)
        except Exception as e:
            logger.warning("Failed to clear metrics log %s: %s", clear_target, e)

    # Check if this is a real-time collection call from core.rs
    # Look for committed_slot in environment or command line
    committed_slot_env = os.environ.get('COMMITTED_SLOT')
    if committed_slot_env is not None and committed_slot is None:
        committed_slot = int(committed_slot_env)

    # If we have a committed_slot, this is real-time collection mode
    if committed_slot is not None:
        print(f"🎯 Real-time collection for committed slot {committed_slot}")

        # Trigger when committed_slot is a positive multiple of window_size (last slot of the epoch).
        # Slots start from 1. Epoch k (0-based): [k*W+1, (k+1)*W+1) i.e. slots k*W+1 .. (k+1)*W
        if committed_slot > 0 and window_size > 0 and committed_slot % window_size == 0:
            # epoch_idx (0-based): committed_slot / window_size - 1
            epoch_idx = committed_slot // window_size - 1
            # Window: [start_slot, end_slot) half-open
            start_slot = committed_slot - window_size + 1   # inclusive
            end_slot = committed_slot + 1                   # exclusive
            print(f"   Epoch {epoch_idx}: collecting state for slot window [{start_slot}, {end_slot}) "
                  f"(committed_slot={committed_slot}, window_size={window_size})")

            # Collect state for the slot window
            # For end-to-end metrics, use the previous epoch's data (epoch_idx - 1)
            # end_to_end_epoch_idx = epoch_idx means: use data from epoch (epoch_idx - 1)
            end_to_end_epoch = epoch_idx if epoch_idx > 0 else None
            state = collector._collect_system_state_for_slot_window(start_slot, end_slot, end_to_end_epoch, epoch_slots)
            if state:
                # Save state to JSON file first (for observation/debugging)
                filename = f"epoch_{epoch_idx}_slot_{committed_slot}.json"
                saved_path = collector.save_state(state, filename)
                print(f"   ✅ State saved to JSON file: {saved_path}")

                # Send state directly to Rust via socket (in-memory, similar to BFTBrain)
                print(f"   📤 Sending state to Rust for epoch {epoch_idx} (committed_slot={committed_slot})")
                collector._send_state_to_rust(epoch_idx, state)
                print(f"   ✅ State sent to Rust for epoch {epoch_idx} (in-memory storage)")

                # Also send training message to controller (for RL training)
                # collector._send_training_message(state, saved_path)
                collector.previous_epoch_window = end_slot
                logger.info("📌 Updated previous_epoch_window to %s for next epoch's reward calculation", end_slot)
                _maybe_clear_metrics_log(epoch_idx)
                print("✅ Real-time metrics collection completed successfully")
            else:
                print("❌ Failed to collect state")
                import sys
                sys.exit(1)
        else:
            print(f"   Slot {committed_slot} does not trigger collection")

        # For real-time collection mode, always exit after attempting collection
        return

    # If no committed_slot, run in daemon mode: connect to core's Unix socket and listen for requests
    print("=== Running metrics_collector in daemon mode ===")
    print(f"Epoch slots (h): {epoch_slots}, Window size (j): {window_size}")
    print("Running as background daemon process...")
    print("Connecting to core's Unix socket for collection requests")
    print("Press Ctrl+C to stop.\n")

    import time
    import signal
    import sys
    import socket
    import threading
    import select

    # Get node index for socket path - use the node_index passed via command line
    current_node_idx = node_index if node_index is not None else 0
    core_socket_path = f"/tmp/autobahn_core_{current_node_idx}.sock"

    # Connect to core's socket server (retry with backoff)
    core_socket = None
    max_retries = 20
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            logger.info(f"🔌 Attempting to connect to core socket: {core_socket_path} (attempt {attempt + 1}/{max_retries})")
            core_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            core_socket.connect(core_socket_path)
            logger.info(f"✅ Successfully connected to core socket: {core_socket_path}")
            break
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️  Connection attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = 2.0  # Exponential backoff, max 10s
            else:
                logger.error(f"❌ Failed to connect to core socket after {max_retries} attempts: {e}")
                sys.stdout.flush()
                sys.stderr.flush()
                raise

    if core_socket is None:
        logger.error("❌ Could not establish connection to core socket")
        sys.stdout.flush()
        sys.stderr.flush()
        raise RuntimeError("Failed to connect to core socket")
    
    # Signal handler for graceful shutdown
    shutdown_requested = False
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        print("\n🛑 Shutting down metrics_collector daemon...")
        shutdown_requested = True
        try:
            core_socket.close()
        except:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep running as daemon - listen for requests from core
    try:
        sockets_to_read = [core_socket]
        logger.info("🔄 Starting daemon socket listening loop")
        while not shutdown_requested:
            # Use select to wait for data with timeout
            logger.debug("🔍 Waiting for socket data (1s timeout)...")
            readable, _, _ = select.select(sockets_to_read, [], [], 1.0)
            if readable:
                logger.debug("✅ Socket has data available")
            else:
                logger.debug("⏰ Select timeout - no data available")

            if readable:
                logger.debug("📡 Socket became readable, reading data...")
                # Read and process request
                try:
                    buffer = b''
                    chunk_count = 0
                    while True:
                        chunk = core_socket.recv(4096)
                        if not chunk:
                            logger.info("🔌 Connection closed by core")
                            shutdown_requested = True
                            break
                        buffer += chunk
                        chunk_count += 1
                        logger.debug(f"📦 Received chunk {chunk_count}, buffer size: {len(buffer)} bytes")
                        if b'\n' in buffer:
                            logger.debug("📝 Found complete message (newline detected)")
                            break

                    if shutdown_requested:
                        break

                    # Process message
                    try:
                        buffer_str = buffer.decode('utf-8')
                        logger.debug(f"🔤 Decoded buffer to string: {buffer_str.strip()}")
                        lines = buffer_str.split('\n')
                        logger.debug(f"📋 Split into {len(lines)} lines")
                        for line in lines[:-1]:
                            if not line.strip():
                                logger.debug("⚪ Skipping empty line")
                                continue
                            logger.debug(f"📝 Processing line: {line.strip()}")
                            try:
                                request = json.loads(line.strip())
                                logger.info(f"📨 Received raw request: {line.strip()}")

                                action = request.get('action')
                                request_id = request.get('request_id')
                                epoch = request.get('epoch')  # NEW: epoch number from Rust
                                committed_slot = request.get('committed_slot')
                                req_epoch_slots = request.get('epoch_slots', epoch_slots)
                                req_window_size = request.get('window_size', window_size)
                                
                                # NEW: Transaction-based metrics
                                total_committed_tx = request.get('total_committed_transactions')
                                epoch_tx = request.get('epoch_transactions')
                                window_tx = request.get('window_transactions')
                                
                                # NEW: Slot mapping from core.rs
                                slot_start = request.get('slot_start')
                                slot_end = request.get('slot_end')

                                logger.info(f"🔧 Parsed request - action: {action}, request_id: {request_id}, epoch: {epoch}, committed_slot: {committed_slot}")
                                logger.info(f"   Transaction-based: total_tx={total_committed_tx}, epoch_tx={epoch_tx}, window_tx={window_tx}")
                                logger.info(f"   Slot mapping from core: slot_start={slot_start}, slot_end={slot_end}")
                                logger.info(f"   Slot-based (deprecated): epoch_slots={req_epoch_slots}, window_size={req_window_size}")

                                if action == 'collect_state':
                                    is_slot_mode = committed_slot is not None

                                    if not is_slot_mode:
                                        logger.warning(f"❌ Invalid request: action={action}, committed_slot={committed_slot}")
                                        error_response = {'request_id': request_id, 'error': 'Invalid request'}
                                        error_json = json.dumps(_sanitize_floats_for_json(error_response))
                                        logger.info(f"📤 Sending error response: {error_json}")
                                        core_socket.sendall((error_json + '\n').encode('utf-8'))
                                        continue

                                    # Slot-based mode: slots start from 1, epoch k covers [k*W+1, (k+1)*W+1).
                                    # Prefer slot_start/slot_end sent directly from Rust; fall back to calculation.
                                    if slot_start is not None and slot_end is not None:
                                        # Rust already computed the correct window
                                        epoch_idx = epoch if epoch is not None else (committed_slot // req_window_size - 1 if req_window_size > 0 else 0)
                                        start_slot = slot_start   # inclusive
                                        end_slot = slot_end       # exclusive
                                    else:
                                        # Fallback: derive from epoch_idx
                                        epoch_idx = epoch if epoch is not None else (committed_slot // req_window_size - 1 if req_window_size > 0 else 0)
                                        start_slot = epoch_idx * req_window_size + 1   # inclusive
                                        end_slot = start_slot + req_window_size        # exclusive

                                    logger.info(f"📥 Processing collect_state request {request_id} for epoch {epoch_idx} (committed_slot={committed_slot})")
                                    logger.info(f"📊 Calculated collection window: epoch {epoch_idx}, slots [{start_slot}, {end_slot})")

                                    end_to_end_epoch = epoch_idx if epoch_idx > 0 else None
                                    state = collector._collect_system_state_for_slot_window(start_slot, end_slot, end_to_end_epoch, req_epoch_slots)

                                    if state:
                                        filename = f"epoch_{epoch_idx}_slot_{committed_slot}.json"
                                        saved_path = collector.save_state(state, filename)
                                        logger.info(f"✅ State collected and saved to: {saved_path}")
                                        response = {
                                            'request_id': request_id,
                                            'status': 'collected',
                                            'epoch': epoch_idx,
                                            'committed_slot': committed_slot,
                                            'slot_range': f'[{start_slot},{end_slot})',
                                            'state_file': str(saved_path),
                                            'timestamp': state.timestamp,
                                            'window_duration': state.window_duration
                                        }
                                        response_json = json.dumps(_sanitize_floats_for_json(response))
                                        logger.info(f"📤 Sending success response: {response_json}")
                                        core_socket.sendall((response_json + '\n').encode('utf-8'))
                                        collector.previous_epoch_window = end_slot
                                        logger.info("📌 Updated previous_epoch_window to %s for next epoch's reward calculation", end_slot)
                                        _maybe_clear_metrics_log(epoch_idx)
                                    else:
                                        logger.warning(f"❌ Failed to collect state for epoch {epoch_idx}")
                                        error_response = {
                                            'request_id': request_id,
                                            'status': 'failed',
                                            'error': f'Failed to collect state for epoch {epoch_idx}, slot_range=[{start_slot},{end_slot})'
                                        }
                                        error_json = json.dumps(_sanitize_floats_for_json(error_response))
                                        logger.info(f"📤 Sending error response: {error_json}")
                                        core_socket.sendall((error_json + '\n').encode('utf-8'))
                                else:
                                    logger.warning(f"❌ Invalid request: action={action}, committed_slot={committed_slot}")
                                    error_response = {'request_id': request_id, 'error': 'Invalid request'}
                                    error_json = json.dumps(_sanitize_floats_for_json(error_response))
                                    logger.info(f"📤 Sending error response: {error_json}")
                                    core_socket.sendall((error_json + '\n').encode('utf-8'))

                            except json.JSONDecodeError as e:
                                logger.error(f"❌ JSON decode error for request line: {line.strip()} - {str(e)}")
                                error_response = {'error': f'Invalid JSON: {str(e)}'}
                                error_json = json.dumps(_sanitize_floats_for_json(error_response))
                                logger.info(f"📤 Sending JSON error response: {error_json}")
                                core_socket.sendall((error_json + '\n').encode('utf-8'))
                            except Exception as e:
                                logger.error(f"❌ Request processing error: {str(e)}")
                                error_response = {'error': f'Processing error: {str(e)}'}
                                error_json = json.dumps(_sanitize_floats_for_json(error_response))
                                logger.info(f"📤 Sending processing error response: {error_json}")
                                core_socket.sendall((error_json + '\n').encode('utf-8'))

                    except Exception as e:
                        logger.error(f"Error processing buffer: {e}")

                except Exception as e:
                    logger.error(f"Error reading socket: {e}")
                    shutdown_requested = True

    except KeyboardInterrupt:
        print("\n🛑 Shutting down metrics_collector daemon...")
    except Exception as e:
        logger.error(f"Error in daemon mode: {e}")
        print(f"❌ Error in daemon mode: {e}")
        raise
    finally:
        try:
            core_socket.close()
        except Exception:
            pass

    print("🛑 metrics_collector daemon stopped.")
    sys.stdout.flush()
    return

if __name__ == "__main__":
    # Parse parameters from command line (from core.rs or manual)
    epoch_slots = 20  # Default h parameter
    window_size = 5   # Default j parameter
    committed_slot = None
    node_index = None

    # Parse command line arguments
    i = 1
    log_dir = None # Default log directory
    parameters_file = None
    clear_metrics_every = 10
    clear_metrics_log_path = None
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.isdigit() and i == 1:
            epoch_slots = int(arg)
        elif arg.isdigit() and i == 2:
            window_size = int(arg)
        elif arg == "--slot" and i + 1 < len(sys.argv):
            committed_slot = int(sys.argv[i + 1])
            i += 1
        elif arg == "--node-index" and i + 1 < len(sys.argv):
            node_index = int(sys.argv[i + 1])
            i += 1
        elif arg == "--log-dir" and i + 1 < len(sys.argv):
            log_dir = sys.argv[i + 1]
            i += 1
        elif arg == "--parameters-file" and i + 1 < len(sys.argv):
            parameters_file = sys.argv[i + 1]
            i += 1
        elif arg == "--clear-metrics-every" and i + 1 < len(sys.argv):
            clear_metrics_every = int(sys.argv[i + 1])
            i += 1
        elif arg == "--clear-metrics-log-path" and i + 1 < len(sys.argv):
            clear_metrics_log_path = sys.argv[i + 1]
            i += 1
        i += 1

    print(f"📊 Metrics Collection: h={epoch_slots}, j={window_size}, slot={committed_slot}, node={node_index}")
    print(f"🚀 METRICS_COLLECTOR_STARTED_BY_AUTOBahn: PID={os.getpid()}, ARGS={sys.argv}")
    logger.info(f"📊 Metrics Collection: h={epoch_slots}, j={window_size}, slot={committed_slot}, node={node_index}")
    logger.info(f"🚀 METRICS_COLLECTOR_STARTED_BY_AUTOBahn: PID={os.getpid()}, ARGS={sys.argv}")

    # Validate required parameters
    if log_dir is None:
        logger.error("❌ --log-dir parameter is required but not provided")
        sys.exit(1)

    # Set environment variables
    if node_index is not None:
        os.environ['NODE_INDEX'] = str(node_index)
    if committed_slot is not None:
        os.environ['COMMITTED_SLOT'] = str(committed_slot)

    # Initialize metrics logger with the correct node index and log directory
    get_metrics_logger(log_dir=log_dir, node_index=node_index)

    # Call main function with parsed parameters
    main(epoch_slots=epoch_slots, window_size=window_size, committed_slot=committed_slot,
         node_index=node_index, log_dir=log_dir, parameters_file=parameters_file,
         clear_metrics_every=clear_metrics_every, clear_metrics_log_path=clear_metrics_log_path)
