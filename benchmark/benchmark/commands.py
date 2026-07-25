# Copyright(C) Facebook, Inc. and its affiliates.
import os
import subprocess
from os.path import join

from benchmark.utils import PathMaker


class CommandMaker:
    AGENT_VENV_PATH = '/home/ccclr0302/autopilot-venv'

    @staticmethod
    def agent_venv_python():
        return f'{CommandMaker.AGENT_VENV_PATH}/bin/python'

    @staticmethod
    def agent_venv_pip():
        return f'{CommandMaker.AGENT_VENV_PATH}/bin/pip'

    @staticmethod
    def agent_python(python_bin=None):
        if python_bin:
            return python_bin
        venv_python = CommandMaker.agent_venv_python()
        if os.path.isfile(venv_python):
            return venv_python
        return 'python3'

    @staticmethod
    def ensure_agent_venv(requirements_file):
        """Create the Agent venv locally and install dependencies."""
        venv_path = CommandMaker.AGENT_VENV_PATH
        venv_pip = CommandMaker.agent_venv_pip()
        if not os.path.isdir(venv_path):
            subprocess.run(['python3', '-m', 'venv', venv_path], check=True)
        subprocess.run([venv_pip, 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([venv_pip, 'install', '-r', requirements_file], check=True)
        return CommandMaker.agent_venv_python()

    @staticmethod
    def remote_agent_venv_setup_cmds(repo_name):
        """Shell commands to create the Agent venv on a remote host."""
        venv_path = CommandMaker.AGENT_VENV_PATH
        venv_pip = CommandMaker.agent_venv_pip()
        return [
            'sudo apt-get update',
            'sudo apt-get -y install python3-pip python3-venv',
            f'test -d {venv_path} || python3 -m venv {venv_path}',
            f'{venv_pip} install --upgrade pip',
            f'(cd {repo_name} && {venv_pip} install -r Agent/requirements.txt)',
        ]

    @staticmethod
    def cleanup():
        return (
            f'rm -r .db-* ; rm .*.json ; mkdir -p {PathMaker.results_path()}'
        )

    @staticmethod
    def clean_logs():
        return f'rm -r {PathMaker.logs_path()} ; mkdir -p {PathMaker.logs_path()}'

    @staticmethod
    def clean_metrics(repo_name, node_id):
        return (
            f'rm -f /home/ccclr0302/{repo_name}/benchmark/latency_full_matrix_*.npy ; '
            f'rm -f /home/ccclr0302/{repo_name}/benchmark/latency_region_matrix_*.npy ; '
            f'rm -f /home/ccclr0302/{repo_name}/benchmark/latency_vector_*.npy ; '
            # f'rm -rf /home/ccclr0302/{repo_name}/metrics-{node_id}; '
            f'rm -f /tmp/autopilot_rl_param_*.sock ; '
            f'rm -f /tmp/autopilot_rl_param_abandon_*.signal ; '
            f'rm -f /tmp/autopilot_core_*.sock ; '
            f'rm -f /tmp/autopilot_controller_*.sock ; '
            f'sudo rm -rf /home/ccclr0302/metrics-* /home/ccclr0302/{repo_name}/metrics-* || true'
        )

    @staticmethod
    def compile():
        return 'cargo build --quiet --release --features benchmark'

    @staticmethod
    def generate_key(filename):
        assert isinstance(filename, str)
        return f'./node generate_keys --filename {filename}'

    @staticmethod
    def run_primary(keys, committee, store, parameters, debug=False, node_index=None):
        assert isinstance(keys, str)
        assert isinstance(committee, str)
        assert isinstance(parameters, str)
        assert isinstance(debug, bool)
        v = '-vvv' if debug else '-vv'
        socket_env = f'RUST_STATE_SOCKET_PATH=/tmp/autopilot_core_{node_index}.sock' if node_index is not None else ''
        cmd = f'./node {v} run --keys {keys} --committee {committee} '
        cmd += f'--store {store} --parameters {parameters} primary'
        if socket_env:
            cmd = f'{socket_env} {cmd}'
        return cmd

    @staticmethod
    def run_worker(keys, committee, store, parameters, id, debug=False):
        assert isinstance(keys, str)
        assert isinstance(committee, str)
        assert isinstance(parameters, str)
        assert isinstance(debug, bool)
        v = '-vvv' if debug else '-vv'
        return (f'./node {v} run --keys {keys} --committee {committee} '
                f'--store {store} --parameters {parameters} worker --id {id}')

    @staticmethod
    def run_client(address, size, rate, nodes, node_id=None, hotspot_config=None):
        assert isinstance(address, str)
        assert isinstance(size, int) and size > 0
        assert isinstance(rate, int) and rate >= 0
        assert isinstance(nodes, list)
        assert all(isinstance(x, str) for x in nodes)
        
        # Build base command
        nodes_str = f'--nodes {" ".join(nodes)}' if nodes else ''
        cmd = f'./benchmark_client {address} --size {size} --rate {rate} {nodes_str}'
        
        cmd += f' --node-id {node_id}'
        
        # Add hotspot configuration if provided
        if hotspot_config and hotspot_config.get('enable_hotspot'):
            windows = hotspot_config.get('hotspot_windows', [])
            hotspot_node_ids_per_window = hotspot_config.get('hotspot_node_ids_per_window', [])
            hotspot_node_rates_per_window = hotspot_config.get('hotspot_node_rates_per_window', [])

            if windows:
                windows_str = ' '.join([f'{w[0]}:{w[1]}' for w in windows])
                cmd += f' --hotspot-windows {windows_str}'

                # Region-based: pass resolved node ids per window (format "2,5|3,4")
                if hotspot_node_ids_per_window:
                    ids_str = '|'.join([','.join(str(i) for i in ids) for ids in hotspot_node_ids_per_window])
                    cmd += f' --hotspot-node-ids "{ids_str}"'
                    if hotspot_node_rates_per_window:
                        rates_by_node_str = '|'.join(
                            [','.join(str(float(r)) for r in rates) for rates in hotspot_node_rates_per_window]
                        )
                        cmd += f' --hotspot-node-rates "{rates_by_node_str}"'
                else:
                    # Original: pass hotspot node count per window
                    hotspot_nodes = hotspot_config.get('hotspot_nodes', [])
                    if hotspot_nodes:
                        def _hotspot_count(n):
                            return sum(n) if isinstance(n, list) else n
                        nodes_str = ' '.join([str(_hotspot_count(n)) for n in hotspot_nodes])
                        cmd += f' --hotspot-nodes {nodes_str}'

        return cmd

    @staticmethod
    def kill():
        return 'tmux kill-server'

    @staticmethod
    def alias_binaries(origin):
        assert isinstance(origin, str)
        node, client = join(origin, 'node'), join(origin, 'benchmark_client')
        return f'rm node ; rm benchmark_client ; ln -s {node} . ; ln -s {client} .'

    @staticmethod
    def run_metrics_collector(epoch_slots, window_size, node_index=None, repo_name=None, log_dir=None, parameters_file=None, python_bin=None):
        """Generate command to run metrics_collector as a background process"""
        assert isinstance(epoch_slots, int) and epoch_slots > 0
        assert isinstance(window_size, int) and window_size > 0
        # Use relative path from benchmark directory (../Agent/metrics_collector.py)
        # or absolute path if repo_name is provided
        metrics_collector_path = f'/home/ccclr0302/{repo_name}/Agent/metrics_collector.py'
        socket_path = f'/tmp/autopilot_core_{node_index}.sock'
        python = CommandMaker.agent_python(python_bin)
        cmd = f'RUST_STATE_SOCKET_PATH={socket_path} {python} {metrics_collector_path}'
        cmd += f' {epoch_slots} {window_size}'
        cmd += f' --node-index {node_index}'
        cmd += f' --log-dir {log_dir}'
        cmd += f' --parameters-file {parameters_file}'
        return cmd

    @staticmethod
    def run_controller(node_index=None, repo_name=None, log_dir=None, parameters_file=None, python_bin=None):
        """Generate command to run controller as a background process"""
        controller_path = f'/home/ccclr0302/{repo_name}/Agent/rl/controllers/controller.py'
        # update_parameters_path = f'/home/ccclr0302/{repo_name}/Agent/update_parameters_{node_index}.json'
        python = CommandMaker.agent_python(python_bin)
        cmd = f'{python} {controller_path}'
        cmd += f' --metrics-dir /home/ccclr0302/metrics-{node_index}'
        cmd += f' --node-index {node_index}'
        cmd += f' --log-dir {log_dir}'
        cmd += f' --parameters-file {parameters_file}'
        return cmd