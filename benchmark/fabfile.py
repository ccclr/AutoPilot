# Copyright(C) Facebook, Inc. and its affiliates.
import sys
import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _getargspec_compat(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec_compat

from fabric import task, Connection
import time
import numpy as np

from benchmark.local import LocalBench
from benchmark.logs import ParseError, LogParser
from benchmark.utils import Print
from benchmark.plot import Ploter, PlotError
from benchmark.gcp_instance import InstanceManager
from benchmark.remote import Bench, BenchError
from fabric.transfer import Transfer
from paramiko import RSAKey, SSHException
from invoke.exceptions import UnexpectedExit
import os
from invoke import Responder

@task
def local(ctx, debug=False):
    ''' Run benchmarks on localhost '''
    bench_params = {
        'faults': 0, 
        'nodes': 4,
        'workers': 1,
        'rate': 50000,
        'tx_size': 512,
        'duration': 60,

        # RL algorithm: "cmab" or "gp_bo".
        'rl_algo': 'cmab',
        # Unified RL warmup:
        # - cmab: skip policy updates for the first N iterations
        # - gp_bo: collect N cold-start samples before first GP fit
        'rl_warmup_iterations': 5,
        # Checkpoint path to resume RL, or None to train from scratch.
        'cmab_resume_from': None,

        # Unused
        'simulate_partition': False,
        'partition_start': 5,
        'partition_duration': 5,
        'partition_nodes': 1,
        
        'enable_hotspot': False,
        'hotspot_windows':[[0, 1500]],
        'hotspot_nodes': [1],
    }
    node_params = {
        'timeout_delay': 4_000,  # ms
        'header_size': 32,  # bytes
        'max_header_delay': 400,  # ms
        'gc_depth': 40,  # rounds
        'sync_retry_delay': 1_000,  # ms
        'sync_retry_nodes': 4,  # number of nodes
        'batch_size': 100_000,  # bytes
        'max_batch_delay': 400,  # ms
        'use_optimistic_tips': False,
        'use_parallel_proposals': True,
        'k': 1,
        'epoch_slots': 65,
        'window_size': 10,
        'applied_begin': 30,
        'use_fast_path': True,
        'fast_path_timeout': 0,
        'use_ride_share': False,
        'car_timeout': 2000,
        'cut_condition_type': 1,

        'simulate_asynchrony': False,
        'asynchrony_type': [6],

        'asynchrony_start': [0], #s
        'asynchrony_duration': [1500], #s
        'affected_nodes': [2],
        'egress_penalty': 200, #ms

        'use_fast_sync': True,
        'use_exponential_timeouts': True,

        'aggregation_strategy': 'mean',

        'data_pollution_node_ids': [0],
        'data_pollution_prob': 1.0,
        'data_pollution_strategy': 'mean_equalize',
    }
    try:
        ret = LocalBench(bench_params, node_params).run(debug)
        print(ret.result())
    except BenchError as e:
        Print.error(e)


@task
def create(ctx, nodes=3):
    ''' Create a testbed'''
    try:
        InstanceManager.make().create_instances(nodes)
    except BenchError as e:
        Print.error(e)


@task
def destroy(ctx):
    ''' Destroy the testbed '''
    try:
        InstanceManager.make().terminate_instances()
    except BenchError as e:
        Print.error(e)


@task
def start(ctx, max=4):
    ''' Start at most `max` machines per data center '''
    try:
        InstanceManager.make().start_instances(max)
    except BenchError as e:
        Print.error(e)


@task
def stop(ctx):
    ''' Stop all machines '''
    try:
        InstanceManager.make().stop_instances()
    except BenchError as e:
        Print.error(e)


@task
def info(ctx):
    ''' Display connect information about all the available machines '''
    try:
        InstanceManager.make().print_info()
    except BenchError as e:
        Print.error(e)


@task
def install(ctx):
    ''' Install the codebase on all machines '''
    try:
        Bench(ctx).install()
    except BenchError as e:
        Print.error(e)

from fabric import task


@task
def remote(ctx, debug=False):
    ''' Run benchmarks on GCP '''
    bench_params = {
        'faults': 0,
        'nodes': [4],
        'workers': 1,
        'collocate': True,
        'rate': [20_000],
        'tx_size': 512,
        'duration': 100,
        'runs': 1,

        # RL algorithm: "cmab" (discrete RF-TS) or "gp_bo" (GP-UCB Bayesian Optimization).
        'rl_algo': 'gp_bo',
        # Unified RL warmup:
        # - cmab: skip policy updates for the first N iterations
        # - gp_bo: collect N cold-start samples before first GP fit
        # (GP-BO no longer also skips trainer updates — that was double warmup.)
        'rl_warmup_iterations': 5,
        # Checkpoint path to resume RL, or None to train from scratch.
        # CMAB example: '/home/ccclr0302/checkpoints/cmab_checkpoint_120.pkl'
        # GP-BO example: '/home/ccclr0302/gp_bo_checkpoints/gp_bo_checkpoint_120.pkl'
        'cmab_resume_from': None,

        # Unused5
        'simulate_partition': False,
        'partition_start': 5,
        'partition_duration': 5,
        'partition_nodes': 2,
        
        'enable_hotspot': False,
        'hotspot_windows': [[0, 120]],
        'hotspot_regions': [['asia-east2-b']],  # one list per window
        # Per-window per-region hotspot node counts (align with hotspot_regions window entry).
        # Example: [[1, 1, 2]] => 1 from asia-southeast1-b, 1 from us-central1-c, 2 from us-central1-f.
        'hotspot_nodes': [[5]],
        # Optional: per-window per-region rates (align with hotspot_regions window entry).
        # Example below means asia-east2-a uses 0.9, us-central1-c uses 0.6.
        'hotspot_region_rates': [[0.9]],
    }
    node_params = {
        'timeout_delay': 5_000,  # ms
        'header_size': 32,  # bytes
        'max_header_delay': 5000,  # ms
        'gc_depth': 50,  # rounds
        'sync_retry_delay': 5000,  # ms
        'sync_retry_nodes': 3,  # number of nodes
        'batch_size': 500_000,  # bytes
        'max_batch_delay': 5000,  # ms
        'use_optimistic_tips': True,
        'use_parallel_proposals': True,
        'k': 4,
        'epoch_slots': 20,
        'window_size': 12,
        'applied_begin': 18,
        'use_fast_path': True,
        'fast_path_timeout': 100,
        'use_ride_share': False,
        'car_timeout': 2000,
        'cut_condition_type': 3,

        'simulate_asynchrony': False,
        'asynchrony_type': [6],

        'asynchrony_start': [0], #s
        'asynchrony_duration': [120], #s
        "affected_nodes":[1],
        # Optional: region-based selection of async/malicious nodes (per window).
        # Each region gets `asynchrony_nodes[w]` nodes. E.g. regions=[A,B], n=1 => 1 from A + 1 from B.
        'asynchrony_nodes': [ ],  # per region when asynchrony_regions is set
        # 'asynchrony_regions': [['us-central1-f', 'us-central1-c']],  # one list per window
        # Optional: per-window per-region egress penalty (ms), aligned with asynchrony_regions.
        # Example: [['us-central1-c', 'asia-east2-a']], [[300, 120]] => us-central1-c=300ms, asia-east2-a=120ms.
        'asynchrony_regions': [[ ]],
        'egress_penalty': [[]],

        'use_fast_sync': True,
        'use_exponential_timeouts': True,

        # Ablation: aggregation strategy for global state.
        # "normal"  -> original behaviour (max for growth_rates, median for reward/fpr).
        # "mean"    -> arithmetic mean for all three metrics.
        'aggregation_strategy': 'normal',

        # Ablation: data-pollution simulation.
        # List the 0-based node indices that act as polluters.
        'data_pollution_node_ids': [ ],
        # Probability [0.0, 1.0] that a polluter reports fake metrics.
        'data_pollution_prob': 1.0,
        'data_pollution_strategy': 'random_scale',
    }
    try:
        Bench(ctx).run(bench_params, node_params, debug)
    except BenchError as e:
        Print.error(e)


@task
def plot(ctx):
    ''' Plot performance using the logs generated by "fab remote" '''
    plot_params = {
        'faults': [0],
        'nodes': [4],
        'workers': [1, 4, 7, 10],
        'collocate': True,
        'tx_size': 512,
        'max_latency': [2_000, 2_500]
    }
    try:
        Ploter.plot(plot_params)
    except PlotError as e:
        Print.error(BenchError('Failed to plot performance', e))


@task
def kill(ctx):
    ''' Stop execution on all machines '''
    try:
        Bench(ctx).kill()
    except BenchError as e:
        Print.error(e)


@task
def logs(ctx):
    ''' Print a summary of the logs '''
    try:
        print(LogParser.process('./logs', faults='?').result())
    except ParseError as e:
        Print.error(BenchError('Failed to parse logs', e))


def _get_nodes_from_fab_info():
    """Get node metadata (name, region, ip) from InstanceManager/fab info."""
    manager = InstanceManager.make()
    ids_by_region, ips_by_region = manager._get(['STAGING', 'RUNNING'])

    nodes = []
    for region in sorted(ips_by_region.keys()):
        region_ips = ips_by_region.get(region, [])
        region_names = ids_by_region.get(region, [])
        for idx, ip in enumerate(region_ips):
            name = region_names[idx] if idx < len(region_names) else f"{region}-node-{idx}"
            nodes.append({"name": name, "region": region, "ip": ip})
    return nodes


def _local_ips():
    """Collect non-loopback local IPs (best effort)."""
    import socket
    import subprocess

    ips = set()
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        pass
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True).strip()
        if output:
            ips.update(output.split())
    except Exception:
        pass
    return {ip for ip in ips if ip and not ip.startswith("127.")}


def _detect_current_node_from_fab_info(nodes):
    """Best-effort detect current node by matching local IP to fab-info IPs."""
    local_ips = _local_ips()
    if not local_ips:
        return None
    for node in nodes:
        if node["ip"] in local_ips:
            return node
    return None


def _resolve_source_node(nodes, source_node):
    """Resolve source node by name, or auto-detect from local IPs."""
    if source_node is None:
        source = _detect_current_node_from_fab_info(nodes)
        if source is None:
            raise ValueError(
                "Could not determine current node from fab info; please pass --source-node"
            )
        print(
            f"Auto-detected current node from fab info: {source['name']} "
            f"({source['region']}, {source['ip']})"
        )
        return source

    print(f"Using specified source node: {source_node}")
    for node in nodes:
        if node["name"] == source_node:
            return node
    raise ValueError(f"Source node {source_node} not found in committee")


def _print_latency_stats(values, title, success_total=None):
    """Print mean/std/min/max for a latency list (ignores NaN)."""
    valid = [v for v in values if not np.isnan(v)]
    if not valid:
        print(f"\n=== No Valid Measurements ({title}) ===")
        return
    print(f"\n=== {title} ===")
    print(f"Mean Latency: {np.mean(valid):.2f} ms")
    print(f"Std Deviation: {np.std(valid):.2f} ms")
    print(f"Min Latency: {np.min(valid):.2f} ms")
    print(f"Max Latency: {np.max(valid):.2f} ms")
    if success_total is not None:
        print(f"Success Rate: {len(valid)}/{success_total}")


@task
def latency(ctx, cross_region=False, source_node=None, full_matrix=True):
    """
    Measure SSH latency between nodes.

    Args:
        cross_region: Also report cross-region latency (used with full_matrix).
        source_node: Source node name; None => auto-detect current node.
        full_matrix: If True, also measure the full node-to-node matrix.
    """
    import json
    import time
    from fabric import Connection
    from paramiko import RSAKey, SSHException
    from benchmark.utils import Print

    try:
        ctx.connect_kwargs.pkey = RSAKey.from_private_key_file("/home/ccclr0302/.ssh/gcp_rsa")
        connect_kwargs = ctx.connect_kwargs
    except (IOError, SSHException) as e:
        Print.error(f"Failed to load SSH key: {e}")
        return

    node_records = _get_nodes_from_fab_info()
    if not node_records:
        error_msg = "Failed to read node info from fab/InstanceManager"
        Print.error(BenchError(error_msg, Exception(error_msg)))
        return

    print(f"Total nodes from fab info: {len(node_records)}")
    try:
        source_record = _resolve_source_node(node_records, source_node)
    except ValueError as e:
        Print.error(BenchError(str(e), e))
        return

    source_node = source_record["name"]
    source_ip = source_record["ip"]
    target_nodes = [n for n in node_records if n["name"] != source_node]
    print(f"Source node: {source_node} ({source_record['region']})")
    print(f"Source node IP: {source_ip}")
    print(f"Will measure latency to {len(target_nodes)} other nodes")

    def ssh_latency(src_ip, dst_ip, repeat=1):
        if src_ip == dst_ip:
            return 0.0
        results = []
        for _ in range(repeat):
            try:
                # Source is whichever machine runs this task; we only SSH to dst.
                conn = Connection(host=dst_ip, user="ccclr0302", connect_kwargs=connect_kwargs)
                conn.run("echo warmup", hide=True, timeout=5)
                start = time.time()
                conn.run("echo hello", hide=True, timeout=5)
                end = time.time()
                conn.close()
                results.append((end - start) * 1000)
            except Exception as e:
                print(f"[Error] SSH {src_ip} → {dst_ip}: {e}")
                results.append(np.nan)
        valid = [x for x in results if not np.isnan(x)]
        return np.mean(valid) if valid else np.nan

    def store_latency(value):
        # Preserve historical falsy handling: 0 / None / NaN => NaN.
        return value if value else np.nan

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if full_matrix:
        print("Warning: Full matrix mode requested - this will measure all node pairs and may be slow")
        print("For normal operation, consider using single-source mode (default)")

        all_nodes = list(node_records)
        n_total = len(all_nodes)
        node_names = [n["name"] for n in all_nodes]
        name_to_idx = {n["name"]: i for i, n in enumerate(all_nodes)}
        matrix = np.full((n_total, n_total), np.nan)

        region_to_nodes = {}
        for node in all_nodes:
            region_to_nodes.setdefault(node["region"], []).append(node)
        region_nodes = list(region_to_nodes.items())
        for region, nodes in region_nodes:
            if len(nodes) < 2:
                Print.warn(f"[Skip] Region {region} has <2 nodes.")

        print("=== Measuring Full Node-to-Node Latency Matrix ===")
        for i, src in enumerate(all_nodes):
            for j, dst in enumerate(all_nodes):
                if i == j:
                    matrix[i][j] = 0.0
                    continue
                value = store_latency(ssh_latency(src["ip"], dst["ip"]))
                matrix[i][j] = value
                if not np.isnan(value):
                    print(f"  {src['name']} → {dst['name']}: {value:.2f} ms")
                else:
                    print(f"  {src['name']} → {dst['name']}: Failed")

        print("\n=== Calculating Region Summaries ===")
        for region, nodes in region_nodes:
            idxs = [name_to_idx[n["name"]] for n in nodes]
            values = [
                matrix[j][k]
                for j in idxs
                for k in idxs
                if j != k and not np.isnan(matrix[j][k])
            ]
            if values:
                print(f"  [Average] {region}: {np.mean(values):.2f} ms")
            else:
                print(f"  [Average] {region}: nan")

        if cross_region:
            print("\n=== Measuring Cross-region Latency ===")
            for region_i, nodes_i in region_nodes:
                for region_j, nodes_j in region_nodes:
                    if region_i == region_j:
                        continue
                    src, dst = nodes_i[0], nodes_j[0]
                    # Reuse full-matrix sample for the region representatives.
                    value = matrix[name_to_idx[src["name"]]][name_to_idx[dst["name"]]]
                    if not np.isnan(value):
                        print(f"[Cross] {region_i} → {region_j}: {value:.2f} ms")
                    else:
                        print(f"[Cross] {region_i} → {region_j}: Failed")

        print("\n=== Full Node Latency Matrix (ms) ===")
        print("Nodes:", node_names)
        print(matrix)

        off_diag = [
            matrix[i][j]
            for i in range(n_total)
            for j in range(n_total)
            if i != j and not np.isnan(matrix[i][j])
        ]
        _print_latency_stats(off_diag, "Full Matrix Statistics")

        np.save(f"latency_full_matrix_{timestamp}.npy", matrix)
        print("\nResults saved to:")
        print(f"  - latency_full_matrix_{timestamp}.npy")

    # Always emit single-source latency vector (consumed by callers via JSON markers).
    print("=== Measuring Latency from Current Node ===")
    latency_vector = []
    measured_count = 0
    for target in target_nodes:
        value = store_latency(ssh_latency(source_ip, target["ip"]))
        latency_vector.append(value)
        if not np.isnan(value):
            print(
                f"  {source_node} ({source_record['region']}) → "
                f"{target['name']} ({target['region']}): {value:.2f} ms"
            )
            measured_count += 1
        else:
            print(
                f"  {source_node} ({source_record['region']}) → "
                f"{target['name']} ({target['region']}): Failed"
            )

    print(f"\nSuccessfully measured latency to {measured_count}/{len(target_nodes)} nodes")
    _print_latency_stats(latency_vector, "Latency Statistics", success_total=len(latency_vector))

    output_data = {
        "source_node": source_node,
        "source_region": source_record["region"],
        "source_ip": source_ip,
        "target_nodes": [
            {"name": n["name"], "region": n["region"], "ip": n["ip"]} for n in target_nodes
        ],
        "latencies": [float(x) if not np.isnan(x) else None for x in latency_vector],
        "timestamp": timestamp,
    }
    print("LATENCY_VECTOR_JSON_START")
    print(json.dumps(output_data))
    print("LATENCY_VECTOR_JSON_END")

    np.save(f"latency_vector_{timestamp}.npy", np.array(latency_vector))
    print(f"\nResults also saved to: latency_vector_{timestamp}.npy", file=sys.stderr)

