# Copyright(C) Facebook, Inc. and its affiliates.
from datetime import datetime
from glob import glob
from multiprocessing import Pool
from os.path import join
from re import findall, search
from statistics import mean
import json
import os
from benchmark.utils import Print


class ParseError(Exception):
    pass


class LogParser:
    def __init__(self, clients, primaries, workers, faults=0, parameters_path='.parameters.json'):
        inputs = [clients, primaries, workers]
        assert all(isinstance(x, list) for x in inputs)
        assert all(isinstance(x, str) for y in inputs for x in y)
        assert all(x for x in inputs)

        self.faults = faults
        if isinstance(faults, int):
            self.committee_size = len(primaries) + int(faults)
            self.workers = len(workers) // len(primaries)
        else:
            self.committee_size = '?'
            self.workers = '?'

        self.parameters_json = {}
        if os.path.exists(parameters_path):
            with open(parameters_path, 'r') as f:
                self.parameters_json = json.load(f)

        # Parse the clients logs.
        try:
            with Pool() as p:
                results = p.map(self._parse_clients, clients)
        except (ValueError, IndexError, AttributeError) as e:
            raise ParseError(f'Failed to parse clients\' logs: {e}')
        self.size, self.rate, self.start, misses, self.sent_samples, self.hotspot_info \
            = zip(*results)
        self.misses = sum(misses)
        self.raw_clients = clients

        # Parse the primaries logs.
        try:
            with Pool() as p:
                results = p.map(self._parse_primaries, primaries)
        except (ValueError, IndexError, AttributeError) as e:
            raise ParseError(f'Failed to parse nodes\' logs: {e}')
        proposals, commits, self.configs, primary_ips = zip(*results)
        self.proposals = self._merge_results([x.items() for x in proposals])
        self.commits = self._merge_results([x.items() for x in commits])

        # Parse the workers logs.
        try:
            with Pool() as p:
                results = p.map(self._parse_workers, workers)
        except (ValueError, IndexError, AttributeError) as e:
            raise ParseError(f'Failed to parse workers\' logs: {e}')
        sizes, self.received_samples, workers_ips = zip(*results)
        self.sizes = {
            k: v for x in sizes for k, v in x.items() if k in self.commits
        }

        # Determine whether the primary and the workers are collocated.
        self.collocate = set(primary_ips) == set(workers_ips)

        # Check whether clients missed their target rate.
        if self.misses != 0:
            Print.warn(
                f'Clients missed their target rate {self.misses:,} time(s)'
            )

    def _parse_clients(self, log):
        if search(r'Error', log) is not None:
            raise ParseError('Client(s) panicked')
        
        def parse_int(pattern):
            m = search(pattern, log)
            return int(m.group(1)) if m else None
        
        def parse_time(pattern):
            m = search(pattern, log)
            return self._to_posix(m.group(1)) if m else None
        
        size = parse_int(r'Transactions size: (\d+)')
        rate = parse_int(r'Transactions rate: (\d+)')
        start = parse_time(r'\[(.*Z) .* Start ')
        misses = len(findall(r'rate too high', log))
        
        tmp_new = findall(r'\[(.*Z) .* Sending sample transaction (\d+) with timestamp (\d+)', log)
        if tmp_new:
            samples = {}
            for log_time, tx_id, timestamp_us in tmp_new:
                samples[int(tx_id)] = {
                    'log_time': self._to_posix(log_time),
                    'timestamp_us': int(timestamp_us)
                }
        else:
            tmp = findall(r'\[(.*Z) .* sample transaction (\d+)', log)
            samples = {int(s): self._to_posix(t) for t, s in tmp} if tmp else {}
        
        hotspot_info = {}
        node_id = parse_int(r'Node ID: (\d+)')
        if node_id is not None:
            hotspot_info['node_id'] = node_id
        
        total_nodes = parse_int(r'Total nodes: (\d+)')
        if total_nodes is not None:
            hotspot_info['total_nodes'] = total_nodes
        
        hotspot_config_match = search(r'Hotspot configuration enabled:', log)
        if hotspot_config_match:
            hotspot_info['enabled'] = True
            windows_matches = findall(r'Window \d+: (\d+)s-(\d+)s, (\d+) hotspot nodes, ([\d.]+)% rate increase', log)
            if windows_matches:
                windows = []
                for start_s, end_s, nodes, rate_pct in windows_matches:
                    windows.append({
                        'start': int(start_s),
                        'end': int(end_s),
                        'nodes': int(nodes),
                        'rate_increase': float(rate_pct) / 100.0
                    })
                hotspot_info['windows'] = windows
        else:
            hotspot_info['enabled'] = False
        
        rate_changes = findall(r'Current transaction rate: ([\d.]+) tx/s at time (\d+)s', log)
        if rate_changes:
            hotspot_info['rate_changes'] = [(float(rate), int(time)) for rate, time in rate_changes]
        
        return size, rate, start, misses, samples, hotspot_info

    def _merge_results(self, input):
        # Keep the earliest timestamp.
        merged = {}
        for x in input:
            for k, v in x:
                if not k in merged or merged[k] > v:
                    merged[k] = v
        return merged

    def _parse_primaries(self, log):
        if search(r'(?:panicked|Error)', log) is not None:
            raise ParseError('Primary(s) panicked')

        tmp = findall(r'\[(.*Z) .* Created B\d+\([^ ]+\) -> ([^ ]+=)', log)
        tmp = [(d, self._to_posix(t)) for t, d in tmp]
        proposals = self._merge_results([tmp])

        tmp = findall(r'\[(.*Z) .* Committed B\d+\([^ ]+\) -> ([^ ]+=)', log)
        tmp = [(d, self._to_posix(t)) for t, d in tmp]
        commits = self._merge_results([tmp])

        def parse_int(pattern):
            m = search(pattern, log)
            return int(m.group(1)) if m else None
        def parse_bool(pattern):
            m = search(pattern, log)
            return m.group(1) == 'True' if m else None
        def parse_list(pattern):
            m = search(pattern, log)
            return eval(m.group(1)) if m else None

        configs = {
            'timeout_delay': parse_int(r'Timeout delay .* (\d+)'),
            'header_size': parse_int(r'Header size .* (\d+)'),
            'max_header_delay': parse_int(r'Max header delay .* (\d+)'),
            'gc_depth': parse_int(r'Garbage collection depth .* (\d+)'),
            'sync_retry_delay': parse_int(r'Sync retry delay .* (\d+)'),
            'sync_retry_nodes': parse_int(r'Sync retry nodes .* (\d+)'),
            'batch_size': parse_int(r'Batch size .* (\d+)'),
            'max_batch_delay': parse_int(r'Max batch delay .* (\d+)'),
            'use_optimistic_tips': parse_bool(r'Use optimistic tips: (True|False)'),
            'use_parallel_proposals': parse_bool(r'Use parallel proposals: (True|False)'),
            'k': parse_int(r'k: (\d+)'),
            'use_fast_path': parse_bool(r'Use fast path: (True|False)'),
            'fast_path_timeout': parse_int(r'Fast path timeout: (\d+)'),
            'use_ride_share': parse_bool(r'Use ride share: (True|False)'),
            'car_timeout': parse_int(r'Car timeout: (\d+)'),
            'simulate_asynchrony': parse_bool(r'Simulate asynchrony: (True|False)'),
            'asynchrony_type': parse_list(r'Asynchrony type: (\[.*?\])'),
            'asynchrony_start': parse_list(r'Asynchrony start: (\[.*?\])'),
            'asynchrony_duration': parse_list(r'Asynchrony duration: (\[.*?\])'),
            'affected_nodes': parse_list(r'Affected nodes: (\[.*?\])'),
            'egress_penalty': parse_int(r'Egress penalty: (\d+)'),
            'use_fast_sync': parse_bool(r'Use fast sync: (True|False)'),
            'use_exponential_timeouts': parse_bool(r'Use exponential timeouts: (True|False)'),
            'cut_condition_type': parse_int(r'Cut condition type: (\[.*?\])'),
        }

        m = search(r'booted on (\d+.\d+.\d+.\d+)', log)
        ip = m.group(1) if m else None

        return proposals, commits, configs, ip

    def _parse_workers(self, log):
        if search(r'(?:panic|Error)', log) is not None:
            raise ParseError('Worker(s) panicked')

        tmp = findall(r'Batch ([^ ]+) contains (\d+) B', log)
        sizes = {d: int(s) for d, s in tmp}

        tmp = findall(r'Batch ([^ ]+) contains sample tx (\d+)', log)
        samples = {int(s): d for d, s in tmp}

        ip = search(r'booted on (\d+.\d+.\d+.\d+)', log).group(1)

        return sizes, samples, ip

    def _to_posix(self, string):
        x = datetime.fromisoformat(string.replace('Z', '+00:00'))
        return datetime.timestamp(x)

    def _consensus_throughput(self):
        if not self.commits:
            return 0, 0, 0
        start, end = min(self.proposals.values()), max(self.commits.values())
        duration = end - start
        bytes = sum(self.sizes.values())
        bps = bytes / duration
        tps = bps / self.size[0]
        return tps, bps, duration

    def _consensus_latency(self):
        latency = [c - self.proposals[d] for d, c in self.commits.items()]
        return mean(latency) if latency else 0

    def _end_to_end_throughput(self):
        if not self.commits:
            return 0, 0, 0
        
        start = min(self.start)
        end = max(self.commits.values())
        duration = end - start
        
        total_bytes = sum(self.sizes.values())
        
        bps = total_bytes / duration if duration > 0 else 0
        tps = bps / self.size[0] if self.size and self.size[0] > 0 else 0
        
        return tps, bps, duration

    def _end_to_end_latency(self):
        latency = []
        list_latencies = []
        first_start = 0
        set_first = True
        
        for sent, received in zip(self.sent_samples, self.received_samples):
            for tx_id, batch_id in received.items():
                if batch_id in self.commits:
                    assert tx_id in sent  # We receive txs that we sent.
                    
                    # 检查sent[tx_id]的格式
                    if isinstance(sent[tx_id], dict) and 'timestamp_us' in sent[tx_id]:
                        # 新格式：使用微秒时间戳
                        start = sent[tx_id]['timestamp_us'] / 1_000_000  # 转换为秒
                    else:
                        # 旧格式：直接使用时间戳
                        start = sent[tx_id]
                    
                    end = self.commits[batch_id]
                    
                    # 确保延迟为正数
                    if end >= start:
                        if set_first:
                            first_start = start
                            first_end = end
                            set_first = False
                        latency += [end - start]
                        list_latencies += [(start - first_start, end - first_start, end - start)]
                    else:
                        # 记录负延迟的警告（但不包含在结果中）
                        print(f"Warning: Negative latency for tx {tx_id}: {end - start:.6f}s")

        list_latencies.sort(key=lambda tup: tup[0])
        with open('latencies.txt', 'w') as f:
            for line in list_latencies:
                f.write(str(line[0]) + ',' + str(line[1]) + ',' + str((line[2])) + '\n')
        
        return mean(latency) if latency else 0

    def _analyze_hotspot_performance(self):
        """
        增强热点性能分析
        """
        hotspot_analysis = {}
        
        hotspot_enabled = any(info.get('enabled', False) for info in self.hotspot_info)
        hotspot_analysis['enabled'] = hotspot_enabled
        
        if hotspot_enabled:
            node_performances = {}
            total_transactions = 0
            
            for i, info in enumerate(self.hotspot_info):
                node_id = info.get('node_id', i)
                base_rate = self.rate[i] if i < len(self.rate) else 0
                
                node_performance = {
                    'base_rate': base_rate,
                    'node_id': node_id
                }
                
                if i < len(self.sent_samples):
                    if i < len(self.raw_clients):
                        import re
                        last_tx_matches = re.findall(r'Sending sample transaction (\d+)', self.raw_clients[i])
                        last_tx_number = int(last_tx_matches[-1])
                        node_tx_count = last_tx_number
                    else:
                        node_tx_count = len(self.sent_samples[i])
                    node_performance['transactions_sent'] = node_tx_count
                    total_transactions += node_tx_count
                
                _, _, duration = self._end_to_end_throughput()
                
                actual_rate = node_tx_count / duration
                node_performance['actual_rate'] = actual_rate
                node_performances[node_id] = node_performance
            
            hotspot_analysis['node_performances'] = node_performances
            hotspot_analysis['total_transactions'] = total_transactions
            
            if self.hotspot_info and 'windows' in self.hotspot_info[0]:
                hotspot_analysis['windows'] = self.hotspot_info[0]['windows']
            
            if node_performances:
                total_base_rate = sum(perf['base_rate'] for perf in node_performances.values())
                total_actual_rate = sum(perf['actual_rate'] for perf in node_performances.values())
                if total_base_rate > 0:
                    hotspot_analysis['throughput_increase'] = (total_actual_rate - total_base_rate) / total_base_rate
        
        return hotspot_analysis

    def result(self, extra_config=None):
        cfg = self.parameters_json.copy()
        for k, v in self.configs[0].items():
            if v is not None:
                cfg[k] = v
        if isinstance(extra_config, dict):
            cfg.update(extra_config)
        
        consensus_latency = (self._consensus_latency() or 0) * 1_000
        consensus_tps, consensus_bps, _ = self._consensus_throughput()
        consensus_tps = consensus_tps or 0
        consensus_bps = consensus_bps or 0

        end_to_end_tps, end_to_end_bps, duration = self._end_to_end_throughput()
        end_to_end_tps = end_to_end_tps or 0
        end_to_end_bps = end_to_end_bps or 0
        duration = duration or 0

        end_to_end_latency = (self._end_to_end_latency() or 0) * 1_000

        hotspot_analysis = self._analyze_hotspot_performance()
        
        def format_key(key):
            return key.replace('_', ' ').capitalize()

        def format_value(value):
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=True)
            return str(value)

        preferred_order = [
            'timeout_delay',
            'header_size',
            'max_header_delay',
            'gc_depth',
            'sync_retry_delay',
            'sync_retry_nodes',
            'batch_size',
            'max_batch_delay',
            'use_optimistic_tips',
            'use_parallel_proposals',
            'k',
            'use_fast_path',
            'fast_path_timeout',
            'use_ride_share',
            'car_timeout',
            'simulate_asynchrony',
            'asynchrony_type',
            'asynchrony_start',
            'asynchrony_duration',
            'affected_nodes',
            'egress_penalty',
            'use_fast_sync',
            'use_exponential_timeouts',
            'cut_condition_type',
        ]
        preferred_labels = {
            'gc_depth': 'GC depth',
            'k': 'k',
        }

        ordered_keys = [k for k in preferred_order if k in cfg]
        ordered_keys.extend(k for k in cfg.keys() if k not in ordered_keys)

        all_config_lines = ''.join(
            f' {preferred_labels.get(k, format_key(k))}: {format_value(cfg[k])}\n'
            for k in ordered_keys
        )

        hotspot_summary = ""
        if hotspot_analysis['enabled']:
            hotspot_summary += f' Enable hotspot: {hotspot_analysis["enabled"]}\n'
            
            if 'total_transactions' in hotspot_analysis:
                hotspot_summary += f' Total transactions sent: {hotspot_analysis["total_transactions"]}\n'
            
            if 'windows' in hotspot_analysis:                
                for i, window in enumerate(hotspot_analysis['windows']):
                    hotspot_summary += f' Window {i+1}: {window["start"]}s-{window["end"]}s, '
                    hotspot_summary += f'{window["nodes"]} nodes, {window["rate_increase"]*100:.1f}% increase\n'
            
            if 'node_performances' in hotspot_analysis:
                hotspot_summary += ' Node performances:\n'
                for node_id, perf in hotspot_analysis['node_performances'].items():
                    hotspot_summary += f' Node {node_id}: base={perf["base_rate"]}, '
                    hotspot_summary += f' actual={perf["actual_rate"]:.1f}, '
                    hotspot_summary += f' txs={perf.get("transactions_sent", "N/A")}\n'
        else:
            hotspot_summary += f' Enable hotspot: False\n'
        

        return (
            '\n'
            '-----------------------------------------\n'
            ' SUMMARY:\n'
            '-----------------------------------------\n'
            ' + CONFIG:\n'
            f' Faults: {self.faults} node(s)\n'
            f' Committee size: {self.committee_size} node(s)\n'
            f' Worker(s) per node: {self.workers} worker(s)\n'
            f' Collocate primary and workers: {self.collocate}\n'
            f' Input rate: {", ".join(str(r) for r in self.rate if r is not None)} tx/s\n'
            f' Transaction size: {self.size[0]:,} B\n'
            f' Execution time: {round(duration):,} s\n'
            '\n'
            f'{all_config_lines}'
            '\n'
            ' + HOTSPOT CONFIG:\n'
            f'{hotspot_summary}'
            # f'{hotspot_analysis}'
            '\n'
            ' + RESULTS:\n'
            f' Consensus TPS: {round(consensus_tps):,} tx/s\n'
            f' Consensus BPS: {round(consensus_bps):,} B/s\n'
            f' Consensus latency: {round(consensus_latency):,} ms\n'
            '\n'
            f' End-to-end TPS: {round(end_to_end_tps):,} tx/s\n'
            f' End-to-end BPS: {round(end_to_end_bps):,} B/s\n'
            f' End-to-end latency: {round(end_to_end_latency):,} ms\n'
            '-----------------------------------------\n'
        )
    
    @classmethod
    def process(cls, directory, faults=0):
        assert isinstance(directory, str)

        clients = []
        for filename in sorted(glob(join(directory, 'client-*.log'))):
            with open(filename, 'r') as f:
                clients += [f.read()]
        primaries = []
        for filename in sorted(glob(join(directory, 'primary-*.log'))):
            with open(filename, 'r') as f:
                primaries += [f.read()]
        workers = []
        for filename in sorted(glob(join(directory, 'worker-*.log'))):
            with open(filename, 'r') as f:
                workers += [f.read()]

        return cls(clients, primaries, workers, faults=faults)