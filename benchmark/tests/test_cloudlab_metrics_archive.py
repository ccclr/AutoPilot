import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from benchmark.cloudlab_remote import CloudLabBench


class _Committee:
    def primary_addresses(self, faults):
        return ['10.0.0.1:9000']


class _SuccessfulConnection:
    calls = []

    def __init__(self, host, user, connect_kwargs):
        self.calls.append(('connect', host, user, connect_kwargs))

    def run(self, command, hide, warn):
        self.calls.append(('run', command, hide, warn))
        return SimpleNamespace(ok=True, stdout='242', stderr='', exited=0)


class _LocalConnection:
    def __init__(self, host, user, connect_kwargs):
        pass

    def run(self, command, hide, warn):
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )
        return SimpleNamespace(
            ok=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exited=completed.returncode,
        )


def _bench_parameters(**overrides):
    values = {
        'enable_rl': True,
        'rl_algo': 'cmab',
        'enable_cmab_transition_export': True,
        'cmab_transition_export_dir': '/local/autopilot_offline_data',
        'cmab_environment_label': 'A',
        'faults': 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MetricsArchiveTests(TestCase):
    def setUp(self):
        self.bench = CloudLabBench.__new__(CloudLabBench)
        self.bench.home = '/local'
        self.bench.settings = SimpleNamespace(username='tester')
        self.bench.connect = {'key_filename': '/tmp/test-key'}
        _SuccessfulConnection.calls.clear()

    @patch('benchmark.cloudlab_remote.Print.info')
    @patch('benchmark.cloudlab_remote.Connection', _SuccessfulConnection)
    def test_archives_node0_metrics_into_the_matching_run(self, info):
        self.bench._archive_cmab_metrics(
            _Committee(),
            _bench_parameters(),
            '20260824-090824-nodes4-rate40000-run1',
        )

        run_call = next(
            call for call in _SuccessfulConnection.calls if call[0] == 'run'
        )
        command = run_call[1]
        self.assertIn('/local/metrics-0', command)
        self.assertIn(
            '/local/autopilot_offline_data/A/'
            '20260824-090824-nodes4-rate40000-run1/metrics-0',
            command,
        )
        self.assertIn('.metrics-0.tmp', command)
        self.assertEqual(run_call[2:], (True, True))
        info.assert_called_once_with(
            'Archived CMAB metrics: 242 files -> '
            '/local/autopilot_offline_data/A/'
            '20260824-090824-nodes4-rate40000-run1/metrics-0'
        )

    @patch('benchmark.cloudlab_remote.Connection')
    def test_does_nothing_when_transition_export_is_disabled(self, connection):
        self.bench._archive_cmab_metrics(
            _Committee(),
            _bench_parameters(enable_cmab_transition_export=False),
            'run1',
        )

        connection.assert_not_called()

    @patch('benchmark.cloudlab_remote.Connection', _SuccessfulConnection)
    def test_coverage_collection_archives_its_metrics(self):
        self.bench._archive_cmab_metrics(
            _Committee(),
            _bench_parameters(rl_algo='coverage_round_robin'),
            'coverage-run1',
        )

        run_call = next(
            call for call in _SuccessfulConnection.calls if call[0] == 'run'
        )
        self.assertIn(
            '/local/autopilot_offline_data/A/coverage-run1/metrics-0',
            run_call[1],
        )

    @patch('benchmark.cloudlab_remote.Print.info')
    @patch('benchmark.cloudlab_remote.Connection', _LocalConnection)
    def test_copies_metrics_atomically(self, info):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.bench.home = str(root)
            metrics = root / 'metrics-0'
            metrics.mkdir()
            (metrics / 'global_state_epoch_0.json').write_text(
                '{}', encoding='utf-8'
            )
            (metrics / 'epoch_0_slot_0.json').write_text(
                '{}', encoding='utf-8'
            )

            export_root = root / 'offline'
            run_dir = export_root / 'A' / 'run1'
            run_dir.mkdir(parents=True)
            (run_dir / 'transitions.jsonl').write_text('', encoding='utf-8')

            self.bench._archive_cmab_metrics(
                _Committee(),
                _bench_parameters(
                    cmab_transition_export_dir=str(export_root)
                ),
                'run1',
            )

            archived = run_dir / 'metrics-0'
            self.assertEqual(
                sorted(path.name for path in archived.iterdir()),
                ['epoch_0_slot_0.json', 'global_state_epoch_0.json'],
            )
            self.assertFalse((run_dir / '.metrics-0.tmp').exists())
            info.assert_called_once_with(
                f'Archived CMAB metrics: 2 files -> {archived}'
            )

    @patch('benchmark.cloudlab_remote.Print.warn')
    @patch('benchmark.cloudlab_remote.Connection')
    def test_archive_failure_does_not_raise(self, connection, warn):
        connection.return_value.run.return_value = SimpleNamespace(
            ok=False,
            stdout='',
            stderr='metrics directory missing',
            exited=1,
        )

        self.bench._archive_cmab_metrics(
            _Committee(),
            _bench_parameters(),
            'run1',
        )

        warn.assert_called_once()
