from unittest import TestCase

from benchmark.commands import CommandMaker
from benchmark.config import BenchParameters, ConfigError


def _parameters(**overrides):
    values = {
        'faults': 0,
        'nodes': [4],
        'rate': [40_000],
        'workers': 1,
        'collocate': True,
        'tx_size': 512,
        'duration': 1200,
        'runs': 1,
        'enable_rl': True,
        'rl_algo': 'coverage_round_robin',
        'enable_checkpoint': False,
        'enable_cmab_transition_export': True,
        'cmab_transition_export_dir': '/local/autopilot_offline_data',
        'cmab_environment_label': 'A',
        'coverage_seed': 17,
        'simulate_partition': False,
        'partition_nodes': 2,
        'partition_start': 5,
        'partition_duration': 5,
        'enable_hotspot': False,
    }
    values.update(overrides)
    return values


class CoverageConfigTests(TestCase):
    def test_accepts_coverage_collection_configuration(self):
        parameters = BenchParameters(_parameters())

        self.assertEqual(parameters.rl_algo, 'coverage_round_robin')
        self.assertEqual(parameters.coverage_seed, 17)

    def test_requires_transition_export(self):
        with self.assertRaisesRegex(
            ConfigError, 'enable_cmab_transition_export=true'
        ):
            BenchParameters(
                _parameters(enable_cmab_transition_export=False)
            )

    def test_rejects_policy_checkpoint(self):
        with self.assertRaisesRegex(ConfigError, 'cannot load'):
            BenchParameters(
                _parameters(
                    enable_checkpoint=True,
                    checkpoint_path='/local/dqn.pt',
                )
            )

    def test_controller_command_contains_coverage_arguments(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='coverage_round_robin',
            dqn_action_endpoints='0@10.0.0.1:19100,1@10.0.0.2:19100',
            cmab_transition_export_dir='/local/autopilot_offline_data',
            cmab_environment_label='A',
            cmab_transition_run_id='coverage-run1',
            coverage_seed=17,
        )

        self.assertIn('--rl-algo coverage_round_robin', command)
        self.assertIn('--dqn-action-endpoints', command)
        self.assertIn('--coverage-seed 17', command)
        self.assertIn('--cmab-transition-run-id coverage-run1', command)

    def test_cmab_command_does_not_gain_coverage_arguments(self):
        command = CommandMaker.run_controller(
            node_index=0,
            repo_name='autopilot',
            log_dir='/local/logs',
            parameters_file='/local/.parameters.json',
            rl_algo='cmab',
        )

        self.assertNotIn('--coverage-seed', command)
