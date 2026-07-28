# Copyright(C) Facebook, Inc. and its affiliates.
from collections import defaultdict

from benchmark.utils import Print, BenchError
from benchmark.cloudlab_settings import CloudLabSettings, CloudLabSettingsError


class CloudLabError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CloudLabInstanceManager:
    """
    Manages CloudLab nodes.

    Unlike GCP/AWS InstanceManager, CloudLab nodes are pre-allocated and listed
    in cloudlab_settings.json. create/start/stop/terminate are therefore no-ops
    that only print guidance; hosts() always returns the configured nodes.
    """

    INSTANCE_NAME = 'cloudlab-node'
    SECURITY_GROUP_NAME = 'cloudlab'

    def __init__(self, settings):
        assert isinstance(settings, CloudLabSettings)
        self.settings = settings

    @classmethod
    def make(cls, settings_file='cloudlab_settings.json'):
        try:
            return cls(CloudLabSettings.load(settings_file))
        except CloudLabSettingsError as e:
            raise BenchError('Failed to load CloudLab settings', e)

    def _get(self, state=None):
        """
        Return (ids, ips) grouped by region from pre-allocated CloudLab hosts.

        `state` is accepted for API compatibility with GCP/AWS InstanceManager
        and is ignored because CloudLab nodes are static entries in settings.
        """
        ids, ips = defaultdict(list), defaultdict(list)
        for i, host in enumerate(self.settings.hosts):
            region = host.get('region', 'default')
            hostname = host['hostname']
            ids[region] += [f'{self.INSTANCE_NAME}-{region}-{i}']
            ips[region] += [hostname]
        return ids, ips

    def create_instances(self, instances):
        """
        CloudLab nodes are pre-allocated. Update cloudlab_settings.json instead
        of creating VMs via a cloud API.
        """
        assert isinstance(instances, int) and instances > 0
        available = len(self.settings.hosts)
        Print.heading(
            f'CloudLab uses pre-allocated nodes ({available} configured). '
            f'Requested {instances} per region — edit cloudlab_settings.json hosts.'
        )
        all_ips = self.hosts(flat=True)
        with open('nodes.txt', 'w') as f:
            for ip in all_ips:
                f.write(f'{ip}\n')
        Print.info(f"IP of {len(all_ips)} nodes has been written in 'nodes.txt'")

    def terminate_instances(self):
        Print.heading(
            'CloudLab nodes are pre-allocated; nothing to terminate. '
            'Remove hosts from cloudlab_settings.json if needed.'
        )

    def start_instances(self, max):
        size = min(max, len(self.settings.hosts))
        Print.heading(
            f'CloudLab nodes are always reachable via SSH '
            f'({size} of {len(self.settings.hosts)} hosts available)'
        )

    def stop_instances(self):
        Print.heading(
            'CloudLab nodes are pre-allocated; nothing to stop. '
            'Use fab cloudlab-kill to stop benchmark processes.'
        )

    def hosts(self, flat=False):
        try:
            _, ips = self._get()
            return [x for y in ips.values() for x in y] if flat else dict(ips)
        except Exception as e:
            raise BenchError('Failed to gather CloudLab hosts', CloudLabError(str(e)))

    def internal_hosts(self, flat=False):
        """Alias of hosts() for API compatibility with remote Bench._select_hosts_config."""
        return self.hosts(flat=flat)

    def print_info(self):
        hosts = self.hosts()
        key = self.settings.key_path
        text = ''
        for region, ips in hosts.items():
            text += f'\n Region: {region.upper()}\n'
            for i, ip in enumerate(ips):
                new_line = '\n' if (i + 1) % 6 == 0 else ''
                text += (
                    f'{new_line} {i}\tssh -i {key} '
                    f'{self.settings.username}@{ip}\n'
                )
        print(
            '\n'
            '----------------------------------------------------------------\n'
            ' CLOUDLAB INFO:\n'
            '----------------------------------------------------------------\n'
            f' Available machines: {sum(len(x) for x in hosts.values())}\n'
            f'{text}'
            '----------------------------------------------------------------\n'
        )
