# Copyright(C) Facebook, Inc. and its affiliates.
from json import load, JSONDecodeError


class CloudLabSettingsError(Exception):
    pass


class CloudLabSettings:
    def __init__(self, key_name, key_path, base_port, repo_name, repo_url,
                 branch, username, hosts, home=None, ssh_key_password=None):
        inputs_str = [
            key_name, key_path, repo_name, repo_url, branch, username
        ]
        ok = all(isinstance(x, str) for x in inputs_str)
        ok &= isinstance(base_port, int)
        ok &= isinstance(hosts, list)
        ok &= len(hosts) > 0
        if home is not None:
            ok &= isinstance(home, str)
        if ssh_key_password is not None:
            ok &= isinstance(ssh_key_password, str)

        for host in hosts:
            if not isinstance(host, dict):
                ok = False
                break
            if 'hostname' not in host or not isinstance(host['hostname'], str):
                ok = False
                break

        if not ok:
            raise CloudLabSettingsError('Invalid settings types')

        self.key_name = key_name
        self.key_path = key_path

        self.base_port = base_port

        self.repo_name = repo_name
        self.repo_url = repo_url
        self.branch = branch

        self.username = username
        self.home = home if home else f'/users/{username}'
        self.ssh_key_password = ssh_key_password

        # Normalize hosts: ensure region exists.
        self.hosts = [
            {
                'hostname': h['hostname'],
                'username': h.get('username', username),
                'port': h.get('port', 22),
                'region': h.get('region') or h.get('description', 'default'),
            }
            for h in hosts
        ]
        self.regions = sorted({h['region'] for h in self.hosts})

    @classmethod
    def load(cls, filename):
        try:
            with open(filename, 'r') as f:
                data = load(f)

            key_path = data.get('ssh_key_path') or data['key']['path']
            key_name = data.get('key', {}).get('name', 'cloudlab')
            hosts = data.get('servers') or data['hosts']
            username = data.get('username')
            if not username:
                # Fallback: use the first host username when global username is omitted.
                username = hosts[0].get('username', 'root') if hosts else 'root'

            return cls(
                key_name,
                key_path,
                data['port'],
                data['repo']['name'],
                data['repo']['url'],
                data['repo']['branch'],
                username,
                hosts,
                home=data.get('home'),
                ssh_key_password=data.get('ssh_key_password'),
            )
        except (OSError, JSONDecodeError) as e:
            raise CloudLabSettingsError(str(e))
        except KeyError as e:
            raise CloudLabSettingsError(f'Malformed settings: missing key {e}')
