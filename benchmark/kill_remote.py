#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from fabric import Connection, ThreadingGroup as Group
from fabric.exceptions import GroupException
from paramiko import RSAKey
from paramiko.ssh_exception import PasswordRequiredException, SSHException

sys.path.append(os.path.join(os.path.dirname(__file__), "benchmark"))

from benchmark.cloudlab_settings import CloudLabSettings, CloudLabSettingsError

KILL_CMDS = [
    "sudo su || true",
    "pkill -f autopilot || true",
    "pkill -f '[.]/node ' || true",
    "pkill -f benchmark_client || true",
    "tmux kill-server || true",
]

def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_ssh_connect_kwargs(settings: CloudLabSettings) -> dict:
    try:
        password = settings.ssh_key_password or os.environ.get("SSH_KEY_PASSWORD")
        if password:
            pkey = RSAKey.from_private_key_file(settings.key_path, password=password)
        else:
            pkey = RSAKey.from_private_key_file(settings.key_path)
        return {"pkey": pkey}
    except (IOError, PasswordRequiredException, SSHException) as e:
        raise RuntimeError(f"Failed to load SSH key {settings.key_path}: {e}") from e


def kill_all(settings_path: Path) -> None:
    try:
        settings = CloudLabSettings.load(str(settings_path))
    except (OSError, CloudLabSettingsError) as e:
        raise RuntimeError(f"Failed to load settings {settings_path}: {e}") from e

    hosts = [h["hostname"] for h in settings.hosts]
    if not hosts:
        raise RuntimeError(f"No hosts in {settings_path}")

    connect = _load_ssh_connect_kwargs(settings)
    cmd = " ; ".join(KILL_CMDS)

    print(f"Killing remote processes on {len(hosts)} host(s): {', '.join(hosts)}")
    print(f"Command: {cmd}")

    try:
        g = Group(*hosts, user=settings.username, connect_kwargs=connect)
        results = g.run(cmd, hide=True, warn=True)
    except GroupException as e:
        raise RuntimeError(f"SSH group failure: {e}") from e

    for host, result in results.items():
        # ThreadingGroup keys may be Connection objects.
        name = host.host if isinstance(host, Connection) else str(host)
        if result.ok:
            print(f"  OK  {name}")
        else:
            err = (result.stderr or result.stdout or "").strip()
            print(f"  FAIL {name}: exit={result.exited} {err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kill autopilot processes on all remote hosts")
    parser.add_argument(
        "--settings",
        default=str(_script_dir() / "cloudlab_settings.json"),
        help="Path to cloudlab settings JSON (default: ./cloudlab_settings.json)",
    )
    args = parser.parse_args()

    try:
        kill_all(Path(args.settings))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
