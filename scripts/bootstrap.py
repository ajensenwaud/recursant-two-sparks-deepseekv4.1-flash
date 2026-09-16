#!/usr/bin/env python3
"""Configure and install a pinned two-Spark release; never install host software."""
import json
import os
import pathlib
import sys
import cluster
from artifacts import load_json
from bootstrap_node import fingerprint
import argparse
import fcntl
import shutil
import signal
import subprocess
import dsv41
ROOT = pathlib.Path(__file__).resolve().parents[1]


def configure(path, image, preserve_existing_public_bind=False):
    if path.exists():
        raise FileExistsError('Configuration already exists; refusing to overwrite: ' + str(path))
    c = load_json(ROOT / 'configs/cluster.example.json')
    c['image'] = image
    print('Use existing host-key-verified SSH aliases, including user@host if needed.\n'
          'No packages, drivers, credentials or firewall rules will be installed.\n'
          'Enter absolute, nonoverlapping model and writable cache paths on EACH node.')
    c['deployment'] = input('Deployment name [dsv41-vision]: ').strip() or 'dsv41-vision'
    for rank, node in enumerate(c['nodes']):
        print('Rank ' + str(rank) + (' (API head)' if rank == 0 else ' (worker)'))
        for key, label in [('ssh_host', 'SSH host'), ('fabric_ip', 'Fabric IPv4'),
                           ('socket_interface', 'Fabric network interface'), ('rdma_hca', 'RDMA HCA'),
                           ('model_path', 'Model directory'), ('cache_path', 'Dedicated writable cache directory')]:
            node[key] = input(label + ': ').strip()
    if preserve_existing_public_bind:
        c.update(preserve_existing_public_bind=True, api_bind='0.0.0.0', api_port=8000)
    cluster.validate(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        os.chmod(path, 0o600)
        f.write(json.dumps(c, indent=2) + '\n')
    print('Saved ' + str(path))
    return c


def remote_node(node, action, data):
    # Only reviewed installer sources are transferred; no pip, shell download or host install.
    names = ['scripts/bootstrap_node.py', 'scripts/artifacts.py', 'scripts/check_model.py',
             'tools/retained-model-contract.json']
    payload = {'files': {name: (ROOT/name).read_text() for name in names},
               'action': action, 'data': data}
    code = """import json,pathlib,subprocess,sys,tempfile
p=json.load(sys.stdin)
with tempfile.TemporaryDirectory(prefix='dsv41-install-') as td:
    root=pathlib.Path(td)
    for name,text in p['files'].items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    result=subprocess.run([sys.executable,'-S','-B',str(root/'scripts/bootstrap_node.py'),p['action']],input=json.dumps(p['data']),text=True)
    sys.exit(result.returncode)
"""
    result = cluster.ssh(node, ['python3', '-S', '-B', '-c', code],
                         input=json.dumps(payload), timeout=7*24*3600, check=False)
    if result.stdout: print(result.stdout, end='', flush=True)
    if result.returncode:
        # Node installer catches exceptions and emits concise, credential-free errors.
        if result.stderr: print(result.stderr, file=sys.stderr, end='')
        raise RuntimeError('Node ' + action + ' failed on ' + node['ssh_host'])


def health(c):
    code = "import urllib.request;urllib.request.urlopen('http://127.0.0.1:" + str(c['api_port']) + "/health',timeout=5).read()"
    if cluster.ssh(c['nodes'][0], ['python3', '-S', '-c', code], timeout=15, check=False).returncode:
        raise RuntimeError('Owned service is not healthy; inspect ./status.sh and logs. No automatic replacement.')


def existing(c, state_path):
    state = load_json(state_path)
    if state.get('configuration_sha256') != fingerprint(c):
        raise ValueError('Configuration changed since launch; refusing to replace deployment')
    items = state.get('containers', [])
    if len(items) != 2 or {i['rank'] for i in items} != {0, 1}:
        raise ValueError('Partial deployment state: inspect owned containers; no automatic replacement')
    running = []
    for item in items:
        d = cluster.owned(c['nodes'][item['rank']], item, state['owner'])
        if d['State']['OOMKilled']:
            raise ValueError('Owned deployment had an OOM; inspect it before explicit recovery')
        running.append(d['State']['Running'])
    if not any(running):
        cluster.restart_owned(c, state_path)
        return
    if not all(running):
        raise ValueError('Partially running deployment; inspect it before explicit recovery')
    health(c)
    print('Existing owned deployment is healthy; unchanged. No downloads or model scans.')


def start(c, release):
    cluster.validate(c)
    directory = ROOT/'.state'
    directory.mkdir(exist_ok=True)
    state_path = directory/(c['deployment']+'.json')
    with (directory/(c['deployment']+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            existing(c, state_path)
            return
        if release is None:
            raise ValueError('Published release required for first installation')
        if c.get('local_prebuilt') or c['image'] != release['image']:
            raise ValueError('First installation requires the exact published release image; use dsv41.py for advanced local-prebuilt deployment')
        for rank, node in enumerate(c['nodes']):
            print('Checking prerequisites on ' + node['ssh_host'], flush=True)
            remote_node(node, 'preflight', {'node': node, 'rank': rank, 'config': c, 'release': release})
        for rank, node in enumerate(c['nodes']):
            print('Installing pinned artifacts on ' + node['ssh_host'] + ' (resumable; this may take hours)', flush=True)
            remote_node(node, 'install', {'node': node, 'rank': rank, 'config': c, 'release': release})
        # Reuses the complete transaction, including locks and occupied-GPU/name guards.
        cluster.start(c, state_path)


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument('--config', type=pathlib.Path, default=ROOT/'configs/cluster.local.json', help='existing configuration for unattended installation')
    p.add_argument('--release', type=pathlib.Path, default=ROOT/'configs/release.json')
    p.add_argument('--configure-only', action='store_true', help='prompt and save configuration without contacting either node')
    p.add_argument('--preserve-existing-public-bind', action='store_true', help='explicitly preserve an already approved 0.0.0.0:8000 endpoint (configuration only)')
    a = p.parse_args(argv)
    if sys.version_info < (3, 11): raise ValueError('Python 3.11+ required; install it yourself before retrying')
    if a.config.exists():
        if a.configure_only or a.preserve_existing_public_bind:
            raise ValueError('Configuration exists; refusing to overwrite it or change API exposure')
        c = cluster.validate(load_json(a.config))
    else:
        if not sys.stdin.isatty():
            raise ValueError('No configuration. Run ./start.sh interactively once, or pass --config EXISTING_FILE; nothing installed.')
        # A real release is required even for configuration: never save invented image pins.
        r = dsv41.release(a.release)
        c = configure(a.config, r['image'], a.preserve_existing_public_bind)
        if a.configure_only: return
    if not shutil.which('ssh'):
        raise ValueError('OpenSSH client missing; install it yourself. No automated host changes.')
    state_path = ROOT/'.state'/(c['deployment']+'.json')
    r = None if state_path.exists() else dsv41.release(a.release)
    signal.signal(signal.SIGTERM, cluster.on_termination)
    start(c, r)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError, EOFError) as error:
        print('start: ' + str(error), file=sys.stderr)
        sys.exit(1)
