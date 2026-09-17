#!/usr/bin/env python3
"""Stdlib-only node installer, sent over verified SSH; no system installation."""
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import re
import platform
from artifacts import confined, download, load_json
RESERVE = 10 * 1024**3


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def paths(model, cache):
    model, cache = pathlib.Path(model).absolute(), pathlib.Path(cache).absolute()
    for p in (model, cache):
        if p != p.resolve() or p == pathlib.Path('/'):
            raise ValueError('Model/cache paths must be canonical absolute paths, without symlinks')
    if model == cache or model in cache.parents or cache in model.parents:
        raise ValueError('Model and cache paths overlap')
    return model, cache


def storage(path):
    path = pathlib.Path(path)
    while not path.exists():
        path = path.parent
    return path.stat().st_dev, shutil.disk_usage(path).free


def available(path):
    return storage(path)[1]


def require_budgets(budgets):
    grouped = {}
    for path, needed in budgets:
        device, free = storage(path)
        previous = grouped.get(device, (0, free, path))
        grouped[device] = (previous[0] + needed, min(previous[1], free), path)
    for needed, free, path in grouped.values():
        if free < needed + RESERVE:
            raise ValueError(f'Insufficient disk at {path}: need {needed + RESERVE} free bytes, have {free}')


def missing_bytes(destination, rows):
    needed = 0
    for row in rows:
        if not confined(destination, row['path']).exists():
            partial = confined(destination, row['path'] + '.part')
            size = partial.stat().st_size if partial.exists() else 0
            if size > row['size']: raise ValueError('Partial artifact exceeds pinned size')
            needed += row['size'] - size
    return needed


def require_space(path, needed):
    free = available(path)
    if free < needed + RESERVE:
        raise ValueError(f'Insufficient disk at {path}: need {needed + RESERVE} free bytes, have {free}')


def save_marker(path, record):
    confined(path.parent, path.name)
    fd, tmp = tempfile.mkstemp(prefix='.verified-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(record, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def validate_model(destination):
    subprocess.run([sys.executable, '-S', '-B', str(pathlib.Path(__file__).with_name('check_model.py')),
                    '--model', str(destination)], check=True)


@contextlib.contextmanager
def installation_lock(cache):
    cache.mkdir(parents=True, exist_ok=True)
    fd = os.open(confined(cache, '.recipe-install.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def install_model(model, destination, cache):
    destination, cache = paths(destination, cache)
    record = {'model': fingerprint(model), 'destination': str(destination), 'version': 1}
    with installation_lock(cache):
        marker = confined(cache, '.recipe-model-verified.json')
        if marker.exists() and load_json(marker) == record:
            print('Pinned model already installed (verification receipt; no payload scan).')
            return
        require_space(destination, missing_bytes(destination, model['files']))
        destination.mkdir(parents=True, exist_ok=True)
        for row in model['files']:
            print('Obtaining and verifying ' + row['path'], flush=True)
            url = 'https://huggingface.co/' + model['repository'] + '/resolve/' + model['revision'] + '/' + row['path']
            download(url, confined(destination, row['path']), row)
        validate_model(destination)
        save_marker(marker, record)


def image_content(image):
    return {key: image.get(key, '' if key == 'Variant' else None)
            for key in ['Config', 'RootFS', 'Os', 'Architecture', 'Variant']}


def validate_archive(release):
    a = release['image_archive']
    if not isinstance(a, dict): raise ValueError('image_archive must be an object')
    patterns = {'repository': r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*',
                'revision': '[a-f0-9]{40}', 'sha256': '[a-f0-9]{64}',
                'path': r'[A-Za-z0-9][A-Za-z0-9._-]*\.tar(?:\.gz)?'}
    for key, pattern in patterns.items():
        if not isinstance(a.get(key), str) or not re.fullmatch(pattern, a[key]):
            raise ValueError('Invalid pinned archive ' + key)
    for key in ['size', 'unpacked_size']:
        if type(a.get(key)) is not int or a[key] <= 0: raise ValueError('Archive ' + key + ' must be positive bytes')
    if not isinstance(release.get('image'), str) or not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}', release['image']):
        raise ValueError('Archive requires its explicit loaded image tag')
    m = a.get('image_metadata')
    if not isinstance(m, dict) or not isinstance(m.get('Config'), dict) or m.get('Architecture') != 'arm64' or m.get('Os') != 'linux':
        raise ValueError('Complete ARM64/Linux archive image metadata required')
    if m['Config'].get('Labels', {}).get('org.deepseek-vision-recipe.sources') != release.get('source_manifest_sha256'):
        raise ValueError('Archive source label mismatch')
    rootfs = m.get('RootFS', {})
    layers = rootfs.get('Layers')
    if rootfs.get('Type') != 'layers' or not isinstance(layers, list) or not layers or any(not isinstance(x, str) or not re.fullmatch('sha256:[a-f0-9]{64}', x) for x in layers):
        raise ValueError('Archive ordered RootFS layer digests required')
    return a


def command(args, **kwargs):
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)
    if result.returncode: raise RuntimeError('Prerequisite/operation failed: ' + ' '.join(args[:3]) + '; no host software was installed')
    return result.stdout


def inspect_image(release):
    result = subprocess.run(['docker', 'image', 'inspect', release['image']], capture_output=True, text=True, timeout=60)
    if result.returncode: return False
    image = json.loads(result.stdout)[0]
    if 'image_archive' in release:
        if image_content(image) != image_content(release['image_archive']['image_metadata']):
            raise ValueError('Existing image content differs from pinned archive; refusing to replace tag')
    elif (image.get('Architecture') != 'arm64' or image.get('Os') != 'linux'
          or release['image'] not in image.get('RepoDigests', [])
          or image.get('Config', {}).get('Labels', {}).get('org.deepseek-vision-recipe.sources') != release['source_manifest_sha256']):
        raise ValueError('Registry image readback mismatch')
    return True


def install_image(release, cache, docker_root):
    cache = pathlib.Path(cache)
    with installation_lock(cache):
        if inspect_image(release):
            print('Pinned runtime image already installed (metadata readback; no download).')
            return
        if 'image_archive' in release:
            a = validate_archive(release)
            target = confined(cache, a['path'])
            # Account for separate or shared filesystems and resumable archive bytes.
            require_budgets([(cache, missing_bytes(cache, [a])), (pathlib.Path(docker_root), 2*a['unpacked_size'])])
            url = 'https://huggingface.co/' + a['repository'] + '/resolve/' + a['revision'] + '/' + a['path']
            download(url, target, a)
            command(['docker', 'load', '--input', str(target)], timeout=7200)
        else:
            require_space(pathlib.Path(docker_root), RESERVE)
            command(['docker', 'pull', '--platform=linux/arm64', release['image']], timeout=7200)
        if not inspect_image(release): raise ValueError('Image missing after install')
        print('Runtime image installed and pinned content verified.')


def gpu_support(info, spec_paths=None):
    """Prove Docker can hand a container the GPU, by either supported mechanism.

    - a registered ``nvidia`` runtime: the classic nvidia-container-toolkit entry in
      ``/etc/docker/daemon.json``, visible in ``docker info``.
    - a CDI spec written by the toolkit: Docker 25+ injects devices from
      ``nvidia.com/gpu`` CDI devices for ``--gpus`` with no daemon.json runtime entry
      at all. The ASUS Ascent GX10 GB10 nodes are configured exactly this way, so
      requiring the runtime entry rejects machines that run GPU containers fine.

    Either way the caller still requires a working ``nvidia-smi`` on the node, so a
    machine with neither mechanism - or with a driver that cannot report - fails closed.
    """
    if 'nvidia' in (info.get('Runtimes') or {}):
        return 'registered nvidia runtime'
    for spec in (spec_paths if spec_paths is not None else ['/var/run/cdi/nvidia.yaml', '/etc/cdi/nvidia.yaml']):
        path = pathlib.Path(spec)
        if path.is_file() and 'nvidia.com/gpu' in path.read_text(errors='replace'):
            return 'nvidia CDI spec ' + spec
    raise ValueError('Docker GPU support missing: no registered nvidia runtime and no nvidia.com/gpu CDI spec; '
                     'configure NVIDIA Container Toolkit yourself')


def preflight(data):
    node, c, release = data['node'], data['config'], data['release']
    if sys.version_info < (3, 11): raise ValueError('Python 3.11+ required on each node')
    if platform.machine() not in ['aarch64', 'arm64'] or sys.platform != 'linux':
        raise ValueError('Linux ARM64 GB10 nodes required')
    for binary in ['docker', 'nvidia-smi', 'ip']:
        if not shutil.which(binary): raise ValueError(binary + ' missing; install prerequisites manually, then retry')
    info = json.loads(command(['docker', 'info', '--format', '{{json .}}'], timeout=30))
    gpu_support(info)
    if command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], timeout=30).strip():
        raise ValueError('Active GPU workloads; refusing installation/start; nothing is stopped automatically')
    name = c['deployment'] + '-rank' + str(data['rank'])
    if command(['docker', 'ps', '-a', '--filter', 'name=^/' + name + '$', '--format', '{{.ID}}'], timeout=30).strip():
        raise ValueError('Container name already exists; no matching local ownership state, refusing replacement')
    for path in ['/sys/class/net/' + node['socket_interface'], '/sys/class/infiniband/' + node['rdma_hca'], '/dev/infiniband']:
        if not pathlib.Path(path).is_dir(): raise ValueError('Missing configured network/RDMA device: ' + path)
    interfaces = json.loads(command(['ip', '-j', '-4', 'addr', 'show', 'dev', node['socket_interface']], timeout=30))
    if not any(addr.get('local') == node['fabric_ip'] for interface in interfaces for addr in interface.get('addr_info', [])):
        raise ValueError('Configured fabric IPv4 is not assigned to the selected interface')
    model, cache = paths(node['model_path'], node['cache_path'])
    needed = missing_bytes(model, release['model']['files'])
    archive = release.get('image_archive')
    docker_root = info.get('DockerRootDir')
    if not isinstance(docker_root, str) or not docker_root.startswith('/'):
        raise ValueError('Docker storage root unavailable')
    archive_budget = missing_bytes(cache, [archive]) if archive else 0
    image_budget = 2*archive['unpacked_size'] if archive else 2*RESERVE
    require_budgets([(model, needed), (cache, archive_budget), (pathlib.Path(docker_root), image_budget)])
    print('Prerequisites and conservative disk budget passed; no host changes.')
    return docker_root


def main():
    data = json.load(sys.stdin)
    if sys.argv[1] == 'preflight':
        preflight(data)
    elif sys.argv[1] == 'install':
        docker_root = preflight(data)
        node = data['node']
        install_image(data['release'], pathlib.Path(node['cache_path']), docker_root)
        install_model(data['release']['model'], pathlib.Path(node['model_path']), pathlib.Path(node['cache_path']))
    elif sys.argv[1] == 'image':
        node = data['node']
        _, cache = paths(node['model_path'], node['cache_path'])
        info = json.loads(command(['docker', 'info', '--format', '{{json .}}'], timeout=30))
        install_image(data['release'], cache, info['DockerRootDir'])
    else:
        raise ValueError('Unknown node action')


if __name__ == '__main__':
    try: main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print('node installer: ' + str(error), file=sys.stderr)
        sys.exit(1)
