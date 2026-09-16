#!/usr/bin/env python3
"""Explicit local ARM64 container build; no host packages, no automatic start."""
import argparse, fcntl, hashlib, json, pathlib, re, shutil, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument('--tag', default='dsv41-vision:retained-v1')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--acknowledge-build-gaps', action='store_true')
    a = p.parse_args()
    if not re.fullmatch('[a-z0-9][a-z0-9._/-]*:[a-zA-Z0-9_.-]+', a.tag):
        raise ValueError('invalid local image tag')
    manifest = ROOT/'runtime/source-manifest.json'
    rows = json.loads(manifest.read_text())
    for row in rows:
        file = ROOT/row['path']
        if hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('source checksum mismatch: '+row['path'])
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    command = ['docker','build','--platform=linux/arm64','--progress=plain',
               '--label','org.deepseek-vision-recipe.sources='+digest,
               '--tag',a.tag,'--file',str(ROOT/'runtime/Dockerfile'),str(ROOT)]
    if a.dry_run:
        print(json.dumps({'argv':command,'source_manifest_sha256':digest,
                          'warning':'fresh build and apt artifact pinning not release-verified'},indent=2));return
    if not (a.execute and a.acknowledge_build_gaps):
        p.error('requires --execute --acknowledge-build-gaps; consult pins.json')
    if not shutil.which('docker'):
        raise ValueError('Docker is required; install prerequisites manually')
    state = ROOT/'.state';state.mkdir(exist_ok=True)
    with (state/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        existing = subprocess.run(['docker','image','inspect',a.tag],capture_output=True)
        if existing.returncode == 0:
            raise ValueError('tag already exists; choose a new tag, never overwrite silently')
        subprocess.run(command,check=True)
        image = json.loads(subprocess.check_output(['docker','image','inspect',a.tag]))[0]
        if image['Config']['Labels'].get('org.deepseek-vision-recipe.sources') != digest:
            raise ValueError('built image label readback mismatch')
        subprocess.run(['docker','run','--rm','--network=none','-e','NVIDIA_VISIBLE_DEVICES=void',
                        '--entrypoint','python3',image['Id'],'-S','/opt/recipe/runtime_check.py','--root','/'],check=True)
        (state/'build.json').write_text(json.dumps({'image':image['Id'],'tag':a.tag,
                    'source_manifest_sha256':digest,'gpu_verified':False},indent=2)+'\n')
        print('Built and CPU-verified',image['Id'],'; GPU serving remains unverified')
if __name__ == '__main__':main()
