#!/usr/bin/env python3
"""Download only the pinned official source assets. Does not build or serve."""
import argparse, fcntl, json, re, shutil
from pathlib import Path
from urllib.parse import quote
from artifacts import confined, download, load_json, validate_index, verify_file
ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument('--destination', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    manifest = load_json(ROOT/'configs/source-model.json')
    revision = manifest['revision']
    if not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('immutable model revision required')
    rows = manifest['files']
    names = [r['path'] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError('duplicate artifacts')
    for name in names:
        confined(Path.cwd(), name)
    summary = {'repository':manifest['repository'], 'revision':revision,
               'files':len(rows), 'shards':sum(n.endswith('.safetensors') for n in names),
               'bytes':sum(r['size'] for r in rows)}
    if a.dry_run:
        print(json.dumps(summary, indent=2)); return
    if a.destination is None:
        p.error('--destination is required outside --dry-run')
    root = a.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with confined(root,'.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        remaining = sum(max(0,r['size']-((root/(r['path']+'.part')).stat().st_size
                        if (root/(r['path']+'.part')).exists() else 0))
                        for r in rows if not confined(root,r['path']).exists())
        if not a.verify_only and shutil.disk_usage(root).free < remaining + 20*1024**3:
            raise ValueError('insufficient disk: remaining download plus 20 GiB reserve required')
        for r in rows:
            target = confined(root, r['path'])
            if a.verify_only:
                verify_file(target,r)
            else:
                url = f"https://huggingface.co/{manifest['repository']}/resolve/{revision}/{quote(r['path'])}"
                download(url,target,r)
            print('verified',r['path'],flush=True)
        summary.update(validate_index(root))
        print(json.dumps(summary,indent=2))
if __name__ == '__main__':
    main()
