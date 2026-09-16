"""Pinned artifact retrieval and verification, Python standard library only."""
from pathlib import Path, PurePosixPath

def confined(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or '\\' in name or any(ord(c) < 32 for c in name):
        raise ValueError('unsafe artifact path')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name or name == '.':
        raise ValueError('unsafe artifact path')
    result = root.joinpath(*path.parts)
    for p in [result, *result.parents]:
        if p == root:
            break
        if p.is_symlink():
            raise ValueError('symlink in artifact path')
    if not result.resolve().is_relative_to(root):
        raise ValueError('artifact escapes root')
    return result


def verify_file(path, metadata):
    import hashlib
    size = metadata['size']
    if type(size) is not int or size < 0 or path.stat().st_size != size:
        raise ValueError('artifact size mismatch: ' + path.name)
    sha = metadata.get('sha256')
    expected = sha or metadata.get('git_blob_sha1')
    import re
    if not expected or not re.fullmatch('[a-f0-9]{64}' if sha else '[a-f0-9]{40}', expected):
        raise ValueError('missing valid pinned digest')
    digest = hashlib.sha256() if sha else hashlib.sha1()
    if not sha:
        digest.update(f'blob {size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('artifact digest mismatch: ' + path.name)


def download(url, path, metadata, *, allow_test_http=False):
    import os, urllib.parse, urllib.request
    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password or not (parsed.scheme == 'https' or
            (allow_test_http and parsed.scheme == 'http' and parsed.hostname == '127.0.0.1')):
        raise ValueError('HTTPS public artifact URL required')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    confined(path.parent, path.name)
    if path.exists():
        verify_file(path, metadata)
        return
    part = confined(path.parent, path.name + '.part')
    size = metadata['size']
    if type(size) is not int or size < 0:
        raise ValueError('invalid artifact size')
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError('partial artifact exceeds pinned size')
    if offset < size:
        headers = {'User-Agent': 'deepseek-vision-recipe/0.1', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
            final = urllib.parse.urlsplit(r.url)
            if final.scheme != 'https' and not (allow_test_http and final.hostname == '127.0.0.1'):
                raise ValueError('unsafe artifact redirect')
            if offset and r.status == 200:
                offset = 0  # Server ignored Range: restart, never append a whole file.
            elif offset and (r.status != 206 or r.headers.get('Content-Range') != f'bytes {offset}-{size-1}/{size}'):
                raise ValueError('invalid resume response')
            elif not offset and r.status != 200:
                raise ValueError('unexpected download status')
            with part.open('ab' if offset else 'wb') as f:
                count = offset
                while chunk := r.read(min(8 * 1024 * 1024, size - count + 1)):
                    count += len(chunk)
                    if count > size:
                        raise ValueError('download exceeds pinned size')
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
    elif not part.exists():
        part.touch(exist_ok=False)
    verify_file(part, metadata)
    if path.exists():
        raise ValueError('destination appeared during download')
    part.rename(path)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def load_json(path):
    import json
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def header(path):
    import json, math, struct
    sizes = {'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E5M2':1,'F8_E8M0':1,
             'F8_E4M3FN':1,'F8_E8M0FNU':1,'I16':2,'U16':2,'F16':2,'BF16':2,
             'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}
    with Path(path).open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('short safetensors prefix')
        length, = struct.unpack('<Q', prefix)
        if not 2 <= length <= 64 * 1024 * 1024:
            raise ValueError('unsafe safetensors header length')
        raw = f.read(length)
    if len(raw) != length:
        raise ValueError('short safetensors header')
    tensors = json.loads(raw, object_pairs_hook=unique_object)
    tensors.pop('__metadata__', None)
    end = Path(path).stat().st_size - 8 - length
    intervals = []
    for name, t in tensors.items():
        a, b = t['data_offsets']
        if type(a) is not int or type(b) is not int or not 0 <= a <= b <= end:
            raise ValueError('invalid tensor offsets')
        if not all(type(x) is int and x >= 0 for x in t['shape']):
            raise ValueError('invalid tensor shape')
        if t['dtype'] not in sizes or math.prod(t['shape']) * sizes[t['dtype']] != b-a:
            raise ValueError('tensor byte count mismatch')
        intervals.append((a,b))
    cursor = 0
    for a,b in sorted(intervals):
        if a != cursor:
            raise ValueError('tensor gap or overlap')
        cursor = b
    if cursor != end:
        raise ValueError('unindexed tensor bytes')
    return tensors, 8 + length


def validate_index(root):
    root = Path(root)
    index = load_json(confined(root, 'model.safetensors.index.json'))['weight_map']
    names = set(index.values())
    if not names or any(not n.endswith('.safetensors') for n in names):
        raise ValueError('invalid shard index')
    if set(p.name for p in root.glob('*.safetensors')) != names:
        raise ValueError('missing or extra finalized shards')
    if any(root.glob('*.part')) or any(root.glob('*.tmp')) or any(root.glob('*.incomplete')):
        raise ValueError('unfinished artifacts present')
    observed = {}
    for name in sorted(names):
        tensors, _ = header(confined(root, name))
        for tensor in tensors:
            if tensor in observed:
                raise ValueError('duplicate tensor across shards')
            observed[tensor] = name
    if observed != index:
        raise ValueError('index does not match every tensor header')
    return {'shards':len(names), 'tensors':len(observed)}
