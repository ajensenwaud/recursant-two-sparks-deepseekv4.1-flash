#!/usr/bin/env python3
"""CPU integrity check. Import/discovery is NOT real-device kernel validation."""
import argparse, hashlib, importlib.metadata, json, pathlib, sys, struct, subprocess, re

def compare_parity(records):
    if len(records)!=2:raise ValueError('two runtime parity records required')
    for r in records:
        p=r.get('parity',{})
        if p.get('schema')!=1 or not p.get('versions') or not p.get('python') or not p.get('native'):
            raise ValueError('missing runtime code parity evidence')
        if not any('vllm_exl3_c' in n for n in p['native']) or not any('exllamav3_ext' in n for n in p['native']):
            raise ValueError('native extension parity evidence missing')
        for name,code in p['native'].items():
            if any(not code.get(k) for k in ['text','elf_header','program_headers','allocated']) or not re.fullmatch('[a-f0-9]{64}',code.get('layout_sha256','')):
                raise ValueError('missing ELF ABI/load-layout evidence')
            if any(ext in name for ext in ['vllm_exl3_c','exllamav3_ext']):
                if any(not re.fullmatch('[a-f0-9]{64}',code.get(k,'')) for k in ['text','sass','ptx']):
                    raise ValueError('missing native CUDA code parity evidence')
    if records[0]['parity']!=records[1]['parity']:
        raise ValueError('cross-rank Python/native runtime code differs')


def elf_code(path):
    """ELF64 LE host sections plus device SASS/PTX, not fatbin bookkeeping."""
    import platform
    data=path.read_bytes()
    def bounds(offset,size):
        if offset<0 or size<0 or offset+size>len(data):raise ValueError('ELF range outside file')
    bounds(0,64)
    if data[:7]!=b'\x7fELF\x02\x01\x01':raise ValueError('ELF64 little endian version 1 required')
    header=struct.unpack_from('<HHIQQQIHHHHHH',data,16)
    kind,machine,version,entry,phoff,offset,flags,ehsize,phsize,phcount,size,count,strings=header
    expected={'x86_64':62,'aarch64':183}.get(platform.machine())
    if expected is None or machine!=expected or kind not in (2,3) or version!=1 or ehsize!=64:
        raise ValueError('unsupported native ELF ABI/type')
    if size!=64 or not count or not 0<strings<count or phsize!=56 or not phcount or phcount==65535:
        raise ValueError('unsupported ELF tables')
    bounds(offset,count*size);bounds(phoff,phcount*phsize)
    programs=[struct.unpack_from('<IIQQQQQQ',data,phoff+i*phsize) for i in range(phcount)]
    for typ,perm,off,addr,phys,filesz,memsz,align in programs:
        bounds(off,filesz)
        if align and align&(align-1):raise ValueError('invalid segment alignment')
        if typ==1 and (filesz>memsz or (align>1 and off%align!=addr%align)):
            raise ValueError('invalid load segment')
    if not any(r[0]==1 for r in programs):raise ValueError('ELF load segment missing')
    rows=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
    for r in rows:
        if r[1]!=8:bounds(r[4],r[5])
        if r[8] and r[8]&(r[8]-1):raise ValueError('invalid section alignment')
        if r[2]&2 and not any(p[0]==1 and p[3]<=r[3] and r[3]+r[5]<=p[3]+p[6] for p in programs):
            raise ValueError('allocated section outside load segments')
    if rows[strings][1]!=3:raise ValueError('section names must be a string table')
    names=data[rows[strings][4]:rows[strings][4]+rows[strings][5]]
    sections={}
    for r in rows:
        if r[0]>=len(names) or b'\0' not in names[r[0]:]:raise ValueError('invalid section name')
        name=names[r[0]:].split(b'\0',1)[0].decode()
        if name in sections:raise ValueError('ambiguous section names')
        sections[name]=r
    def digest(r):return hashlib.sha256(data[r[4]:r[4]+r[5]]).hexdigest()
    if '.text' not in sections or sections['.text'][2]&6!=6:raise ValueError('ELF text section missing')
    result={'text':digest(sections['.text']),
            'elf_header':list(data[:9])+list(header[:5])+[flags,ehsize],
            'program_headers':programs}
    cuda=any(n in sections for n in ['.nv_fatbin','.nvFatBinSegment'])
    # Geometry is never discarded, even for build IDs, CUDA bookkeeping or BSS.
    excluded=['.note.gnu.build-id','.nv_fatbin','.nvFatBinSegment']
    result['allocated']={n:{'metadata':list(r[1:]),'sha256':None if r[1]==8 or n in excluded else digest(r)}
                         for n,r in sections.items() if r[2]&2}
    # Conservatively retain all other bytes, including load padding and tables.
    # Only same-layout debug/name bytes, a validated GNU build-ID descriptor and
    # disassembled CUDA payloads may differ. Layout changes fail closed.
    canonical=bytearray(data)
    for n,r in sections.items():
        off,length=r[4],r[5]
        if r[1]==8:continue
        if n=='.note.gnu.build-id':
            if r[1]!=7 or length<16:raise ValueError('invalid build ID note')
            namesz,descsz,typ=struct.unpack_from('<III',data,off)
            if namesz!=4 or typ!=3 or data[off+12:off+16]!=b'GNU\0' or 16+((descsz+3)//4)*4!=length:
                raise ValueError('unsupported build ID note')
            canonical[off+16:off+16+descsz]=b'\0'*descsz
        elif (n.startswith('.debug') and not r[2]&2) or n=='.shstrtab' or (cuda and n in ['.nv_fatbin','.nvFatBinSegment']):
            if n.startswith('.debug') or n=='.shstrtab':
                if any(p[0]==1 and off<p[2]+p[5] and p[2]<off+length for p in programs):
                    raise ValueError('debug/name bytes overlap a load segment; parity unproven')
            canonical[off:off+length]=b'\0'*length
    result['layout_sha256']=hashlib.sha256(canonical).hexdigest()
    if cuda:
        for mode in ['sass','ptx']:
            run=subprocess.run(['cuobjdump','--dump-'+mode,str(path)],capture_output=True,timeout=120)
            if run.returncode:raise ValueError('CUDA code extraction failed')
            output=run.stdout.replace(str(path).encode(),b'ELF')
            if mode=='sass' and b'Function :' not in output:raise ValueError('missing CUDA SASS evidence')
            result[mode]=hashlib.sha256(output).hexdigest()
    return result


def runtime_parity(site,versions):
    python={};native={}
    for package in ['vllm','vllm_exl3','exllamav3','torch','triton','flashinfer']:
        root=site/package
        if not root.is_dir():raise ValueError('runtime package tree missing')
        for path in sorted(root.rglob('*')):
            if not path.is_file():continue
            name=str(path.relative_to(site))
            if path.suffix in ['.py','.cu','.cuh','.h','.cpp','.c','.ptx','.cubin']:
                python[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            elif '.so' in path.name:native[name]=elf_code(path)
    for pattern in ['vllm_exl3_c*.so','exllamav3_ext*.so']:
        paths=list(site.glob(pattern))
        if len(paths)!=1:raise ValueError('native extension missing or ambiguous')
        native[paths[0].name]=elf_code(paths[0])
    return {'schema':1,'versions':versions,'python':python,'native':native}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument('--root', type=pathlib.Path, default=pathlib.Path('/'))
    p.add_argument('--manifest', type=pathlib.Path, default=pathlib.Path('/opt/recipe/source-manifest.json'))
    p.add_argument('--sources-only', action='store_true')
    a = p.parse_args()
    rows = json.loads(a.manifest.read_text()); count = 0
    for r in rows:
        if r['path'].startswith('runtime/rootfs/'):
            path = a.root/r['path'].removeprefix('runtime/rootfs/')
            if hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256']:
                raise ValueError('retained source mismatch: '+str(path.relative_to(a.root)))
            count += 1
    if not count:
        raise ValueError('empty overlay manifest')
    result = {'source_files':count,'source_match':True,'gpu_verified':False}
    if not a.sources_only:
        if sys.version_info[:2] != (3,12):
            raise ValueError('Python 3.12 required')
        site = pathlib.Path('/usr/local/lib/python3.12/dist-packages')
        sys.path.extend([str(site),'/usr/lib/python3/dist-packages'])
        pins = json.loads(pathlib.Path('/opt/recipe/pins.json').read_text())
        expected = {'vllm':pins['runtime_version'],'torch':pins['torch'],
                    'triton':pins['triton'],'flashinfer-python':pins['flashinfer-python'],
                    'exllamav3':pins['exllama']['version'],'vllm-exl3':pins['plugin']['version']}
        actual = {k:importlib.metadata.version(k) for k in expected}
        if actual != expected:
            raise ValueError('runtime package version mismatch')
        extensions = list(site.glob('vllm_exl3_c*.so'))
        if len(extensions) != 1 or not list(site.glob('exllamav3_ext*.so')):
            raise ValueError('required native extensions missing or ambiguous')
        result['versions'] = actual
        # Installation-only import check of our native additions; no vendor disassembly.
        import torch
        for name in ['exllamav3_ext','vllm_exl3_c']:
            importlib.import_module(name)
        result['native_imports'] = ['exllamav3_ext','vllm_exl3_c']
        result['native_elf_sha256'] = hashlib.sha256(extensions[0].read_bytes()).hexdigest()
        if pathlib.Path('/usr/lib/python3.12/sitecustomize.py').resolve() != pathlib.Path('/etc/python3.12/sitecustomize.py'):
            raise ValueError('sitecustomize symlink target mismatch')
    print(json.dumps(result,indent=2))
if __name__ == '__main__':main()
