#!/usr/bin/env python3
"""Convert a pinned official source into a NEW uncalibrated 2-bit MCG pack.
Not a byte-reproduction or quality-certification claim for the retained pack.
Run only inside the built image with exclusive GPU access and sufficient disk.
"""
import argparse, fcntl, hashlib, json, os, pathlib, shutil, subprocess, sys
from artifacts import confined, header, load_json, validate_index, verify_file
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE if (HERE/'tools').is_dir() else HERE.parent
sys.path.insert(0,str(ROOT/'tools'))
from pack_meta import is_routed_expert_tensor, is_routed_expert_weight


def required_free_bytes(source_size):
    # Preserved Engram shards can exceed 100 GB. Never assume the small
    # routed-expert shard footprint or successful cross-mount hardlinking.
    return max(source_size,12*1024**3)+40*1024**3


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def compare_shard(source, output):
    source_h,source_offset=header(source); output_h,output_offset=header(output)
    stems={n[:-7] for n in source_h if is_routed_expert_weight(n)}
    keep={n for n in source_h if not is_routed_expert_tensor(n)}
    expected=keep | {s+'.'+suffix for s in stems for suffix in ['trellis','mcg','suh','svh']}
    if set(output_h)!=expected:
        raise ValueError('converted tensor coverage mismatch')
    with source.open('rb') as sf, output.open('rb') as of:
        for name in keep:
            a,b=source_h[name]['data_offsets'];c,d=output_h[name]['data_offsets']
            if source_h[name]['dtype']!=output_h[name]['dtype'] or source_h[name]['shape']!=output_h[name]['shape'] or b-a!=d-c:
                raise ValueError('preserved tensor metadata mismatch')
            sf.seek(source_offset+a);of.seek(output_offset+c)
            remaining=b-a
            while remaining:
                n=min(8*1024*1024,remaining)
                if sf.read(n)!=of.read(n):raise ValueError('preserved tensor bytes mismatch')
                remaining-=n
    for stem in stems:
        swap=stem.endswith('.w2')
        expected_shapes={'trellis':('I16',[144,320,32] if swap else [320,144,32]),
                         'mcg':('I32',[]),'suh':('F16',[2304] if swap else [5120]),
                         'svh':('F16',[5120] if swap else [2304])}
        for suffix,(dtype,shape) in expected_shapes.items():
            t=output_h[stem+'.'+suffix]
            if (t['dtype'],t['shape'])!=(dtype,shape):
                raise ValueError('not the required V4.1 2-bit MCG tensor geometry')
    return len(stems)


def argv(source,destination,name):
    return [sys.executable,str(ROOT/'tools/quantize_experts_exl3.py'),'--src',str(source),
            '--dst',str(destination),'--bits','2','--batch','1','--greedy','--beam','16',
            '--device','cuda:0','--only-files',name]


def main():
    p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    p.add_argument('--source',required=True,type=pathlib.Path)
    p.add_argument('--destination',required=True,type=pathlib.Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--acknowledge-unvalidated-conversion',action='store_true')
    a=p.parse_args();source=a.source.resolve();dest=a.destination.resolve()
    if source==dest or source.is_relative_to(dest) or dest.is_relative_to(source):
        raise ValueError('source and output must be separate, non-nested directories')
    manifest=load_json(ROOT/'configs/source-model.json')
    shards=[r for r in manifest['files'] if r['path'].endswith('.safetensors')]
    plan={'shards':len(shards),'example_argv':argv(source,dest/'work','model-00003-of-00048.safetensors'),
          'byte_identical_to_retained_claim':False,'requires':'exclusive GPU, pinned runtime and verified source',
          'reserve_bytes':40*1024**3,'expected_output_bytes_approx':358107269776}
    if a.dry_run:print(json.dumps(plan,indent=2));return
    if not (a.execute and a.acknowledge_unvalidated_conversion):
        p.error('requires --execute --acknowledge-unvalidated-conversion')
    if sys.version_info[:2]!=(3,12):raise ValueError('run inside the pinned Python 3.12 image')
    subprocess.run([sys.executable,'-S',str(ROOT/'runtime_check.py'),'--root','/'],check=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if gpu.strip():raise ValueError('GPU has active compute processes; refusing conversion')
    dest.mkdir(parents=True,exist_ok=True)
    with confined(dest,'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for r in manifest['files']:verify_file(confined(source,r['path']),r)
        validate_index(source)
        tool_hash={x.name:digest(x) for x in (ROOT/'tools').glob('*.py')}
        total=0;index={};output_rows=[]
        for row in shards:
            name=row['path'];target=confined(dest,name);receipt=confined(dest,name+'.verified.json')
            if target.exists():
                if not receipt.exists():raise ValueError('unverified existing output; use a new destination')
                saved=load_json(receipt)
                if saved['source_sha256']!=row['sha256'] or saved['tools']!=tool_hash:
                    raise ValueError('resume provenance mismatch')
                verify_file(target,saved)
                count=compare_shard(confined(source,name),target)
            else:
                if shutil.disk_usage(dest).free < required_free_bytes(row['size']):
                    raise ValueError('need next-shard worst-case copy bytes plus 40 GiB reserve; source deletion is never performed')
                work=confined(dest,'work-'+name)
                if work.exists():raise ValueError('unfinished work directory; inspect it and use a new destination')
                work.mkdir()
                subprocess.run(argv(source,work,name),check=True)
                converted=confined(work,name)
                count=compare_shard(confined(source,name),converted)
                converted.rename(target)
                saved={'path':name,'size':target.stat().st_size,'sha256':digest(target),
                       'source_sha256':row['sha256'],'tools':tool_hash,'converted_matrices':count}
                receipt.write_text(json.dumps(saved,indent=2)+'\n')
                # Only remove this newly-created, strictly-owned per-shard work.
                shutil.rmtree(work)
            total+=count;output_rows.append(saved)
            for tensor in header(target)[0]:
                if tensor in index:raise ValueError('duplicate output tensor')
                index[tensor]=name
            print('verified shard',name,'converted matrices',count,flush=True)
        if total!=46080 or len(output_rows)!=48:raise ValueError('full-pack coverage mismatch')
        # Only inference sidecars from the pinned download allowlist.
        for row in manifest['files']:
            if row['path'] in {'config.json','tokenizer.json','tokenizer_config.json','LICENSE',
                               'processor_config.json','preprocessor_config.json','generation_config.json'}:
                target=confined(dest,row['path'])
                if row['path']=='config.json':
                    from pack_meta import apply_pack_config,build_quantization_config
                    content=json.dumps(apply_pack_config(load_json(source/'config.json'),build_quantization_config(bits=2)),indent=2).encode()+b'\n'
                else:content=confined(source,row['path']).read_bytes()
                if target.exists() and target.read_bytes()!=content:raise ValueError('sidecar collision')
                target.write_bytes(content)
        temporary=confined(dest,'model.safetensors.index.json.tmp')
        temporary.write_text(json.dumps({'metadata':{'total_size':sum(r['size'] for r in output_rows)},'weight_map':index},indent=2)+'\n')
        temporary.replace(confined(dest,'model.safetensors.index.json'))
        result=validate_index(dest)
        (dest/'conversion.json').write_text(json.dumps({'source_revision':manifest['revision'],'files':output_rows,
                 'converted_matrices':total,'quality_validated':False,'byte_identical_to_retained':False},indent=2)+'\n')
        print(json.dumps(result|{'quality_validated':False},indent=2))
if __name__=='__main__':main()
