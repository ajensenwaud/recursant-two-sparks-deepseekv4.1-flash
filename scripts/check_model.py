#!/usr/bin/env python3
"""Full index/header/codebook census; optional expensive conversion-receipt hashes."""
import argparse,json,pathlib,re,sys
from artifacts import confined,header,load_json,validate_index,verify_file


def contract():
    base=pathlib.Path(__file__).resolve().parent
    path=base/'tools/retained-model-contract.json'
    if not path.is_file():path=base.parent/'tools/retained-model-contract.json'
    return load_json(path)


def check_config(config):
    expected=contract()['config']
    # The retained architecture and delegated quantization form one contract.
    # Permit only informational producer-version changes, not precision knobs.
    for key,value in expected.items():
        if key=='transformers_version':continue
        if config.get(key)!=value:raise ValueError('retained model configuration mismatch: '+key)


def check_mcg(raw):
    if raw != bytes.fromhex('ed1faccb'):
        raise ValueError('unsupported MCG multiplier payload')


def check_headers(tensors):
    expected=contract()
    if len(tensors)!=expected['tensor_count']:raise ValueError('total tensor coverage mismatch')
    non_routed={};stems=set();found=set()
    suffixes=['trellis','mcg','suh','svh']
    for name,t in tensors.items():
        m=re.fullmatch(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(.+)',name)
        if not m or int(m[1])>=40:
            non_routed[name]={k:t[k] for k in ['dtype','shape']};continue
        if int(m[2])>=384 or m[4] not in suffixes:raise ValueError('unexpected routed expert tensor')
        stem=name.rsplit('.',1)[0];stems.add(stem);found.add(name)
        swap=m[3]=='w2'
        dtype,shape={'trellis':('I16',[144,320,32] if swap else [320,144,32]),
                     'mcg':('I32',[]),'suh':('F16',[2304] if swap else [5120]),
                     'svh':('F16',[5120] if swap else [2304])}[m[4]]
        if (t['dtype'],t['shape'])!=(dtype,shape):raise ValueError('2-bit tensor geometry mismatch')
    expected_stems={f'layers.{l}.ffn.experts.{e}.w{w}' for l in range(40) for e in range(384) for w in [1,2,3]}
    if stems!=expected_stems or found!={s+'.'+x for s in stems for x in suffixes}:
        raise ValueError('expert stem/suffix coverage mismatch')
    if non_routed!=expected['non_routed']:raise ValueError('non-routed/draft/vision tensor contract mismatch')
    return len(stems)


def check_head(tensor):
    if tensor.get('dtype')!='BF16' or tensor.get('shape')!=[129280,5120]:
        raise ValueError('retained BF16 output-head geometry required')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    p.add_argument('--model',type=pathlib.Path,required=True)
    p.add_argument('--checksums',action='store_true')
    a=p.parse_args();root=a.model.resolve();check_config(load_json(confined(root,'config.json')))
    result=validate_index(root)
    if result['shards']!=48:raise ValueError('expected 48 complete shards')
    index=load_json(root/'model.safetensors.index.json')['weight_map']
    if 'head.weight' not in index:raise ValueError('output head missing')
    check_head(header(confined(root,index['head.weight']))[0]['head.weight'])
    tensors={}
    for shard in sorted(set(index.values())):
        path=confined(root,shard);rows,offset=header(path)
        tensors.update(rows)
        # Only four payload bytes per expert, never full weight materialization.
        with path.open('rb') as f:
            for name,t in rows.items():
                if name.endswith('.mcg'):
                    f.seek(offset+t['data_offsets'][0]);check_mcg(f.read(4))
    matrices=check_headers(tensors)
    if a.checksums:
        record=load_json(confined(root,'conversion.json'))
        if record.get('source_revision')!='dba1be0a40aa45a94ad051997016db3960a90277':
            raise ValueError('conversion source revision mismatch')
        rows=record['files']
        if len(rows)!=48 or {r['path'] for r in rows}!=set(index.values()):raise ValueError('receipt shard coverage mismatch')
        for row in rows:verify_file(confined(root,row['path']),row)
    print(json.dumps(result|{'converted_matrices':matrices,'validation':'headers-and-receipt-sha256' if a.checksums else 'headers-and-mcg-markers',
          'quality_certified':False,'exact_retained_bytes_certified':False},indent=2))
if __name__=='__main__':main()
