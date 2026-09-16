#!/usr/bin/env python3
"""Prebuilt-first entry point. Unpublished artifacts fail closed; never build implicitly."""
import argparse, hashlib, json, pathlib, re, subprocess, sys
import cluster
from artifacts import load_json, confined, download
ROOT=pathlib.Path(__file__).resolve().parents[1]

def release(path, *, model_only=False):
    r=load_json(path)
    if not isinstance(r,dict):raise ValueError('release must be an object')
    if model_only:
        if r.get('status')!='published' and (not isinstance(r.get('model'),dict) or r['model'].get('status')!='published'):
            raise ValueError('Exact model artifact is not published')
    elif r.get('status')!='published':
        raise ValueError('Release not published: image digest and exact model revision are unavailable. Use advanced local build instructions only; no substitute is selected.')
    if not model_only and not re.fullmatch(r'[a-z0-9][a-z0-9.-]+/[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}',r.get('image') or ''):
        raise ValueError('registry image must be immutable sha256 reference')
    digest=hashlib.sha256((ROOT/'runtime/source-manifest.json').read_bytes()).hexdigest()
    if r.get('source_manifest_sha256')!=digest:
        raise ValueError('release source fingerprint does not match this checkout')
    m=r.get('model')
    if not isinstance(m,dict) or not isinstance(m.get('repository'),str) or not isinstance(m.get('revision'),str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*',m['repository']) or not re.fullmatch('[a-f0-9]{40}',m['revision']):
        raise ValueError('exact model repository and immutable revision required')
    files=m.get('files')
    if not isinstance(files,list) or not files:raise ValueError('model file manifest required')
    seen=set()
    for row in files:
        if not isinstance(row,dict):raise ValueError('model file entry must be an object')
        name=row.get('path','')
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name) or name in seen:
            raise ValueError('unsafe or duplicate model asset')
        seen.add(name)
        if type(row.get('size')) is not int or row['size']<0 or not isinstance(row.get('sha256'),str) or not re.fullmatch('[a-f0-9]{64}',row['sha256']):
            raise ValueError('model size and SHA256 required')
    if not seen:raise ValueError('model file manifest required')
    return r

def pull(c,r):
    for n in c['nodes']:
        cluster.ssh(n,['docker','pull','--platform=linux/arm64',r['image']],timeout=3600)
        image=json.loads(cluster.ssh(n,['docker','image','inspect',r['image']]).stdout)[0]
        if image.get('Architecture')!='arm64' or image.get('Os')!='linux' or r['image'] not in image.get('RepoDigests',[]) or image.get('Config',{}).get('Labels',{}).get('org.deepseek-vision-recipe.sources')!=r['source_manifest_sha256']:
            raise ValueError('pulled image readback mismatch')
    print('Digest-pinned ARM64 image verified on both nodes; no service started.')

def download_model(m,destination):
    import fcntl,os
    destination=destination.absolute()
    if destination.is_symlink() or destination.resolve()==pathlib.Path('/'):
        raise ValueError('dedicated non-root model directory required')
    destination.mkdir(parents=True,exist_ok=True)
    fd=os.open(confined(destination,'.download.lock'),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in m['files']:
            url='https://huggingface.co/'+m['repository']+'/resolve/'+m['revision']+'/'+row['path']
            download(url,confined(destination,row['path']),row)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_model.py'),'--model',str(destination)],check=True)
    print('Exact pinned model files verified locally; repeat on the other node.')

def observe(c,action):
    state=load_json(ROOT/'.state'/(c['deployment']+'.json'))
    if state['configuration_sha256']!=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest():
        raise ValueError('configuration changed since launch')
    items=state['containers'] if action=='logs' else [next(i for i in state['containers'] if i['rank']==0)]
    for item in items:
        n=c['nodes'][item['rank']]
        cluster.owned(n,item,state['owner'])
        cmd=['docker','logs','--tail','100',item['id']] if action=='logs' else ['docker','exec',item['id'],'python3','-S','/opt/recipe/verify.py','--base-url','http://127.0.0.1:'+str(c['api_port'])]
        result=cluster.ssh(n,cmd,timeout=600)
        print(result.stdout)
        if result.stderr:print(result.stderr,file=sys.stderr)

def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    p.add_argument('action',choices=['configure','pull','download-model','plan','start','stop','status','logs','verify'])
    p.add_argument('--config',type=pathlib.Path,default=ROOT/'configs/cluster.local.json')
    p.add_argument('--release',type=pathlib.Path,default=ROOT/'configs/release.json')
    p.add_argument('--destination',type=pathlib.Path)
    p.add_argument('--local-image',help='configure with an installed local image tag or full sha256 image ID; no registry or model download')
    p.add_argument('--preserve-existing-public-bind',action='store_true',help='explicitly preserve an already approved 0.0.0.0:8000 endpoint; no firewall changes')
    a=p.parse_args(argv)
    if a.preserve_existing_public_bind and a.action!='configure':p.error('--preserve-existing-public-bind requires configure')
    if a.local_image is not None and (a.action!='configure' or not re.fullmatch(r'(?:sha256:[a-f0-9]{64}|[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})',a.local_image)):
        p.error('--local-image requires configure and a local image tag or full sha256 image ID')
    if a.action=='configure':
        c=load_json(ROOT/'configs/cluster.example.json')
        raw={} if a.local_image else load_json(a.release)
        c['image']=a.local_image or (release(a.release)['image'] if raw.get('status')=='published' else 'UNPUBLISHED')
        if a.local_image:c['local_prebuilt']=True
        if a.preserve_existing_public_bind:c.update(preserve_existing_public_bind=True,api_bind='0.0.0.0',api_port=8000)
        with a.config.open('x') as f:f.write(json.dumps(c,indent=2)+'\n')
        print('Configuration created. Edit node settings; UNPUBLISHED cannot start.');return
    if a.action=='download-model':
        if a.destination is None:p.error('download-model requires --destination on this node')
        download_model(release(a.release,model_only=True)['model'],a.destination);return
    local=False
    if a.action in ['pull','plan','start']:
        config=load_json(a.config)
        local=config.get('local_prebuilt') is True or ('local_prebuilt' not in config and re.fullmatch('sha256:[a-f0-9]{64}',config.get('image','')))
    if a.action in ['pull','plan','start']:
        if local:
            if a.action=='pull':raise ValueError('Local prebuilt images cannot be pulled; transfer docker save/load explicitly to both nodes.')
            r={'image':load_json(a.config)['image']}
        else:r=release(a.release)
    if a.action in ['stop','status']:
        cmd=[sys.executable,'-B',str(ROOT/'scripts/cluster.py'),a.action,'--config',str(a.config)]
        if a.action=='stop':cmd+=['--execute']
        subprocess.run(cmd,check=True);return
    c=cluster.validate(load_json(a.config))
    if a.action in ['pull','plan','start']:
        if c['image']!=r['image']:raise ValueError('configuration image must equal the published immutable release image')
        if a.action=='pull':pull(c,r);return
        cmd=[sys.executable,'-B',str(ROOT/'scripts/cluster.py'),a.action,'--config',str(a.config)]
        if a.action=='start':cmd+=['--execute','--acknowledge-unreleased']
        subprocess.run(cmd,check=True);return
    observe(c,a.action)

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.SubprocessError) as e:
        print(str(e),file=sys.stderr);sys.exit(1)
