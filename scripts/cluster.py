#!/usr/bin/env python3
"""Coordinated two-node launcher. Own containers only; no replacement or host setup."""
import argparse, contextlib, fcntl, hashlib, ipaddress, json, os, pathlib, re
import select, shlex, subprocess, time, uuid, tempfile
from artifacts import load_json
ROOT=pathlib.Path(__file__).resolve().parents[1]
LABEL='org.deepseek-vision-recipe.owner'


def validate(c):
    if set(c)-{'local_prebuilt','preserve_existing_public_bind'}!={'deployment','image','api_bind','api_port','master_port','startup_timeout_seconds','nodes'}:
        raise ValueError('unknown or missing cluster fields')
    if 'local_prebuilt' in c and type(c['local_prebuilt']) is not bool:
        raise ValueError('local_prebuilt must be a boolean')
    if not re.fullmatch('[a-z][a-z0-9-]{2,40}',c['deployment']):raise ValueError('invalid deployment name')
    if not re.fullmatch(r'(?:sha256:[a-f0-9]{64}|[a-z0-9][a-z0-9._/-]*(?::[A-Za-z0-9_.-]+|@sha256:[a-f0-9]{64}))',c['image']):raise ValueError('invalid image tag or digest')
    if 'preserve_existing_public_bind' in c:
        if c['preserve_existing_public_bind'] is not True or c['api_bind']!='0.0.0.0' or c['api_port']!=8000:
            raise ValueError('existing public bind opt-in requires true and exactly 0.0.0.0:8000')
    elif c['api_bind']!='127.0.0.1':raise ValueError('loopback required; expose through an authenticated TLS proxy explicitly')
    for key in ['api_port','master_port']:
        if type(c[key]) is not int or not 1024<=c[key]<=65535:raise ValueError('invalid port')
    if c['api_port']==c['master_port']:raise ValueError('ports collide')
    if type(c['startup_timeout_seconds']) is not int or not 60<=c['startup_timeout_seconds']<=1800:
        raise ValueError('startup timeout must be 60..1800 seconds')
    if len(c['nodes'])!=2:raise ValueError('exactly two ranks required')
    for n in c['nodes']:
        if set(n)!={'ssh_host','fabric_ip','socket_interface','rdma_hca','model_path','cache_path'}:
            raise ValueError('unknown or missing node fields')
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.@-]*',n['ssh_host']):raise ValueError('invalid SSH host')
        address=ipaddress.ip_address(n['fabric_ip'])
        if address.version!=4 or address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError('invalid fabric IPv4')
        for k in ['socket_interface','rdma_hca']:
            if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.:-]*',n[k]):raise ValueError('invalid network device')
        for k in ['model_path','cache_path']:
            p=pathlib.PurePosixPath(n[k])
            if not p.is_absolute() or '..' in p.parts or not re.fullmatch('/[A-Za-z0-9_./ -]+',n[k]) or str(p)=='/' or str(p)!=n[k] or n[k].startswith('//'):
                raise ValueError('invalid mount path')
        model,cache=(pathlib.PurePosixPath(n[k]) for k in ['model_path','cache_path'])
        if model == cache or model in cache.parents or cache in model.parents:
            raise ValueError('cache and model paths must be distinct and nonoverlapping')
    if len({n['ssh_host'] for n in c['nodes']})!=2 or len({n['fabric_ip'] for n in c['nodes']})!=2:
        raise ValueError('ranks need distinct hosts and fabric addresses')
    return c


def serve_argv(c,rank):
    args=['serve','/model','--tensor-parallel-size','2','--nnodes','2','--node-rank',str(rank),
          '--distributed-executor-backend','mp','--master-addr',c['nodes'][0]['fabric_ip'],
          '--master-port',str(c['master_port'])]
    args+=['--headless'] if rank else ['--host',c['api_bind'],'--port',str(c['api_port'])]
    args+=['--max-model-len','1048576','--kv-cache-dtype','fp8','--kv-cache-memory','4294967296',
           '--gpu-memory-utilization','0.75','--max-num-seqs','8','--max-num-batched-tokens','2048',
           '--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}',
           '--kernel-config','{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}',
           '--block-size','64','--quantization','exl3','--tokenizer-mode','deepseek_v41',
           '--tool-call-parser','deepseek_v41','--enable-auto-tool-choice','--reasoning-parser','deepseek_v41',
           '--default-chat-template-kwargs','{"thinking":true,"reasoning_effort":"high"}',
           '--served-model-name','deepseek-ai/DeepSeek-V4.1-Flash','--limit-mm-per-prompt','{"image":1}',
           '--speculative-config','{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}']
    return args


def create_argv(c,rank,image,owner):
    n=c['nodes'][rank]
    env={'VLLM_HOST_IP':n['fabric_ip'],'TP_SOCKET_IFNAME':n['socket_interface'],
         'NCCL_SOCKET_IFNAME':n['socket_interface'],'GLOO_SOCKET_IFNAME':n['socket_interface'],
         'NCCL_IB_HCA':n['rdma_hca'],'NCCL_IB_DISABLE':'0','NCCL_NET':'IB','NCCL_CROSS_NIC':'1',
         'NCCL_CUMEM_ENABLE':'0','NCCL_NVLS_ENABLE':'0','NCCL_DEBUG':'WARN',
         'VLLM_PLUGINS':'vllm_exl3','DSV41_ENGRAM_DISK':'1','DSV41_MXFP8_MODE':'off',
         'VLLM_EXL3_MOE_KERNEL':'native','TORCH_CUDA_ARCH_LIST':'12.1a','FLASHINFER_CUDA_ARCH_LIST':'12.1a',
         'FLASHINFER_DISABLE_VERSION_CHECK':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
         'VLLM_ENGINE_READY_TIMEOUT_S':str(c['startup_timeout_seconds']),'HF_HOME':'/cache/huggingface',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    args=['docker','create','--name',c['deployment']+'-rank'+str(rank),'--label',LABEL+'='+owner,
          '--restart=no','--network=host','--ipc=host','--gpus=all','--shm-size=32g',
          '--device=/dev/infiniband','--cap-add=IPC_LOCK','--ulimit=memlock=-1:-1',
          '--security-opt=label=disable','--mount','type=bind,src='+n['model_path']+',dst=/model,readonly',
          '--mount','type=bind,src='+n['cache_path']+',dst=/cache/huggingface']
    for k,v in sorted(env.items()):args+=['--env',k+'='+v]
    return args+[image]+serve_argv(c,rank)


def ssh(n,args,timeout=60,check=True,input=None):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',n['ssh_host'],shlex.join(args)],
                     input=input,capture_output=True,text=True,timeout=timeout)
    if check and r.returncode:
        # No remote stdout/stderr (possibly credentials) in exception text.
        raise RuntimeError('remote command failed on '+n['ssh_host']+' (exit '+str(r.returncode)+')')
    return r


@contextlib.contextmanager
def remote_locks(c):
    holders=[]
    try:
        for n in c['nodes']:
            path=resolve_cache(n)['cache_path']+'/.recipe-control.lock'
            code="import os,fcntl,sys;f=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);s=os.fstat(f);assert s.st_uid==os.getuid();fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print('LOCKED',flush=True);sys.stdin.read()"
            proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',n['ssh_host'],
                  shlex.join(['python3','-u','-c',code,path])],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
            holders.append(proc)
            if not select.select([proc.stdout],[],[],20)[0] or proc.stdout.readline().strip()!='LOCKED':
                raise RuntimeError('remote control lock unavailable')
        def check():
            if len(holders)!=len(c['nodes']) or any(p.poll() is not None for p in holders):
                raise RuntimeError('remote control lock connection lost')
        check()
        yield check
        check()
    finally:
        for proc in holders:
            if proc.stdin:proc.stdin.close()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)


def inspect(n,cid):
    r=ssh(n,['docker','inspect',cid])
    return json.loads(r.stdout)[0]


def owned(n,item,owner):
    d=inspect(n,item['id'])
    if d['Id']!=item['id'] or d['Config']['Labels'].get(LABEL)!=owner:
        raise ValueError('container ownership changed; refusing mutation')
    return d


def stop_owned(c,state):
    errors=[]
    for item in state['containers']:
        n=c['nodes'][item['rank']]
        try:
            d=owned(n,item,state['owner'])
            if d['State']['Running']:ssh(n,['docker','stop','--time','30',item['id']],timeout=60)
            if owned(n,item,state['owner'])['State']['Running']:raise ValueError('stop readback failed')
        except Exception as e:errors.append(type(e).__name__)
    if errors:raise RuntimeError('owned container cleanup failed; manual recovery required: '+','.join(errors))


def resolve_cache(n):
    # Shutdown needs only the dedicated cache lock, not a surviving model mount.
    code = """import json,pathlib,sys
c=pathlib.Path(sys.argv[1]).resolve(strict=True)
m=pathlib.Path(sys.argv[2]).resolve(strict=False)
if not c.is_dir() or c==pathlib.Path('/') or c==m or c in m.parents or m in c.parents:
    raise ValueError('unsafe resolved cache directory')
print(json.dumps({'cache_path':str(c)}))
"""
    paths=json.loads(ssh(n,['python3','-S','-c',code,n['cache_path'],n['model_path']]).stdout)
    if not re.fullmatch('/[A-Za-z0-9_./ -]+',paths['cache_path']):raise ValueError('unsafe resolved cache path')
    return paths


def resolve_mounts(n):
    # Run on the host owning the filesystem, before any bind or lock-file write.
    code = """import json,pathlib,sys
m,c=[pathlib.Path(p).resolve(strict=True) for p in sys.argv[1:]]
if not m.is_dir() or not c.is_dir() or m==pathlib.Path('/') or c==pathlib.Path('/') or m==c or m in c.parents or c in m.parents:
    raise ValueError('unsafe resolved mount directories')
print(json.dumps({'model_path':str(m),'cache_path':str(c)}))
"""
    paths=json.loads(ssh(n,['python3','-S','-c',code,n['model_path'],n['cache_path']]).stdout)
    check=dict(n,**paths)
    # Also reject unsafe characters introduced through symlink targets.
    for value in paths.values():
        if not re.fullmatch('/[A-Za-z0-9_./ -]+',value):raise ValueError('unsafe resolved mount path')
    return {k:check[k] for k in ['model_path','cache_path']}


def host_probe(n):
    code="import json,pathlib,sys;v=dict(x.split(':',1) for x in pathlib.Path('/proc/meminfo').read_text().splitlines());o=dict(x.split() for x in pathlib.Path('/proc/vmstat').read_text().splitlines());assert all(pathlib.Path(p).is_dir() for p in sys.argv[1:]);print(json.dumps({'available':int(v['MemAvailable'].split()[0])*1024,'oom':int(o['oom_kill'])}))"
    return json.loads(ssh(n,['python3','-S','-c',code,n['model_path'],n['cache_path'],
                      '/sys/class/net/'+n['socket_interface'],'/sys/class/infiniband/'+n['rdma_hca']]).stdout)


def start(c,state_path,lock_check=None):
    if state_path.exists():raise ValueError('deployment state exists; stop and inspect it rather than replace')
    source_digest=hashlib.sha256((ROOT/'runtime/source-manifest.json').read_bytes()).hexdigest()
    state={'owner':str(uuid.uuid4()),'containers':[],'configuration_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()}
    def save():
        fd,name=tempfile.mkstemp(prefix='.'+state_path.name+'.',dir=state_path.parent)
        try:
            with os.fdopen(fd,'w') as f:
                f.write(json.dumps(state,indent=2)+'\n');f.flush();os.fsync(f.fileno())
            os.replace(name,state_path)
            directory=os.open(state_path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:
            if os.path.exists(name):os.unlink(name)
    recovered=False
    def recover(failure):
        nonlocal recovered
        recovered=True
        state['ready']=False
        # Recovery must not depend on writable local storage or a live lock channel.
        try:stop_owned(c,state)
        except BaseException as cleanup:
            failure.add_note('ownership-checked cleanup failed: '+type(cleanup).__name__)
        else:failure.add_note('ownership-checked cleanup completed')
        try:save()
        except BaseException as persistence:
            failure.add_note('recovery state persistence failed: '+type(persistence).__name__)
    try:
        with (remote_locks(c) if lock_check is None else contextlib.nullcontext(lock_check)) as check:
            try:
                check()
                c=dict(c,nodes=[dict(n,**resolve_mounts(n)) for n in c['nodes']])
                images=[];image_records=[];initial=[]
                for rank,n in enumerate(c['nodes']):
                    check()
                    probe=host_probe(n);initial.append(probe)
                    if probe['available']<6*1024**3:raise ValueError('less than 6 GiB available')
                    if ssh(n,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).stdout.strip():
                        raise ValueError('active GPU workloads; refusing to start')
                    # Existing names are never stopped, removed, or renamed.
                    if ssh(n,['docker','ps','-a','--filter','name=^/'+c['deployment']+'-rank'+str(rank)+'$','--format','{{.ID}}']).stdout.strip():
                        raise ValueError('container name already exists')
                    image=json.loads(ssh(n,['docker','image','inspect',c['image']]).stdout)[0]
                    if image['Config'].get('Labels',{}).get('org.deepseek-vision-recipe.sources')!=source_digest:
                        raise ValueError('image was not built from this source manifest')
                    images.append(image['Id'])
                    image_records.append(image)
                if images[0]!=images[1]:
                    # Docker backends can expose manifest IDs versus config IDs.
                    # Compare only metadata from the existing image inspections.
                    contents=[]
                    for image in image_records:
                        rootfs=image.get('RootFS') or {}
                        layers=rootfs.get('Layers')
                        if (not isinstance(image.get('Config'),dict) or not image['Config']
                            or rootfs.get('Type')!='layers' or not isinstance(layers,list) or not layers
                            or any(not isinstance(layer,str) or not re.fullmatch('sha256:[a-f0-9]{64}',layer) for layer in layers)
                            or not image.get('Os') or not image.get('Architecture')):
                            raise ValueError('different image IDs require complete image content metadata')
                        contents.append((image['Config'],layers,image['Os'],image['Architecture'],image.get('Variant','')))
                    if contents[0]!=contents[1]:raise ValueError('both ranks require identical image config, layers and platform')
                state['runtime_parity']={'identical_image_ids':images[0]==images[1],'identical_image_content':True}
                # Persist ownership before create can succeed remotely but lose its reply.
                # A detached controller needs this nonce to reconcile an untracked name.
                save()
                for rank in [1,0]:
                    check()
                    n=c['nodes'][rank]
                    cid=ssh(n,create_argv(c,rank,images[rank],state['owner'])).stdout.strip()
                    if not re.fullmatch('[a-f0-9]{64}',cid):raise ValueError('invalid create response')
                    item={'rank':rank,'id':cid,'image':images[rank]};state['containers'].append(item);save()
                    d=owned(n,item,state['owner'])
                    mounts={m['Destination']:(m['Source'],m['RW'],m['Type']) for m in d.get('Mounts',[])}
                    if mounts.get('/model')!=(n['model_path'],False,'bind') or mounts.get('/cache/huggingface')!=(n['cache_path'],True,'bind'):
                        raise ValueError('created mount readback mismatch')
                    if d['State']['Running'] or d['Image']!=images[rank] or d['Config']['Cmd']!=serve_argv(c,rank):
                        raise ValueError('created container readback mismatch')
                for item in state['containers']:
                    check()
                    ssh(c['nodes'][item['rank']],['docker','start',item['id']])
                wait_ready(c,state,initial,check)
                state['ready']=True;save();check()
            except BaseException as failure:
                recover(failure)
                raise
        print(json.dumps({'ready':True,'containers':state['containers'],'verification':'run explicit verify'},indent=2))
    except BaseException as failure:
        if not recovered:recover(failure)
        raise


def wait_ready(c,state,initial,check):
    end=time.monotonic()+c['startup_timeout_seconds']
    while time.monotonic()<end:
        check()
        for item in state['containers']:
            n=c['nodes'][item['rank']];d=owned(n,item,state['owner']);probe=host_probe(n)
            if not d['State']['Running'] or d['State']['OOMKilled'] or probe['oom']!=initial[item['rank']]['oom'] or probe['available']<6*1024**3:
                raise RuntimeError('startup exit, OOM, or 6 GiB reserve guard tripped')
        url='http://127.0.0.1:'+str(c['api_port'])
        code="import urllib.request;urllib.request.urlopen("+repr(url+'/health')+",timeout=5).read()"
        if ssh(c['nodes'][0],['python3','-S','-c',code],timeout=15,check=False).returncode==0:return
        time.sleep(3)
    raise TimeoutError('bounded startup deadline')


def restart_owned(c,state_path):
    # Same containers, same recorded ownership and launch argv; never recreate by name.
    from bootstrap_node import save_marker
    state=load_json(state_path)
    if state.get('configuration_sha256')!=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest():
        raise ValueError('configuration changed since launch')
    if len(state.get('containers',[]))!=2 or {i['rank'] for i in state['containers']}!={0,1}:
        raise ValueError('partial deployment; explicit recovery required')
    started=[];recovered=False
    try:
        with remote_locks(c) as check:
            initial={}
            for item in state['containers']:
                check();n=c['nodes'][item['rank']]
                d=owned(n,item,state['owner'])
                resolved=resolve_mounts(n)
                mounts={m['Destination']:(m['Source'],m['RW'],m['Type']) for m in d.get('Mounts',[])}
                if (d['State']['Running'] or d['State']['OOMKilled'] or d['Image']!=item['image']
                    or d['Config']['Cmd']!=serve_argv(c,item['rank'])
                    or mounts.get('/model')!=(resolved['model_path'],False,'bind')
                    or mounts.get('/cache/huggingface')!=(resolved['cache_path'],True,'bind')):
                    raise ValueError('restart requires unchanged, stopped, non-OOM owned containers')
                initial[item['rank']]=host_probe(n)
                if initial[item['rank']]['available']<6*1024**3:raise ValueError('less than 6 GiB available')
                if ssh(n,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).stdout.strip():
                    raise ValueError('active GPU workloads; refusing restart')
            try:
                state['ready']=False;save_marker(state_path,state)
                for item in sorted(state['containers'],key=lambda i:-i['rank']):
                    check();n=c['nodes'][item['rank']]
                    if owned(n,item,state['owner'])['State']['Running']:
                        raise ValueError('container changed during restart')
                    started.append(item)
                    ssh(n,['docker','start',item['id']])
                wait_ready(c,state,initial,check)
                state['ready']=True;save_marker(state_path,state);check()
            except BaseException as failure:
                recovered=True
                try:stop_owned(c,dict(state,containers=started))
                except BaseException as cleanup:failure.add_note('restart cleanup failed: '+type(cleanup).__name__)
                state['ready']=False
                try:save_marker(state_path,state)
                except BaseException as persistence:failure.add_note('restart state persistence failed: '+type(persistence).__name__)
                raise
    except BaseException as failure:
        if started and not recovered:
            try:stop_owned(c,dict(state,containers=started))
            except BaseException as cleanup:failure.add_note('restart cleanup failed: '+type(cleanup).__name__)
            state['ready']=False
            try:save_marker(state_path,state)
            except BaseException as persistence:failure.add_note('restart state persistence failed: '+type(persistence).__name__)
        raise
    print('Owned stopped containers restarted and healthy; no downloads or payload scans.')


def on_termination(signum,frame):
    raise InterruptedError('termination requested; stop owned containers before exit')


def main():
    import signal
    signal.signal(signal.SIGTERM,on_termination)
    p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    p.add_argument('action',choices=['configure','plan','start','stop','status'])
    p.add_argument('--config',type=pathlib.Path,default=ROOT/'configs/cluster.local.json')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--acknowledge-unreleased',action='store_true')
    a=p.parse_args()
    if a.action=='configure':
        with a.config.open('x') as f:f.write((ROOT/'configs/cluster.example.json').read_text())
        print('Created non-secret configuration; edit every node/network/path value before launch.');return
    c=validate(load_json(a.config))
    if a.action=='plan':
        print(json.dumps({'commands':[{'ssh_host':n['ssh_host'],'argv':create_argv(c,r,c['image'],'OWNER_ASSIGNED_AT_START')} for r,n in enumerate(c['nodes'])],
                          'order':[1,0],'startup_timeout_seconds':c['startup_timeout_seconds']},indent=2));return
    if a.action in ['start','stop'] and not a.execute:p.error('state changes require --execute')
    if a.action=='start' and not a.acknowledge_unreleased:p.error('consult release blockers; requires --acknowledge-unreleased')
    directory=ROOT/'.state';directory.mkdir(exist_ok=True)
    state_path=directory/(c['deployment']+'.json')
    with (directory/(c['deployment']+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=='status':
            state=load_json(state_path)
            print(json.dumps([{'rank':i['rank'],'running':owned(c['nodes'][i['rank']],i,state['owner'])['State']['Running']} for i in state['containers']],indent=2));return
        if a.action=='start':start(c,state_path)
        else:
            with remote_locks(c):
                state=load_json(state_path)
                if state['configuration_sha256']!=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest():
                    raise ValueError('configuration changed since launch')
                stop_owned(c,state);print('Owned containers verified stopped; retained for inspection. No removal performed.')
if __name__=='__main__':main()
