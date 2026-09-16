"""Isolated publication regressions: CPU and mocked remote operations only."""
import json, pathlib, sys, tempfile, unittest, subprocess
from unittest.mock import patch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tools'))

class ELFParityTests(unittest.TestCase):
    def test_elf_abi_and_load_geometry_changes_fail_parity(self):
        import runtime_check,struct
        original=bytearray(pathlib.Path(sys.executable).resolve().read_bytes())
        shoff=struct.unpack_from('<Q',original,40)[0];phoff=struct.unpack_from('<Q',original,32)[0]
        count=struct.unpack_from('<H',original,60)[0]
        bss=next(shoff+i*64 for i in range(count) if struct.unpack_from('<I',original,shoff+i*64+4)[0]==8)
        mutations=[('machine',18,'H',183 if struct.unpack_from('<H',original,18)[0]==62 else 62),('type',16,'H',1),('load_flags',phoff+4,'I',7),('load_address',phoff+16,'Q',8192),('bss',bss+32,'Q',8192),('bss_alignment',bss+48,'Q',4096),('section_bounds',shoff+64+24,'Q',len(original)+1)]
        with tempfile.TemporaryDirectory() as td:
            a=pathlib.Path(td)/'a';b=pathlib.Path(td)/'b';a.write_bytes(original)
            baseline=runtime_check.elf_code(a)
            for name,offset,fmt,value in mutations:
                with self.subTest(mutation=name):
                    changed=bytearray(original);struct.pack_into('<'+fmt,changed,offset,value);b.write_bytes(changed)
                    try:record=runtime_check.elf_code(b)
                    except ValueError:continue
                    self.assertNotEqual(baseline,record)

    def test_real_elf_build_id_and_debug_payload_are_nonsemantic(self):
        import runtime_check,struct
        original=bytearray(pathlib.Path(sys.executable).resolve().read_bytes())
        offset=struct.unpack_from('<Q',original,40)[0];count,strings=struct.unpack_from('<HH',original,60)
        rows=[struct.unpack_from('<IIQQQQIIQQ',original,offset+i*64) for i in range(count)]
        names=original[rows[strings][4]:rows[strings][4]+rows[strings][5]]
        changed=bytearray(original);modified=[]
        for row in rows:
            name=bytes(names[row[0]:]).split(b'\0',1)[0]
            if name==b'.note.gnu.build-id':changed[row[4]+row[5]-1]^=1;modified.append(name)
            if name.startswith(b'.debug') and row[5]:changed[row[4]]^=1;modified.append(name)
        self.assertIn(b'.note.gnu.build-id',modified)
        with tempfile.TemporaryDirectory() as td:
            a=pathlib.Path(td)/'a';b=pathlib.Path(td)/'b';a.write_bytes(original);b.write_bytes(changed)
            self.assertEqual(runtime_check.elf_code(a),runtime_check.elf_code(b))

    def test_actual_local_elf_host_code_is_hashed(self):
        import runtime_check
        record=runtime_check.elf_code(pathlib.Path(sys.executable).resolve())
        self.assertEqual(len(record['text']),64)
        self.assertIn('.text',record['allocated'])

    def test_fatbin_metadata_difference_requires_same_actual_disassembly(self):
        import runtime_check,struct
        def fixture(path, fatbin, text=b'host-code', loaded_names=False):
            names=b'\0.shstrtab\0.text\0.nv_fatbin\0'
            import platform
            data=bytearray(120);data[:7]=b'\x7fELF\x02\x01\x01'
            struct.pack_into('<HHIQQQIHHHHHH',data,16,3,{'x86_64':62,'aarch64':183}[platform.machine()],1,0,64,0,0,64,56,1,64,0,0)
            sections=[(0,0,0,0,0,0,0,0,0,0)]
            for name,payload,flags in [(b'.shstrtab',names,0),(b'.text',text,6),(b'.nv_fatbin',fatbin,2)]:
                offset=len(data);data.extend(payload)
                sections.append((names.index(name),3 if name==b'.shstrtab' else 1,flags,offset,offset,len(payload),0,0,1,0))
            offset=len(data)
            for row in sections:data.extend(struct.pack('<IIQQQQIIQQ',*row))
            struct.pack_into('<Q',data,40,offset);struct.pack_into('<HHH',data,58,64,len(sections),1)
            start=0 if loaded_names else sections[2][4]
            end=sections[-1][4]+sections[-1][5]
            struct.pack_into('<IIQQQQQQ',data,64,1,5,start,start,start,end-start,end-start,1)
            path.write_bytes(data)
        with tempfile.TemporaryDirectory() as td:
            a=pathlib.Path(td)/'a.so';b=pathlib.Path(td)/'b.so'
            fixture(a,b'bookkeeping-a');fixture(b,b'bookkeeping-b')
            def disassemble(args,**kwargs):
                return subprocess.CompletedProcess(args,0,b'Function : kernel\nidentical device instructions\n',b'')
            with patch.object(runtime_check.subprocess,'run',side_effect=disassemble) as tool:
                self.assertEqual(runtime_check.elf_code(a),runtime_check.elf_code(b))
                self.assertEqual(tool.call_count,4)
                fixture(b,b'bookkeeping-b',loaded_names=True)
                with self.assertRaises(ValueError):runtime_check.elf_code(b)
                fixture(b,b'bookkeeping-b',text=b'changed-host-code')
                self.assertNotEqual(runtime_check.elf_code(a),runtime_check.elf_code(b))
            with patch.object(runtime_check.subprocess,'run',return_value=subprocess.CompletedProcess([],1,b'',b'')):
                with self.assertRaises(ValueError):runtime_check.elf_code(a)

    def test_package_inventory_uses_real_files_and_refuses_missing_packages(self):
        import runtime_check,shutil
        with tempfile.TemporaryDirectory() as td:
            site=pathlib.Path(td)
            with self.assertRaises(ValueError):runtime_check.runtime_parity(site,{'test':'fixture'})
            for package in ['vllm','vllm_exl3','exllamav3','torch','triton','flashinfer']:
                (site/package).mkdir();(site/package/'fixture.py').write_text('pass\n')
            for name in ['vllm_exl3_c.so','exllamav3_ext.so']:
                shutil.copyfile(pathlib.Path(sys.executable).resolve(),site/name)
            a=runtime_check.runtime_parity(site,{'test':'fixture'})
            self.assertEqual(len(a['python']),6);self.assertEqual(len(a['native']),2)
            (site/'vllm/fixture.py').write_text('changed = True\n')
            self.assertNotEqual(a,runtime_check.runtime_parity(site,{'test':'fixture'}))

class CompleteModelTests(unittest.TestCase):
    def test_retained_config_positive_and_precision_architecture_negatives(self):
        import check_model,copy
        config=json.loads((ROOT/'tools/retained-model-contract.json').read_text())['config']
        check_model.check_config(config)
        for path,value in [('architectures',[]),('vision_config',None),('text_config.num_hidden_layers',1),('text_config.num_nextn_predict_layers',1),('text_config.dspark_n_routed_experts',384),('quantization_config.non_routed_quantization.expert_dtype','fp8'),('quantization_config.non_routed_quantization.fmt','e5m2'),('quantization_config.non_routed_quantization.activation_scheme','static')]:
            bad=copy.deepcopy(config);target=bad;parts=path.split('.')
            for k in parts[:-1]:target=target[k]
            target[parts[-1]]=value
            with self.subTest(field=path),self.assertRaises(ValueError):check_model.check_config(bad)

    def test_complete_header_coverage_and_marker_payload(self):
        import check_model,copy,struct
        self.assertTrue(hasattr(check_model,'check_headers'),'complete header contract missing')
        contract=json.loads((ROOT/'tools/retained-model-contract.json').read_text())
        headers=copy.deepcopy(contract['non_routed'])
        for layer in range(40):
            for expert in range(384):
                for w in ['w1','w2','w3']:
                    for suffix,(dtype,shape) in {'trellis':('I16',[144,320,32] if w=='w2' else [320,144,32]),'mcg':('I32',[]),'suh':('F16',[2304] if w=='w2' else [5120]),'svh':('F16',[5120] if w=='w2' else [2304])}.items():
                        headers[f'layers.{layer}.ffn.experts.{expert}.{w}.{suffix}']={'dtype':dtype,'shape':shape}
        self.assertEqual(len(headers),188245)
        self.assertEqual(check_model.check_headers(headers),46080)
        for prefix in ['vision.','mtp.','aligner.','layers.0.attn.']:
            name=next(n for n in headers if n.startswith(prefix));value=headers.pop(name)
            with self.subTest(prefix=prefix),self.assertRaises(ValueError):check_model.check_headers(headers)
            headers[name]=value
        name=next(n for n in headers if n.startswith('vision.'));old=headers[name];headers[name]={'dtype':'I8','shape':old['shape']}
        with self.assertRaises(ValueError):check_model.check_headers(headers)
        headers[name]=old
        check_model.check_mcg(struct.pack('<I',0xCBAC1FED))
        for raw in [struct.pack('<I',0),b'',b'bad']:
            with self.assertRaises(ValueError):check_model.check_mcg(raw)

class TransactionTests(unittest.TestCase):
    def launch_fixture(self, fail_save=0, lock_check=None, parity=None, exit_lock=False, competitor=False, fail_start=False, enforce_locks=False, intervening_owner=False, image_metadata=None, fail_create=False, observed=None):
        import cluster, hashlib, contextlib, io, stat
        c=json.loads((ROOT/'configs/cluster.example.json').read_text())
        digest=hashlib.sha256((ROOT/'runtime/source-manifest.json').read_bytes()).hexdigest()
        containers={}; calls=[]; writes=[]; held=False; violations=[]
        synced=[];real_fsync=cluster.os.fsync;real_replace=cluster.os.replace
        def sync(fd):
            real_fsync(fd)
            synced.append('directory' if stat.S_ISDIR(cluster.os.fstat(fd).st_mode) else 'file')
        def remote(n,args,**kwargs):
            calls.append(args)
            if enforce_locks and not held:violations.append(args)
            output=''
            if args[0]=='nvidia-smi' and competitor and held:output='competing-pid'
            rank=c['nodes'].index(n)
            if args[:3]==['docker','image','inspect']:
                image={'Id':'sha256:'+str(rank if parity else 0)*64,'Config':{'Labels':{'org.deepseek-vision-recipe.sources':digest}}}
                if image_metadata is not None:
                    extra=dict(image_metadata[rank]);image['Config'].update(extra.pop('Config',{}));image.update(extra)
                output=json.dumps([image])
            elif args[:2]==['docker','create']:
                if observed is not None:
                    observed.setdefault('before_create',[]).append({
                        'state':json.loads(state_path.read_text()) if state_path.exists() else None,
                        'synced':list(synced)})
                cid=str(rank+1)*64;owner=args[args.index('--label')+1].split('=',1)[1]
                containers[cid]={'Id':cid,'Name':'/'+args[args.index('--name')+1],'Image':'sha256:'+str(rank if parity else 0)*64,'Config':{'Labels':{cluster.LABEL:owner},'Cmd':cluster.serve_argv(c,rank)},'State':{'Running':False,'OOMKilled':False},
                  'Mounts':[{'Destination':'/model','Source':n['model_path'],'RW':False,'Type':'bind'},{'Destination':'/cache/huggingface','Source':n['cache_path'],'RW':True,'Type':'bind'}]};output=cid
                if fail_create:raise subprocess.TimeoutExpired(args,60)
            elif args[:2]==['docker','start']:
                containers[args[2]]['State']['Running']=True
                if fail_start:raise RuntimeError('injected start failure')
            elif args[:2]==['docker','stop']:containers[args[-1]]['State']['Running']=False
            elif args[:2]==['docker','exec']:output='{"passed":true}'
            elif '/opt/recipe/runtime_check.py' in args:output=json.dumps(parity[rank] if parity else {})
            return subprocess.CompletedProcess(args,0,output,'')
        def inspected(n,cid):
            if enforce_locks and not held:violations.append(['ownership-readback',cid])
            return containers[cid]
        def host_probe(n):
            if enforce_locks and not held:violations.append(['host-preflight'])
            return {'available':10*1024**3,'oom':0}
        def failing_write(*args,**kwargs):
            writes.append(1)
            if len(writes)>=fail_save:raise OSError('injected state persistence failure')
            return real_replace(*args,**kwargs)
        @contextlib.contextmanager
        def locks():
            nonlocal held
            held=True
            try:
                yield lambda:None
                if exit_lock:
                    if intervening_owner:
                        next(iter(containers.values()))['Config']['Labels'][cluster.LABEL]='intervening-owner'
                    raise RuntimeError('lock lost on context exit')
            finally:held=False
        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            state_path=pathlib.Path(td)/'state.json'
            stack.enter_context(patch.object(cluster.os,'fsync',side_effect=sync))
            stack.enter_context(patch.object(cluster,'resolve_mounts',side_effect=lambda n:{k:n[k] for k in ['model_path','cache_path']}))
            stack.enter_context(patch.object(cluster,'ssh',side_effect=remote))
            stack.enter_context(patch.object(cluster,'remote_locks',side_effect=lambda c:locks()))
            stack.enter_context(patch.object(cluster,'host_probe',side_effect=host_probe))
            stack.enter_context(patch.object(cluster,'inspect',side_effect=inspected))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if fail_save:
                stack.enter_context(patch.object(cluster.os,'replace',side_effect=failing_write))
            error=None
            try:
                if lock_check is None:cluster.start(c,state_path)
                else:cluster.start(c,state_path,lock_check=lock_check)
            except BaseException as e:error=e
            if observed is not None:
                observed['state']=json.loads(state_path.read_text()) if state_path.exists() else None
        self.assertFalse(violations, 'remote operations outside healthy locks')
        return containers,calls,error

    def test_first_create_timeout_retains_durable_owner_for_reconciliation(self):
        import cluster,hashlib
        observed={}
        containers,calls,error=self.launch_fixture(fail_create=True,fail_save=2,observed=observed,enforce_locks=True)
        self.assertIsInstance(error,subprocess.TimeoutExpired)
        self.assertEqual(len(containers),1,'timeout must occur after the create side effect')
        self.assertEqual(sum(a[:2]==['docker','create'] for a in calls),1)
        self.assertFalse(any(a[:2] in [['docker','start'],['docker','stop'],['docker','rm'],['docker','rename']] for a in calls))
        before=observed['before_create'][0]
        self.assertIsNotNone(before['state'],'owner must be on disk before create, not first written by recovery')
        self.assertEqual(before['synced'],['file','directory'])
        self.assertEqual(observed['state'],before['state'],'failed recovery write must preserve initial ownership')
        state=observed['state'];self.assertEqual(state['containers'],[])
        self.assertFalse(state.get('ready'))
        c=json.loads((ROOT/'configs/cluster.example.json').read_text())
        self.assertEqual(state['configuration_sha256'],hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest())
        # The detached controller can discover the stopped candidate by name,
        # then require its exact ID, recorded owner and immutable image.
        cid,d=next(iter(containers.items()))
        self.assertRegex(cid,r'^[a-f0-9]{64}$');self.assertEqual(d['Id'],cid)
        self.assertEqual(d['Name'],'/'+c['deployment']+'-rank1')
        self.assertTrue(state['owner']);self.assertEqual(d['Config']['Labels'][cluster.LABEL],state['owner'])
        self.assertEqual(d['Image'],'sha256:'+'0'*64)
        self.assertFalse(d['State']['Running'])
        self.assertTrue(any('recovery state persistence failed' in note for note in error.__notes__))
        with tempfile.TemporaryDirectory() as td,patch.object(cluster,'ssh') as remote:
            path=pathlib.Path(td)/'state.json';path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'deployment state exists'):cluster.start(c,path)
            with self.assertRaisesRegex(ValueError,'partial deployment'):cluster.restart_owned(c,path)
            remote.assert_not_called()

    def test_initial_state_write_failure_prevents_create(self):
        observed={}
        containers,calls,error=self.launch_fixture(fail_save=1,observed=observed,enforce_locks=True)
        self.assertIsInstance(error,OSError)
        self.assertEqual(containers,{})
        self.assertFalse(any(a[:2]==['docker','create'] for a in calls))
        self.assertIsNone(observed['state'])
        self.assertTrue(any('recovery state persistence failed' in note for note in error.__notes__))

    def test_owner_is_durably_recorded_before_remote_create(self):
        import cluster
        observed={}
        containers,calls,error=self.launch_fixture(observed=observed,enforce_locks=True)
        self.assertIsNone(error)
        self.assertEqual(len(observed['before_create']),2)
        for index,before in enumerate(observed['before_create']):
            self.assertIsNotNone(before['state'],'create reached before ownership persistence')
            self.assertEqual(len(before['state']['containers']),index)
            self.assertEqual(before['synced'],['file','directory']*(index+1))
            self.assertEqual(before['state']['owner'],observed['state']['owner'])
        self.assertTrue(observed['state']['ready'])
        self.assertTrue(all(d['Config']['Labels'][cluster.LABEL]==observed['state']['owner'] for d in containers.values()))

    def test_equivalent_content_with_backend_specific_ids_pins_each_rank(self):
        metadata={'RootFS':{'Type':'layers','Layers':['sha256:'+'a'*64]},'Os':'linux','Architecture':'arm64','Config':{'Cmd':['serve']}}
        containers,calls,error=self.launch_fixture(parity=[{},{}],image_metadata=[metadata,metadata],enforce_locks=True)
        self.assertIsNone(error)
        self.assertEqual(len(containers),2)
        creates=[a for a in calls if a[:2]==['docker','create']]
        self.assertEqual([a[a.index('serve')-1] for a in creates],['sha256:'+'1'*64,'sha256:'+'0'*64])
        self.assertTrue(all(d['State']['Running'] for d in containers.values()))
        self.assertEqual(sum(a[:3]==['docker','image','inspect'] for a in calls),2)
        self.assertFalse(any(a[:2]==['docker','exec'] for a in calls))

    def test_different_ids_reject_changed_or_missing_content_before_create(self):
        import copy
        base={'RootFS':{'Type':'layers','Layers':['sha256:'+'a'*64,'sha256:'+'b'*64]},'Os':'linux','Architecture':'arm64','Config':{'Cmd':['serve']}}
        changes=[{'RootFS':{'Type':'layers','Layers':[]}},
                 {'RootFS':{'Type':'layers','Layers':['sha256:'+'c'*64]}},
                 {'RootFS':{'Type':'layers','Layers':list(reversed(base['RootFS']['Layers']))}},
                 {'RootFS':{'Type':'layers','Layers':['invalid']}},
                 {'RootFS':{}},{'Config':{'Cmd':['other']}},
                 {'Os':'windows'},{'Architecture':'amd64'},{'Variant':'v8'},
                 {'Os':None},{'Architecture':None}]
        for change in changes:
            with self.subTest(change=change):
                other=copy.deepcopy(base);other.update(change)
                containers,calls,error=self.launch_fixture(parity=[{},{}],image_metadata=[base,other])
                self.assertIsInstance(error,ValueError)
                self.assertFalse(containers)
                self.assertFalse(any(a[:2]==['docker','create'] for a in calls))

    def test_competitor_at_lock_acquisition_prevents_launch(self):
        containers,calls,error=self.launch_fixture(competitor=True)
        self.assertIsInstance(error,ValueError)
        self.assertFalse(containers)

    def test_healthy_locks_cover_preflights_and_rollback(self):
        for options in [dict(fail_start=True),dict(fail_save=1),dict(fail_save=2),dict(fail_save=3),dict(fail_save=4)]:
            with self.subTest(options=options):
                containers,calls,error=self.launch_fixture(enforce_locks=True,**options)
                self.assertIsNotNone(error)
                self.assertFalse(any(d['State']['Running'] for d in containers.values()))

    def test_dead_remote_lock_never_enters_body(self):
        import cluster, io
        from unittest.mock import MagicMock
        c=json.loads((ROOT/'configs/cluster.example.json').read_text())
        proc=MagicMock();proc.stdout=io.StringIO('LOCKED\n');proc.poll.return_value=1
        entered=False
        with patch.object(cluster,'resolve_cache',side_effect=lambda n:n),patch.object(cluster.subprocess,'Popen',return_value=proc),patch.object(cluster.select,'select',return_value=([1],[],[])):
            with self.assertRaises(RuntimeError):
                with cluster.remote_locks(c):entered=True
        self.assertFalse(entered,'dead holder allowed transaction body')

    def test_lock_context_exit_failure_is_inside_rollback_scope(self):
        containers,calls,error=self.launch_fixture(exit_lock=True)
        self.assertIsInstance(error,RuntimeError)
        self.assertEqual(len(containers),2)
        self.assertFalse(any(d['State']['Running'] for d in containers.values()))
        self.assertEqual(sum(a[:2]==['docker','stop'] for a in calls),2)

    def test_lock_loss_does_not_touch_intervening_owner(self):
        containers,calls,error=self.launch_fixture(exit_lock=True,intervening_owner=True)
        self.assertIsInstance(error,RuntimeError)
        self.assertEqual(sum(d['State']['Running'] for d in containers.values()),1)
        self.assertEqual(sum(a[:2]==['docker','stop'] for a in calls),1)
        self.assertTrue(any('cleanup failed' in n for n in error.__notes__))

    def test_partial_ownership_cleanup_does_not_stop_unrelated_rank(self):
        import cluster
        c=json.loads((ROOT/'configs/cluster.example.json').read_text())
        state={'owner':'owned','containers':[{'rank':0,'id':'a'*64},{'rank':1,'id':'b'*64}]}
        live={'a'*64:True,'b'*64:True}
        def inspect(n,cid):
            return {'Id':cid,'Config':{'Labels':{cluster.LABEL:'owned' if cid[0]=='a' else 'unrelated'}},'State':{'Running':live[cid]}}
        def stop(n,args,**kwargs):live[args[-1]]=False
        with patch.object(cluster,'inspect',side_effect=inspect),patch.object(cluster,'ssh',side_effect=stop) as remote:
            with self.assertRaises(RuntimeError):cluster.stop_owned(c,state)
        self.assertFalse(live['a'*64]);self.assertTrue(live['b'*64]);self.assertEqual(remote.call_count,1)

    def test_mid_start_and_post_readiness_lock_loss_roll_back(self):
        for fail_at in [2,4,6,8,9]:
            checks=[]
            def check():
                checks.append(1)
                if len(checks)>=fail_at:raise RuntimeError('lock lost')
            containers,calls,error=self.launch_fixture(lock_check=check)
            self.assertIsInstance(error,RuntimeError)
            self.assertFalse(any(d['State']['Running'] for d in containers.values()))
            self.assertGreaterEqual(len(checks),fail_at)

    def test_different_rank_images_require_measured_code_parity(self):
        for records in [[{},{}],[{'parity':{'python':{'x':'a'}}},{'parity':{'python':{'x':'b'}}}]]:
            containers,calls,error=self.launch_fixture(parity=records)
            self.assertIsInstance(error,ValueError)
            self.assertFalse(containers,'divergent images reached container creation')

    def test_matching_runtime_code_does_not_replace_identical_image_requirement(self):
        import runtime_check,copy
        self.assertTrue(hasattr(runtime_check,'compare_parity'),'runtime code parity checker missing')
        code={'schema':1,'versions':{'torch':'pinned'},'python':{'vllm/model.py':'a'*64},
              'native':{'vllm_exl3_c.so':{'text':'b'*64,'sass':'c'*64,'ptx':'d'*64},'exllamav3_ext.so':{'text':'b'*64,'sass':'c'*64,'ptx':'d'*64}}}
        # Real host ELF metadata with explicitly mocked device hashes.
        for native in code['native'].values():
            native.update(runtime_check.elf_code(pathlib.Path(sys.executable).resolve()))
        a={'parity':code,'native_elf_sha256':'e'*64};b=copy.deepcopy(a);b['native_elf_sha256']='f'*64
        runtime_check.compare_parity([a,b])
        containers,calls,error=self.launch_fixture(parity=[a,b])
        self.assertIsInstance(error,ValueError)
        self.assertFalse(containers)
        missing=copy.deepcopy(a);missing['parity']['native']['vllm_exl3_c.so'].pop('sass')
        with self.assertRaises(ValueError):runtime_check.compare_parity([missing,copy.deepcopy(missing)])
        for field in ['elf_header','program_headers','allocated','layout_sha256']:
            missing=copy.deepcopy(a)
            for native in missing['parity']['native'].values():native.pop(field,None)
            with self.subTest(field=field),self.assertRaises(ValueError):runtime_check.compare_parity([missing,copy.deepcopy(missing)])
        b['parity']['native']['vllm_exl3_c.so']['sass']='0'*64
        with self.assertRaises(ValueError):runtime_check.compare_parity([a,b])

    def test_state_write_failure_always_rolls_back_owned_containers(self):
        # Initial ownership save is followed by each create save and ready save.
        for fail_at,created,stopped in [(2,1,0),(3,2,0),(4,2,2)]:
            with self.subTest(fail_at=fail_at):
                containers,calls,error=self.launch_fixture(fail_save=fail_at)
                self.assertIsInstance(error,OSError)
                self.assertEqual(len(containers),created,'test did not reach intended creation phase')
                self.assertFalse(any(d['State']['Running'] for d in containers.values()), 'owned ranks left running')
                self.assertEqual(sum(a[:2]==['docker','stop'] for a in calls),stopped)
                self.assertIn('ownership-checked cleanup completed',error.__notes__)
                self.assertTrue(any('recovery state persistence failed' in note for note in error.__notes__))

class MountTests(unittest.TestCase):
    def test_deleted_model_does_not_prevent_owned_stop(self):
        import cluster, io
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);model=root/'model';model.mkdir();cache=root/'cache';cache.mkdir()
            c=json.loads((ROOT/'configs/cluster.example.json').read_text())
            for n in c['nodes']:n.update(model_path=str(model),cache_path=str(cache))
            cluster.validate(c);model.rmdir()
            proc=MagicMock();proc.stdout.readline.return_value='LOCKED\n';proc.poll.return_value=None
            state={'owner':'owner','containers':[{'rank':0,'id':'a'*64}]};running=True;wrong=False
            def inspect(n,cid):
                return {'Id':cid,'Config':{'Labels':{cluster.LABEL:'wrong' if wrong else 'owner'}},'State':{'Running':running}}
            real_run=subprocess.run
            def resolve_remote(n,args,**kwargs):
                nonlocal running
                if args[:2]==['docker','stop']:running=False;return
                # Popen is patched only while acquiring holders; execute resolver directly in a separate saved implementation.
                return real_run([sys.executable,'-S','-B','-c',*args[3:]],capture_output=True,text=True,check=True)
            with patch.object(cluster,'ssh',side_effect=resolve_remote),patch.object(cluster,'inspect',side_effect=inspect):
                # Resolve paths for real before substituting only the lock-holder process.
                resolver=getattr(cluster,'resolve_cache',cluster.resolve_mounts)
                resolved=[resolver(n) for n in c['nodes']]
                with patch.object(cluster,'resolve_cache',side_effect=resolved,create=True),patch.object(cluster.subprocess,'Popen',return_value=proc),patch.object(cluster.select,'select',return_value=([1],[],[])):
                    with cluster.remote_locks(c):cluster.stop_owned(c,state)
                self.assertFalse(running)
                running=True;wrong=True
                with self.assertRaises(RuntimeError):cluster.stop_owned(c,state)
                self.assertTrue(running)
                with self.assertRaises(subprocess.CalledProcessError):cluster.resolve_mounts(c['nodes'][0])

    def test_root_and_model_writable_aliases_rejected(self):
        import cluster, copy
        c = json.loads((ROOT/'configs/cluster.example.json').read_text())
        for model,cache in [('/srv/model','/.'),('/srv/model','///'),('/srv/model','/srv/model/'),('/srv/model','/srv/model/cache'),('/srv/model','/srv')]:
            bad = copy.deepcopy(c); bad['nodes'][0].update(model_path=model,cache_path=cache)
            with self.subTest(model=model,cache=cache),self.assertRaises(ValueError):cluster.validate(bad)

    def test_remote_symlink_aliases_are_rejected_before_bind(self):
        import cluster
        self.assertTrue(hasattr(cluster,'resolve_mounts'), 'remote canonicalization missing')
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td); model=root/'model';model.mkdir();cache=root/'cache';cache.mkdir()
            alias=root/'alias';alias.symlink_to(model,target_is_directory=True)
            def remote(n,args,**kwargs):
                self.assertEqual(args[:3],['python3','-S','-c'])
                return subprocess.run([sys.executable,'-S','-B','-c',*args[3:]],capture_output=True,text=True,check=True)
            with patch.object(cluster,'ssh',side_effect=remote):
                with self.assertRaises((ValueError,subprocess.CalledProcessError)):
                    cluster.resolve_mounts({'model_path':str(model),'cache_path':str(alias)})
                got=cluster.resolve_mounts({'model_path':str(model),'cache_path':str(cache)})
                self.assertEqual(got,{'model_path':str(model.resolve()),'cache_path':str(cache.resolve())})

class RedirectTests(unittest.TestCase):
    def test_api_redirect_policy_blocks_cross_origin_and_downgrade(self):
        import verify, urllib.request, urllib.error
        handler = getattr(verify, 'NoAPIRedirect', urllib.request.HTTPRedirectHandler)()
        req = urllib.request.Request('https://api.example/v1/models', headers={'Authorization':'Bearer '+__import__('uuid').uuid4().hex})
        for target in ['https://other.example/v1/models','http://api.example/v1/models','https://api.example/other']:
            with self.subTest(target=target), self.assertRaises((ValueError, urllib.error.HTTPError)):
                handler.redirect_request(req,None,302,'redirect',{},target)

    def test_actual_loopback_redirect_does_not_send_second_request(self):
        import verify,urllib.request,http.server,threading
        hits=[]
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                if self.path=='/start':
                    self.send_response(302);self.send_header('Location',f'http://localhost:{self.server.server_port}/sink');self.end_headers()
                else:self.send_response(200);self.end_headers()
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/start',headers={'Authorization':'Bearer '+__import__('uuid').uuid4().hex})
            with self.assertRaises(ValueError):verify.api_open(req,timeout=5)
            self.assertEqual(hits,['/start'])
        finally:server.shutdown();server.server_close();thread.join()

    def test_clients_share_redirect_safe_opener(self):
        import ast
        for name in ['verify.py','benchmark.py']:
            tree = ast.parse((ROOT/'scripts'/name).read_text())
            unsafe = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == 'urlopen']
            self.assertEqual(unsafe,[], 'API call bypasses redirect policy')

class ScaleTests(unittest.TestCase):
    def test_all_256_e8m0_values(self):
        import math, mxfp4
        for b in range(256):
            got = mxfp4.decode_e8m0_byte(b)
            if b == 255:self.assertTrue(math.isnan(got))
            else:self.assertEqual(got, math.ldexp(1.0, b-127))

    @unittest.skipUnless(__import__('importlib').util.find_spec('torch'), 'Torch not installed; no install permitted')
    def test_actual_torch_cpu_packed_endpoints(self):
        import torch, mxfp4
        scales = torch.arange(256, dtype=torch.uint8).reshape(256,1)
        packed = torch.tensor([0x80,0x21,0xF7,0xBA]*4, dtype=torch.uint8).repeat(256,1)
        actual = mxfp4.dequant_mxfp4(packed,scales)
        expected = mxfp4.unpack_e2m1(packed) * torch.tensor([mxfp4.decode_e8m0_byte(b) for b in range(256)]).reshape(256,1)
        torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0, equal_nan=True)
        self.assertTrue(torch.signbit(actual[127,1]))
        if hasattr(torch,'float8_e8m0fnu'):
            torch.testing.assert_close(scales.view(torch.float8_e8m0fnu).float().flatten(),
                torch.tensor([mxfp4.decode_e8m0_byte(b) for b in range(256)]),rtol=0,atol=0,equal_nan=True)

class ConverterDryRunTests(unittest.TestCase):
    def test_low_level_dry_run_never_writes_destination(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); src = root/'source'; src.mkdir()
            (src/'config.json').write_text('{}')
            (src/'model.safetensors.index.json').write_text('{"weight_map":{}}')
            (src/'tokenizer.json').write_text('{}')
            dst = root/'destination'
            for existing in [False, True]:
                if existing:
                    dst.mkdir(exist_ok=True)
                    (dst/'config.json').write_bytes(b'sentinel')
                before = {p.name:p.read_bytes() for p in dst.iterdir()} if existing else None
                result = subprocess.run([sys.executable,'-S','-B',str(ROOT/'tools/quantize_experts_exl3.py'),
                    '--src',str(src),'--dst',str(dst),'--dry-run'],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                if existing:self.assertEqual({p.name:p.read_bytes() for p in dst.iterdir()},before)
                else:self.assertFalse(dst.exists(), 'dry-run created destination')
