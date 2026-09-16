"""CPU-only recipe contract tests; never a claim of GPU validation."""
import importlib.util,json,pathlib,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
class DownloadTests(unittest.TestCase):
 def test_confined_artifact_paths(self):
  self.assertTrue((ROOT/'scripts/artifacts.py').is_file(),'artifact implementation missing')
  import artifacts
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td)
   self.assertEqual(artifacts.confined(root,'model-01.safetensors'),root/'model-01.safetensors')
   for name in ['../escape','/absolute','a/../../b','a\\b','a\nfile','']:
    with self.subTest(name=name),self.assertRaises(ValueError):artifacts.confined(root,name)
   (root/'link').symlink_to(root.parent,target_is_directory=True)
   with self.assertRaises(ValueError):artifacts.confined(root,'link/file')
class TransferTests(unittest.TestCase):
 def test_resume_checks_actual_bytes(self):
  import artifacts, hashlib, http.server, threading
  self.assertTrue(hasattr(artifacts,'download'),'verified downloader missing')
  data=b'pinned-test-artifact'*200
  class Handler(http.server.BaseHTTPRequestHandler):
   def do_GET(self):
    offset=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
    self.send_response(206 if offset else 200)
    self.send_header('Content-Length',str(len(data)-offset))
    if offset:self.send_header('Content-Range',f'bytes {offset}-{len(data)-1}/{len(data)}')
    self.end_headers();self.wfile.write(data[offset:])
   def log_message(self,*args):pass
  server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with tempfile.TemporaryDirectory() as td:
    p=pathlib.Path(td)/'fixture.bin';p.with_suffix('.bin.part').write_bytes(data[:127])
    meta={'path':p.name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    artifacts.download(f'http://127.0.0.1:{server.server_port}/file',p,meta,allow_test_http=True)
    self.assertEqual(p.read_bytes(),data)
    p.write_bytes(b'x'*len(data))
    with self.assertRaises(ValueError):artifacts.download(f'http://127.0.0.1:{server.server_port}/file',p,meta,allow_test_http=True)
  finally:server.shutdown();server.server_close();thread.join()
class ModelTests(unittest.TestCase):
 def test_header_rejects_truncation_and_duplicate_keys(self):
  import artifacts, struct
  self.assertTrue(hasattr(artifacts,'header'),'bounded header reader missing')
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/'x.safetensors'
   h=json.dumps({'x':{'dtype':'I16','shape':[2],'data_offsets':[0,4]}}).encode()
   p.write_bytes(struct.pack('<Q',len(h))+h+b'1234')
   self.assertEqual(artifacts.header(p)[0]['x']['shape'],[2])
   p.write_bytes(struct.pack('<Q',len(h))+h+b'123')
   with self.assertRaises(ValueError):artifacts.header(p)
   h=b'{"x":{},"x":{}}';p.write_bytes(struct.pack('<Q',len(h))+h)
   with self.assertRaises(ValueError):artifacts.header(p)
 def test_download_plan_is_pinned_and_side_effect_free(self):
  import subprocess
  p=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/download.py'),'--dry-run'],capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)
  d=json.loads(p.stdout);self.assertEqual(d['revision'],'dba1be0a40aa45a94ad051997016db3960a90277')
  self.assertEqual(d['shards'],48);self.assertGreater(d['bytes'],500_000_000_000)
class BuildTests(unittest.TestCase):
 def test_build_is_explicit_and_pinned(self):
  import subprocess
  p=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/build.py'),'--dry-run'],capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)
  d=json.loads(p.stdout);self.assertIn('docker',d['argv'][0]);self.assertIn('--platform=linux/arm64',d['argv'])
  dockerfile=(ROOT/'runtime/Dockerfile').read_text()
  self.assertIn('@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58',dockerfile)
  self.assertIn('12.1a',dockerfile);self.assertNotIn(':latest',dockerfile)
class RuntimeTests(unittest.TestCase):
 def test_exact_retained_overlay_audit(self):
  import subprocess
  p=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/runtime_check.py'),
   '--root',str(ROOT/'runtime/rootfs'),'--manifest',str(ROOT/'runtime/source-manifest.json'),'--sources-only'],capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)
  self.assertEqual(json.loads(p.stdout)['source_files'],17)
class PrepareTests(unittest.TestCase):
 def test_conversion_plan_has_retained_parameters(self):
  import subprocess
  p=subprocess.run([sys.executable,'-S','-B',str(ROOT/'scripts/prepare.py'),'--source','/source','--destination','/output','--dry-run'],capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)
  d=json.loads(p.stdout);self.assertEqual(d['shards'],48)
  self.assertIn('--greedy',d['example_argv']);self.assertIn('--beam',d['example_argv'])
  self.assertFalse(d['byte_identical_to_retained_claim'])
 def test_preserved_tensor_bytes_are_checked(self):
  import struct
  self.assertTrue((ROOT/'scripts/prepare.py').exists(),'prepare validator missing')
  import prepare
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);h=json.dumps({'kept':{'dtype':'I16','shape':[2],'data_offsets':[0,4]}}).encode()
   for name in ['a','b']:(root/name).write_bytes(struct.pack('<Q',len(h))+h+b'abcd')
   self.assertEqual(prepare.compare_shard(root/'a',root/'b'),0)
   (root/'b').write_bytes(struct.pack('<Q',len(h))+h+b'abce')
   with self.assertRaises(ValueError):prepare.compare_shard(root/'a',root/'b')
class ClusterTests(unittest.TestCase):
 def test_retained_launch_parameters_and_safe_defaults(self):
  self.assertTrue((ROOT/'scripts/cluster.py').exists(),'cluster launcher missing')
  import cluster
  c=cluster.validate(json.loads((ROOT/'configs/cluster.example.json').read_text()))
  args=cluster.serve_argv(c,0)
  self.assertNotIn('--trust-remote-code',args);self.assertNotIn('--language-model-only',args)
  self.assertEqual(args[args.index('--host')+1],'127.0.0.1')
  self.assertEqual(args[args.index('--max-model-len')+1],'1048576')
  self.assertEqual(args[args.index('--max-num-seqs')+1],'8')
  self.assertEqual(args[args.index('--kv-cache-memory')+1],'4294967296')
  spec=json.loads(args[args.index('--speculative-config')+1]);self.assertEqual(spec['num_speculative_tokens'],5)
  self.assertFalse(spec['enable_adaptive_verification']);self.assertEqual(spec['draft_sample_method'],'probabilistic')
  self.assertIn('--headless',cluster.serve_argv(c,1))
 def test_shell_injection_and_public_bind_refused(self):
  self.assertTrue((ROOT/'scripts/cluster.py').exists(),'cluster launcher missing')
  import cluster,copy
  c=json.loads((ROOT/'configs/cluster.example.json').read_text())
  for field,value in [('ssh_host','-oProxyCommand=evil'),('fabric_ip','1.2.3.4;touch x'),('model_path','/model,x=y'),('rdma_hca','hca;id')]:
   bad=copy.deepcopy(c);bad['nodes'][0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):cluster.validate(bad)
  c['api_bind']='0.0.0.0'
  with self.assertRaises(ValueError):cluster.validate(c)
class VerifyTests(unittest.TestCase):
 def test_png_fixture_and_native_usage_parser(self):
  self.assertTrue((ROOT/'scripts/verify.py').exists(),'HTTP verifier missing')
  import verify,struct,zlib
  png=verify.image_png('CAT',(255,0,0),'square')
  self.assertEqual(png[:8],b'\x89PNG\r\n\x1a\n')
  self.assertEqual(struct.unpack('>II',png[16:24]),(224,192))
  result=verify.parse_events([
   (1.0,{'choices':[{'delta':{'reasoning':'think'},'finish_reason':None}]}),
   (2.0,{'choices':[{'delta':{'content':'OK'},'finish_reason':'stop'}],'usage':{'completion_tokens':8}})
  ],0.0)
  self.assertEqual(result['completion_tokens'],8);self.assertEqual(result['content'],'OK')
  self.assertEqual(result['ttft_seconds'],1.0)
  with self.assertRaises(ValueError):verify.parse_events([(1.0,{'choices':[]})],0)
 def test_bf16_head_is_required(self):
  import check_model
  self.assertTrue(hasattr(check_model,'check_head'),'output-head check missing')
  check_model.check_head({'dtype':'BF16','shape':[129280,5120]})
  with self.assertRaises(ValueError):check_model.check_head({'dtype':'F8_E4M3','shape':[129280,5120]})
 def test_pack_config_rejects_wrong_codebook(self):
  self.assertTrue((ROOT/'scripts/check_model.py').exists(),'pack checker missing')
  import check_model
  with self.assertRaises(ValueError):check_model.check_config({'quantization_config':{'bits':2,'quant_method':'exl3','codebook':'mul1'}})
class SafetyTests(unittest.TestCase):
 def test_stop_checks_exact_owner_before_mutation(self):
  from unittest.mock import patch
  import cluster
  c=cluster.validate(json.loads((ROOT/'configs/cluster.example.json').read_text()))
  state={'owner':'test-owner','containers':[{'rank':0,'id':'a'*64}]}
  with patch.object(cluster,'inspect',return_value={'Id':'a'*64,'Config':{'Labels':{cluster.LABEL:'someone-else'}},'State':{'Running':True}}),patch.object(cluster,'ssh') as remote:
   with self.assertRaises(RuntimeError):cluster.stop_owned(c,state)
   remote.assert_not_called()
 def test_termination_signal_is_catchable_for_rollback(self):
  import cluster
  self.assertTrue(hasattr(cluster,'on_termination'),'SIGTERM rollback handling missing')
  with self.assertRaises(InterruptedError):cluster.on_termination(15,None)
 def test_large_preserved_shard_space_guard(self):
  import prepare
  self.assertTrue(hasattr(prepare,'required_free_bytes'),'preserved-shard disk guard missing')
  self.assertGreaterEqual(prepare.required_free_bytes(101_000_000_000),101_000_000_000+40*1024**3)
 def test_usage_window_requires_server_counts(self):
  self.assertTrue((ROOT/'scripts/benchmark.py').exists(),'benchmark utility missing')
  import benchmark
  self.assertEqual(benchmark.usage_rate([(10.0,20),(12.0,60)]),20.0)
  for rows in [[],[(1.0,2)],[(1.0,5),(2.0,4)]]:
   with self.assertRaises(ValueError):benchmark.usage_rate(rows)
if __name__=='__main__':unittest.main()
