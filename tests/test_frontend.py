import importlib.util,json,pathlib,sys,tempfile,unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class FrontendTests(unittest.TestCase):
 def test_build_installs_disassembler_for_real_cuda_parity(self):
  pins=json.loads((ROOT/'pins.json').read_text())
  self.assertEqual(pins['apt'].get('cuda-nvdisasm-13-0'),'13.0.85-1')
  self.assertIn('cuda-nvdisasm-13-0=13.0.85-1',(ROOT/'runtime/Dockerfile').read_text())

 def test_build_context_includes_model_contract(self):
  patterns=(ROOT/'.dockerignore').read_text().splitlines()
  self.assertIn('!tools/retained-model-contract.json',patterns)

 def test_start_uses_published_digest_and_existing_transaction(self):
  import dsv41
  image='ghcr.io/example/runtime@sha256:'+'a'*64
  c=json.loads((ROOT/'configs/cluster.example.json').read_text());c['image']=image
  with tempfile.TemporaryDirectory() as td,mock.patch('dsv41.release',return_value={'image':image}),mock.patch('dsv41.subprocess.run') as run:
   path=pathlib.Path(td)/'config.json';path.write_text(json.dumps(c))
   dsv41.main(['start','--config',str(path)])
   self.assertEqual(run.call_args.args[0][2:], [str(ROOT/'scripts/cluster.py'),'start','--config',str(path),'--execute','--acknowledge-unreleased'])
   c['image']='other:latest';path.write_text(json.dumps(c));run.reset_mock()
   with self.assertRaisesRegex(ValueError,'immutable'):dsv41.main(['start','--config',str(path)])
   run.assert_not_called()

 def test_logs_preserve_engine_stderr(self):
  import dsv41,hashlib,io,contextlib
  c=json.loads((ROOT/'configs/cluster.example.json').read_text())
  state={'owner':'test','configuration_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),'containers':[{'rank':0,'id':'a'*64}]}
  output=io.StringIO()
  with mock.patch('dsv41.load_json',return_value=state),mock.patch('dsv41.cluster.owned'),mock.patch('dsv41.cluster.ssh',return_value=mock.Mock(stdout='',stderr='engine diagnostic')),contextlib.redirect_stderr(output):
   dsv41.observe(c,'logs')
  self.assertIn('engine diagnostic',output.getvalue())

 def test_download_uses_exact_revision_and_manifest(self):
  import dsv41
  with tempfile.TemporaryDirectory() as td, mock.patch('dsv41.download') as get,mock.patch('dsv41.subprocess.run') as run:
   m={'repository':'example/model','revision':'b'*40,'files':[{'path':'LICENSE','size':1,'sha256':'c'*64}]}
   destination=pathlib.Path(td)/'model'
   dsv41.download_model(m,destination)
   self.assertEqual(get.call_args.args[0],'https://huggingface.co/example/model/resolve/'+'b'*40+'/LICENSE')
   self.assertEqual(get.call_args.args[1],destination/'LICENSE')
   self.assertIn('check_model.py',run.call_args.args[0][2])

 def test_observe_checks_exact_owner_and_config_without_model(self):
  import dsv41,hashlib
  c=json.loads((ROOT/'configs/cluster.example.json').read_text())
  state={'owner':'test-owner','configuration_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),'containers':[{'rank':0,'id':'a'*64},{'rank':1,'id':'b'*64}]}
  with mock.patch('dsv41.load_json',return_value=state),mock.patch('dsv41.cluster.owned') as owned,mock.patch('dsv41.cluster.ssh',return_value=mock.Mock(stdout='ok',stderr='')) as ssh:
   dsv41.observe(c,'logs');self.assertEqual(owned.call_count,2)
   self.assertEqual(ssh.call_args.args[1],['docker','logs','--tail','100','b'*64])
   dsv41.observe(c,'verify');self.assertEqual(ssh.call_args.args[1][:3],['docker','exec','a'*64])
   owned.side_effect=ValueError('ownership changed')
   ssh.reset_mock()
   with self.assertRaises(ValueError):dsv41.observe(c,'verify')
   ssh.assert_not_called()

 def test_pull_pins_and_reads_back_each_node(self):
  import dsv41,hashlib
  image='ghcr.io/example/runtime@sha256:'+'a'*64
  r={'image':image,'source_manifest_sha256':hashlib.sha256((ROOT/'runtime/source-manifest.json').read_bytes()).hexdigest()}
  c=json.loads((ROOT/'configs/cluster.example.json').read_text());c['image']=image
  record={'Architecture':'arm64','Os':'linux','RepoDigests':[image],'Config':{'Labels':{'org.deepseek-vision-recipe.sources':r['source_manifest_sha256']}}}
  def ssh(n,cmd,**kw):
   return mock.Mock(stdout=json.dumps([record]) if cmd[:3]==['docker','image','inspect'] else '')
  with mock.patch('dsv41.cluster.ssh',side_effect=ssh) as remote:
   dsv41.pull(c,r)
   self.assertEqual(remote.call_count,4)
   self.assertEqual(remote.call_args_list[0].args[1],['docker','pull','--platform=linux/arm64',image])
   record['RepoDigests']=[]
   with self.assertRaisesRegex(ValueError,'readback'):dsv41.pull(c,r)

 def test_lifecycle_delegates_existing_safety_controls(self):
  import dsv41
  with mock.patch('dsv41.subprocess.run') as run,mock.patch('dsv41.release',side_effect=AssertionError('recovery must not need published pins')):
   for action in ['stop','status']:
    dsv41.main([action,'--config','/tmp/test-config.json'])
    args=run.call_args.args[0]
    self.assertEqual(args[2:4],[str(ROOT/'scripts/cluster.py'),action])
    self.assertEqual('--execute' in args,action=='stop')

 def test_configure_and_plan_do_not_build_or_download(self):
  import dsv41
  with tempfile.TemporaryDirectory() as td, mock.patch('dsv41.subprocess.run') as run, mock.patch('dsv41.cluster.ssh') as ssh:
   cfg=pathlib.Path(td)/'cluster.json'
   dsv41.main(['configure','--config',str(cfg)])
   self.assertEqual(json.loads(cfg.read_text())['image'],'UNPUBLISHED')
   with self.assertRaises(FileExistsError):dsv41.main(['configure','--config',str(cfg)])
   with self.assertRaisesRegex(ValueError,'not published'):dsv41.main(['start','--config',str(cfg)])
   run.assert_not_called();ssh.assert_not_called()

 def test_cluster_accepts_digest_and_refuses_injection(self):
  import cluster
  c=json.loads((ROOT/'configs/cluster.example.json').read_text())
  c['image']='ghcr.io/example/runtime@sha256:'+'a'*64
  self.assertEqual(cluster.validate(c)['image'],c['image'])
  for image in ['image:latest;true','ghcr.io/example/runtime@sha256:'+'a'*63]:
   c['image']=image
   with self.assertRaises(ValueError):cluster.validate(c)

 def test_unpublished_release_fails_before_network(self):
  import dsv41
  with mock.patch('dsv41.cluster.ssh') as ssh:
   with self.assertRaisesRegex(ValueError,'not published'):
    dsv41.release(ROOT/'configs/release.json')
   ssh.assert_not_called()

 def test_release_rejects_mutable_or_mismatched_pins(self):
  import dsv41,hashlib
  digest=hashlib.sha256((ROOT/'runtime/source-manifest.json').read_bytes()).hexdigest()
  valid={'status':'published','image':'ghcr.io/example/runtime@sha256:'+'a'*64,
         'source_manifest_sha256':digest,'model':{'repository':'example/model','revision':'b'*40,'files':[{'path':'LICENSE','size':1,'sha256':'c'*64}]}}
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/'release.json';p.write_text(json.dumps(valid))
   self.assertEqual(dsv41.release(p)['image'],valid['image'])
   for key,value in [('image','ghcr.io/example/runtime:latest'),('source_manifest_sha256','d'*64)]:
    p.write_text(json.dumps(dict(valid,**{key:value})))
    with self.assertRaises(ValueError):dsv41.release(p)
   for change in [{'revision':'main'},{'files':[{'path':'../secret','size':1,'sha256':'c'*64}]},{'files':[{'path':'LICENSE','size':True,'sha256':'c'*64}]}]:
    p.write_text(json.dumps(dict(valid,model=dict(valid['model'],**change))))
    with self.assertRaises(ValueError):dsv41.release(p)
