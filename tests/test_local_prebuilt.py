import inspect,json,pathlib,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import dsv41,cluster,runtime_check
class LocalPrebuiltTests(unittest.TestCase):
 def test_local_configure_and_plan(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/'cluster.json'
   image='sha256:'+'a'*64
   dsv41.main(['configure','--local-image',image,'--config',str(p)])
   self.assertEqual(cluster.validate(json.loads(p.read_text()))['image'],image)
   dsv41.main(['plan','--config',str(p)])
 def test_local_tag_configure_plan_and_start_without_release(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/'cluster.json'
   with patch.object(dsv41,'release',side_effect=AssertionError('publication lookup')):
    dsv41.main(['configure','--local-image','dsv41-vision:local-prebuilt-20260916','--config',str(p)])
    c=cluster.validate(json.loads(p.read_text()))
    self.assertIs(c['local_prebuilt'],True)
    with patch.object(dsv41.subprocess,'run') as run:
     for action in ['plan','start']:
      dsv41.main([action,'--config',str(p)])
      self.assertIn(action,run.call_args.args[0])
    with self.assertRaises(ValueError):dsv41.main(['pull','--config',str(p)])
 def test_optional_local_flag_is_strict_and_preserves_old_configs(self):
  c=json.loads((ROOT/'configs/cluster.example.json').read_text())
  self.assertEqual(cluster.validate(c),c)
  self.assertNotIn('local_prebuilt',c)
  for value in [True,False]:
   self.assertIs(cluster.validate(dict(c,local_prebuilt=value))['local_prebuilt'],value)
  for value in ['true','false',0,1,None,{}]:
   with self.subTest(value=value),self.assertRaises(ValueError):cluster.validate(dict(c,local_prebuilt=value))
 def test_legacy_sha_config_still_plans_without_release(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as td:
   c=json.loads((ROOT/'configs/cluster.example.json').read_text());c['image']='sha256:'+'a'*64
   p=pathlib.Path(td)/'cluster.json';p.write_text(json.dumps(c))
   with patch.object(dsv41,'release',side_effect=AssertionError('publication lookup')),patch.object(dsv41.subprocess,'run'):
    dsv41.main(['plan','--config',str(p)])
 def test_invalid_local_tags_are_rejected_without_writing(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/'cluster.json'
   for image in ['', 'bad tag:tag','../image:tag','image:-bad','image:tag;touch','UPPER:tag']:
    with self.subTest(image=image),self.assertRaises(SystemExit):
     dsv41.main(['configure','--local-image',image,'--config',str(p)])
    self.assertFalse(p.exists())
 def test_start_is_cheap(self):
  source=inspect.getsource(cluster.start)
  for forbidden in ['runtime_check.py','check_model.py','verify.py','compare_parity(runtime_records)']:
   self.assertNotIn(forbidden,source)

 def test_default_build_check_does_not_disassemble_vendor(self):
  self.assertNotIn('runtime_parity(site,actual)',inspect.getsource(runtime_check.main))
