import contextlib, io, json, pathlib, sys, tempfile, unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import cluster, dsv41
class ExistingBind(unittest.TestCase):
 def test_explicit_configure_opt_in(self):
  with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
   p=pathlib.Path(td)/'config.json'
   dsv41.main(['configure','--local-image','test:local','--preserve-existing-public-bind','--config',str(p)])
   c=json.loads(p.read_text());self.assertTrue(c['preserve_existing_public_bind']);self.assertEqual(c['api_bind'],'0.0.0.0');self.assertEqual(c['api_port'],8000)
   cluster.validate(c)
 def test_default_rejects_and_invalid_optins(self):
  c=json.loads((cluster.ROOT/'configs/cluster.example.json').read_text());c['image']='test:local';c['api_bind']='0.0.0.0'
  with self.assertRaises(ValueError):cluster.validate(c)
  for value in [False,1,'true',None]:
   with self.subTest(value=value),self.assertRaises(ValueError):cluster.validate(dict(c,preserve_existing_public_bind=value))
  for changes in [{'api_port':8001},{'api_bind':'127.0.0.1'},{'api_bind':'192.168.1.1'}]:
   with self.subTest(changes=changes),self.assertRaises(ValueError):cluster.validate(dict(c,preserve_existing_public_bind=True,**changes))
 def test_flag_only_configure(self):
  for args in [['plan','--preserve-existing-public-bind'],['configure','--preserve-existing-public-bind=true']]:
   with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):dsv41.main(args)
