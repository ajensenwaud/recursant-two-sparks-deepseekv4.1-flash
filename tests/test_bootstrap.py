"""CPU-only first-run and repeat-start acceptance tests."""
import contextlib
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

class BootstrapTests(unittest.TestCase):
    def test_first_configuration_prompts_for_both_nodes_and_saves_valid_config(self):
        import bootstrap, cluster
        image = 'ghcr.io/example/runtime@sha256:' + 'a' * 64
        answers = ['demo']
        for rank in range(2):
            answers += [f'user@node{rank}', f'192.0.2.{rank+10}', 'fabric0', 'mlx5_0',
                        f'/srv/models/rank{rank}', f'/srv/cache/rank{rank}']
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td)/'cluster.json'
            with mock.patch('builtins.input', side_effect=answers):
                config = bootstrap.configure(path, image)
            self.assertEqual(cluster.validate(config), json.loads(path.read_text()))
            self.assertEqual(config['api_bind'], '127.0.0.1')
            self.assertNotIn('preserve_existing_public_bind', config)
            with self.assertRaises(FileExistsError):
                bootstrap.configure(path, image)

    def config(self):
        c = json.loads((ROOT/'configs/cluster.example.json').read_text())
        c['image'] = 'ghcr.io/example/runtime@sha256:' + 'a'*64
        return c

    def test_first_start_preflights_both_nodes_before_install_then_uses_transaction(self):
        import bootstrap
        c = self.config()
        r = {'image': c['image'], 'model': {'files': []}}
        events = []
        def node(n, action, data): events.append((action, n['ssh_host']))
        with tempfile.TemporaryDirectory() as td, mock.patch('bootstrap.ROOT', pathlib.Path(td)), mock.patch('bootstrap.remote_node', side_effect=node), mock.patch('bootstrap.cluster.start', side_effect=lambda *a: events.append(('start', 'transaction'))), mock.patch('bootstrap.cluster.remote_locks', return_value=contextlib.nullcontext(lambda: None)):
            bootstrap.start(c, r)
        self.assertEqual(events, [('preflight', 'rank-zero'), ('preflight', 'rank-one'),
                                  ('install', 'rank-zero'), ('install', 'rank-one'), ('start', 'transaction')])

    def test_second_preflight_failure_never_installs_or_starts(self):
        import bootstrap
        c = self.config()
        with tempfile.TemporaryDirectory() as td, mock.patch('bootstrap.ROOT', pathlib.Path(td)), mock.patch('bootstrap.remote_node', side_effect=[None, ValueError('missing prerequisite')]) as node, mock.patch('bootstrap.cluster.start') as launch:
            with self.assertRaisesRegex(ValueError, 'prerequisite'): bootstrap.start(c, {'image': c['image']})
            self.assertEqual([call.args[1] for call in node.call_args_list], ['preflight', 'preflight'])
            launch.assert_not_called()

    def test_second_start_checks_ownership_and_health_without_install_or_release(self):
        import bootstrap, bootstrap_node
        c = self.config()
        state = {'configuration_sha256': bootstrap_node.fingerprint(c), 'owner': 'owner', 'ready': True,
                 'containers': [{'rank': 0, 'id': 'a'*64}, {'rank': 1, 'id': 'b'*64}]}
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); (root/'.state').mkdir()
            (root/'.state'/ (c['deployment']+'.json')).write_text(json.dumps(state))
            with mock.patch('bootstrap.ROOT', root), mock.patch('bootstrap.remote_node', side_effect=AssertionError('installer on restart')), mock.patch('bootstrap.cluster.owned', return_value={'State': {'Running': True, 'OOMKilled': False}}) as owned, mock.patch('bootstrap.cluster.ssh', return_value=mock.Mock(returncode=0)), mock.patch('bootstrap.cluster.start', side_effect=AssertionError('replace healthy deployment')):
                bootstrap.start(c, None)
                self.assertEqual(owned.call_count, 2)

class ModelInstallTests(unittest.TestCase):
    def test_install_hashes_download_once_then_marker_skips_payload_and_network(self):
        import bootstrap_node
        payload = b'actual fixture bytes'
        row = {'path': 'fixture.bin', 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        model = {'repository': 'example/model', 'revision': 'a'*40, 'files': [row]}
        with tempfile.TemporaryDirectory() as td:
            dest, cache = pathlib.Path(td)/'model', pathlib.Path(td)/'cache'
            def retrieve(url, path, metadata):
                path.write_bytes(payload)
                from artifacts import verify_file
                verify_file(path, metadata)
            with mock.patch('bootstrap_node.RESERVE', 0), mock.patch('bootstrap_node.download', side_effect=retrieve) as get, mock.patch('bootstrap_node.validate_model') as validate:
                bootstrap_node.install_model(model, dest, cache)
                self.assertEqual((dest/'fixture.bin').read_bytes(), payload)
                self.assertEqual(get.call_count, 1)
                self.assertEqual(validate.call_count, 1)
            with mock.patch('bootstrap_node.download', side_effect=AssertionError('network/hash on restart')), mock.patch('bootstrap_node.validate_model', side_effect=AssertionError('scan on restart')):
                bootstrap_node.install_model(model, dest, cache)

    def test_real_resumed_http_transfer_and_receipt_no_second_network(self):
        import bootstrap_node, artifacts, http.server, threading
        payload = b'fixture-checksum-payload' * 100
        requests = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                offset = int(self.headers.get('Range', 'bytes=0-')[6:-1])
                requests.append(offset)
                self.send_response(206 if offset else 200)
                self.send_header('Content-Length', str(len(payload)-offset))
                if offset: self.send_header('Content-Range', f'bytes {offset}-{len(payload)-1}/{len(payload)}')
                self.end_headers(); self.wfile.write(payload[offset:])
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        m = {'repository': 'example/model', 'revision': 'a'*40, 'files': [{'path': 'fixture.bin', 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}]}
        try:
            with tempfile.TemporaryDirectory() as td:
                dest, cache = pathlib.Path(td)/'model', pathlib.Path(td)/'cache'
                dest.mkdir(); (dest/'fixture.bin.part').write_bytes(payload[:13])
                def transfer(url, path, row):
                    artifacts.download(f'http://127.0.0.1:{server.server_port}/fixture', path, row, allow_test_http=True)
                with mock.patch('bootstrap_node.RESERVE', 0), mock.patch('bootstrap_node.download', side_effect=transfer), mock.patch('bootstrap_node.validate_model'):
                    bootstrap_node.install_model(m, dest, cache)
                    bootstrap_node.install_model(m, dest, cache)
                self.assertEqual(requests, [13])
                self.assertEqual((dest/'fixture.bin').read_bytes(), payload)
                self.assertFalse((dest/'fixture.bin.part').exists())
                self.assertTrue((cache/'.recipe-model-verified.json').exists())
        finally:
            server.shutdown(); thread.join(); server.server_close()

class ArchiveInstallTests(unittest.TestCase):
    def fixture(self):
        source = hashlib.sha256((ROOT/'runtime/source-manifest.json').read_bytes()).hexdigest()
        metadata = {'Config': {'Labels': {'org.deepseek-vision-recipe.sources': source}},
                    'RootFS': {'Type': 'layers', 'Layers': ['sha256:'+'b'*64]}, 'Os': 'linux', 'Architecture': 'arm64'}
        archive = {'repository': 'example/runtime', 'revision': 'a'*40, 'path': 'runtime.tar.gz',
                   'size': 7, 'unpacked_size': 20, 'sha256': hashlib.sha256(b'archive').hexdigest(), 'image_metadata': metadata}
        return {'status': 'published', 'image': 'example:runtime', 'image_archive': archive,
                'source_manifest_sha256': source, 'model': {'repository': 'example/model', 'revision': 'c'*40,
                'files': [{'path': 'config.json', 'size': 1, 'sha256': 'd'*64}]}}

    def test_archive_release_is_pinned_and_rejects_incomplete_or_mutable_metadata(self):
        import dsv41, copy
        r = self.fixture()
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td)/'release.json'; path.write_text(json.dumps(r))
            self.assertEqual(dsv41.release(path), r)
            for key, bad in [('revision', 'main'), ('sha256', None), ('size', True), ('unpacked_size', 0), ('path', '../runtime.tar'), ('image_metadata', {})]:
                invalid = copy.deepcopy(r); invalid['image_archive'][key] = bad
                path.write_text(json.dumps(invalid))
                with self.subTest(field=key), self.assertRaises(ValueError): dsv41.release(path)

    def test_archive_pull_uses_node_installer_not_registry_or_gpu(self):
        import dsv41
        r = self.fixture()
        c = json.loads((ROOT/'configs/cluster.example.json').read_text()); c['image'] = r['image']
        with mock.patch('bootstrap.remote_node') as node, mock.patch('dsv41.cluster.ssh', side_effect=AssertionError('registry pull for archive')):
            dsv41.pull(c, r)
            self.assertEqual(node.call_count, 2)
            self.assertTrue(all(call.args[1] == 'image' for call in node.call_args_list))

    def test_transferred_installer_runs_in_fresh_stdlib_python_process(self):
        import bootstrap, os
        r = self.fixture()
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            node = {'ssh_host': 'fixture', 'model_path': str(root/'model'), 'cache_path': str(root/'cache')}
            docker = root/'docker'
            docker.write_text('#!' + sys.executable + '\nimport json,sys\n'
                              + 'info=' + repr({'DockerRootDir': str(root)}) + '\n'
                              + 'image=' + repr(dict(r['image_archive']['image_metadata'], Id='sha256:'+'a'*64)) + '\n'
                              + "print(json.dumps(info if sys.argv[1]=='info' else [image]))\n")
            docker.chmod(0o700)
            def local_transport(n, args, **kwargs):
                env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'])
                return subprocess.run([sys.executable]+args[1:], input=kwargs['input'], capture_output=True, text=True, env=env, timeout=30)
            with mock.patch('bootstrap.cluster.ssh', side_effect=local_transport):
                bootstrap.remote_node(node, 'image', {'node': node, 'release': r})
            self.assertFalse((root/'model').exists())
            self.assertTrue((root/'cache/.recipe-install.lock').exists())

    def test_archive_installs_once_and_verifies_content_not_backend_id(self):
        import bootstrap_node
        r = self.fixture(); calls = []; loaded = False
        metadata = dict(r['image_archive']['image_metadata'], Id='sha256:'+'e'*64)
        def run(args, **kw):
            nonlocal loaded
            calls.append(args)
            if args[:3] == ['docker', 'image', 'inspect']:
                return subprocess.CompletedProcess(args, 0 if loaded else 1, json.dumps([metadata]) if loaded else '', '')
            if args[:2] == ['docker', 'load']: loaded = True
            return subprocess.CompletedProcess(args, 0, '', '')
        def get(url, path, row): path.write_bytes(b'archive')
        with tempfile.TemporaryDirectory() as td, mock.patch('bootstrap_node.subprocess.run', side_effect=run), mock.patch('bootstrap_node.require_budgets'), mock.patch('bootstrap_node.download', side_effect=get) as download:
            bootstrap_node.install_image(r, pathlib.Path(td), '/docker')
            bootstrap_node.install_image(r, pathlib.Path(td), '/docker')
            self.assertEqual(download.call_count, 1)
            self.assertEqual(sum(c[:2] == ['docker', 'load'] for c in calls), 1)
            metadata['RootFS'] = {'Type': 'layers', 'Layers': ['sha256:'+'f'*64]}
            with self.assertRaisesRegex(ValueError, 'content'): bootstrap_node.install_image(r, pathlib.Path(td), '/docker')

class DiskBudgetTests(unittest.TestCase):
    def test_missing_bytes_accounts_for_resumable_partial_files(self):
        import bootstrap_node
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root/'a.part').write_bytes(b'partial')
            (root/'b').write_bytes(b'complete')
            rows = [{'path': 'a', 'size': 20}, {'path': 'b', 'size': 8}]
            self.assertEqual(bootstrap_node.missing_bytes(root, rows), 13)

    def test_disk_budgets_sum_shared_storage_but_not_separate_mounts(self):
        import bootstrap_node
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            with mock.patch('bootstrap_node.storage', side_effect=lambda p: (1 if p.name!='other' else 2, 100)), mock.patch('bootstrap_node.RESERVE', 0):
                bootstrap_node.require_budgets([(root/'model', 60), (root/'other', 60)])
                with self.assertRaisesRegex(ValueError, 'disk'):
                    bootstrap_node.require_budgets([(root/'model', 60), (root/'cache', 60)])

class PrerequisiteTests(unittest.TestCase):
    def test_valid_read_only_preflight_checks_fabric_and_disk_without_creating_paths(self):
        import bootstrap_node
        c = json.loads((ROOT/'configs/cluster.example.json').read_text())
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            c['nodes'][0].update(model_path=str(root/'model'), cache_path=str(root/'cache'))
            data = {'node': c['nodes'][0], 'rank': 0, 'config': c, 'release': ArchiveInstallTests().fixture()}
            def command(args, **kwargs):
                if args[:2] == ['docker', 'info']: return json.dumps({'Runtimes': {'nvidia': {}}, 'DockerRootDir': str(root)})
                if args[0] == 'ip': return json.dumps([{'addr_info': [{'local': c['nodes'][0]['fabric_ip']}]}])
                return ''
            original = pathlib.Path.is_dir
            def is_dir(path): return True if str(path).startswith(('/sys/', '/dev/')) else original(path)
            with mock.patch('bootstrap_node.platform.machine', return_value='aarch64'), mock.patch('bootstrap_node.shutil.which', return_value='/bin/tool'), mock.patch('bootstrap_node.command', side_effect=command), mock.patch('pathlib.Path.is_dir', is_dir), mock.patch('bootstrap_node.RESERVE', 0):
                self.assertEqual(bootstrap_node.preflight(data), str(root))
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_prerequisite_and_occupied_gpu_fail_before_install(self):
        import bootstrap_node
        c = json.loads((ROOT/'configs/cluster.example.json').read_text())
        data = {'node': c['nodes'][0], 'rank': 0, 'config': c, 'release': ArchiveInstallTests().fixture()}
        with mock.patch('bootstrap_node.platform.machine', return_value='aarch64'), mock.patch('bootstrap_node.shutil.which', return_value=None), mock.patch('bootstrap_node.install_image') as image, mock.patch('bootstrap_node.install_model') as model:
            with self.assertRaisesRegex(ValueError, 'docker'): bootstrap_node.preflight(data)
            image.assert_not_called(); model.assert_not_called()
        def command(args, **kwargs):
            if args[:2] == ['docker', 'info']: return json.dumps({'Runtimes': {'nvidia': {}}, 'DockerRootDir': '/docker'})
            if args[0] == 'nvidia-smi': return '777\n'
            return ''
        with mock.patch('bootstrap_node.platform.machine', return_value='aarch64'), mock.patch('bootstrap_node.shutil.which', return_value='/bin/tool'), mock.patch('bootstrap_node.command', side_effect=command):
            with self.assertRaisesRegex(ValueError, 'GPU'): bootstrap_node.preflight(data)

    def test_ssh_never_accepts_unknown_host_keys(self):
        import cluster
        with mock.patch('cluster.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')) as run:
            cluster.ssh({'ssh_host': 'example'}, ['true'])
            self.assertIn('StrictHostKeyChecking=yes', run.call_args.args[0])

    def test_foreign_container_name_is_never_removed(self):
        import bootstrap_node
        c = json.loads((ROOT/'configs/cluster.example.json').read_text())
        data = {'node': c['nodes'][0], 'rank': 0, 'config': c, 'release': ArchiveInstallTests().fixture()}
        calls = []
        def command(args, **kwargs):
            calls.append(args)
            if args[:2] == ['docker', 'info']: return json.dumps({'Runtimes': {'nvidia': {}}, 'DockerRootDir': '/docker'})
            if args[:2] == ['docker', 'ps']: return 'foreign-owner\n'
            return ''
        with mock.patch('bootstrap_node.platform.machine', return_value='aarch64'), mock.patch('bootstrap_node.shutil.which', return_value='/bin/tool'), mock.patch('bootstrap_node.command', side_effect=command):
            with self.assertRaisesRegex(ValueError, 'name'): bootstrap_node.preflight(data)
        self.assertFalse(any(x in ['rm', 'stop', 'create', 'start', 'load', 'pull'] for call in calls for x in call))

    def test_model_insufficient_space_and_symlink_alias_leave_no_weights(self):
        import bootstrap_node
        m = {'repository': 'example/model', 'revision': 'a'*40, 'files': [{'path': 'big', 'size': 100, 'sha256': 'a'*64}]}
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td); dest = root/'model'; cache = root/'cache'
            with mock.patch('bootstrap_node.available', return_value=99), mock.patch('bootstrap_node.download') as download:
                with self.assertRaisesRegex(ValueError, 'disk'): bootstrap_node.install_model(m, dest, cache)
                download.assert_not_called(); self.assertFalse(dest.exists())
            dest.mkdir(); alias = root/'alias'; alias.symlink_to(dest)
            with self.assertRaisesRegex(ValueError, 'symlink'): bootstrap_node.install_model(m, alias, cache)

class RestartTests(unittest.TestCase):
    def restart_fixture(self, exit_lock=False, fail_start=False, occupied=False):
        import cluster, bootstrap_node
        c = json.loads((ROOT/'configs/cluster.example.json').read_text())
        items = [{'rank': 1, 'id': 'b'*64, 'image': 'sha256:'+'c'*64}, {'rank': 0, 'id': 'a'*64, 'image': 'sha256:'+'c'*64}]
        state = {'owner': 'owner', 'configuration_sha256': bootstrap_node.fingerprint(c), 'containers': items}
        running = set(); calls = []
        def owned(n, item, owner):
            rank = item['rank']
            return {'Image': item['image'], 'Config': {'Cmd': cluster.serve_argv(c, rank)},
                    'State': {'Running': item['id'] in running, 'OOMKilled': False},
                    'Mounts': [{'Destination': '/model', 'Source': n['model_path'], 'RW': False, 'Type': 'bind'},
                               {'Destination': '/cache/huggingface', 'Source': n['cache_path'], 'RW': True, 'Type': 'bind'}]}
        def ssh(n, args, **kw):
            calls.append(args)
            if args[:2] == ['docker', 'start']:
                running.add(args[2])
                if fail_start: raise RuntimeError('injected start failure')
            if args[:2] == ['docker', 'stop']: running.discard(args[-1])
            return subprocess.CompletedProcess(args, 0, '123' if occupied and args[0]=='nvidia-smi' else '', '')
        @contextlib.contextmanager
        def locks():
            yield lambda: None
            if exit_lock: raise RuntimeError('lock exit failure')
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td)/'state.json'; path.write_text(json.dumps(state))
            with mock.patch('cluster.remote_locks', return_value=locks()), mock.patch('cluster.owned', side_effect=owned), mock.patch('cluster.ssh', side_effect=ssh), mock.patch('cluster.resolve_mounts', side_effect=lambda n: {k: n[k] for k in ['model_path', 'cache_path']}), mock.patch('cluster.host_probe', return_value={'available': 12*1024**3, 'oom': 0}):
                if exit_lock or fail_start or occupied:
                    with self.assertRaises((RuntimeError, ValueError)): cluster.restart_owned(c, path)
                    self.assertFalse(running, 'failed restart left GPU containers running')
                else:
                    cluster.restart_owned(c, path)
                    self.assertTrue(json.loads(path.read_text())['ready'])
            self.assertFalse(any('create' in x or 'pull' in x or 'load' in x for x in calls))
        return calls

    def test_stopped_owned_pair_restarts_existing_ids_without_install_or_create(self):
        calls = self.restart_fixture()
        self.assertEqual([x for x in calls if x[:2] == ['docker', 'start']], [['docker', 'start', 'b'*64], ['docker', 'start', 'a'*64]])

    def test_restart_lock_exit_and_partial_failure_stop_only_started_ids(self):
        for field in ['exit_lock', 'fail_start']:
            with self.subTest(failure=field): self.restart_fixture(**{field: True})

    def test_restart_refuses_unrelated_gpu_without_start_or_stop(self):
        calls = self.restart_fixture(occupied=True)
        self.assertFalse(any(x[:2] in [['docker','start'], ['docker','stop']] for x in calls))

class ShellTests(unittest.TestCase):
    def test_shell_help_and_missing_config_are_side_effect_free(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ['start.sh', 'stop.sh', 'status.sh']:
                path = ROOT/name
                self.assertTrue(path.is_file(), name + ' entry point missing')
                result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run([str(path), '--help'], cwd=td, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--config', result.stdout)
            path = pathlib.Path(td)/'missing.json'
            result = subprocess.run([str(ROOT/'start.sh'), '--config', str(path)], input='', cwd=td, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('No configuration', result.stderr)
            self.assertEqual(list(pathlib.Path(td).iterdir()), [])

    def test_invalid_noninteractive_config_never_contacts_ssh(self):
        import bootstrap
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td)/'bad.json'; path.write_text('{}')
            with mock.patch('bootstrap.cluster.ssh') as ssh:
                with self.assertRaises(ValueError): bootstrap.main(['--config', str(path)])
                ssh.assert_not_called()

if __name__ == '__main__': unittest.main()
