"""Independent model publication: synthetic pins, no Hub or SSH traffic."""
import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import dsv41


class ModelReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = pathlib.Path(self.tmp.name)
        self.path = self.directory / 'release.json'
        self.record = {
            'status': 'unpublished', 'image': None,
            'source_manifest_sha256': hashlib.sha256(
                (ROOT / 'runtime/source-manifest.json').read_bytes()).hexdigest(),
            'model': {'status': 'published', 'repository': 'example/model',
                      'revision': 'b' * 40,
                      'files': [{'path': 'LICENSE', 'size': 1, 'sha256': 'c' * 64}]},
        }

    def save(self, record=None):
        self.path.write_text(json.dumps(self.record if record is None else record))
        return self.path

    def test_download_independent_model_without_config_or_remote_actions(self):
        destination = self.directory / 'model'
        with mock.patch('dsv41.download_model') as download, \
                mock.patch('dsv41.cluster.ssh') as ssh, \
                mock.patch('dsv41.subprocess.run') as run:
            dsv41.main(['download-model', '--release', str(self.save()),
                        '--config', str(self.directory / 'missing.json'),
                        '--destination', str(destination)])
            download.assert_called_once_with(self.record['model'], destination)
            ssh.assert_not_called()
            run.assert_not_called()

    def test_full_release_remains_compatible(self):
        self.record['status'] = 'published'
        self.record['image'] = 'ghcr.io/example/runtime@sha256:' + 'a' * 64
        del self.record['model']['status']
        for model_only in [False, True]:
            self.assertEqual(dsv41.release(self.save(), model_only=model_only), self.record)

    def test_false_model_only_flag_requires_full_publication(self):
        with self.assertRaisesRegex(ValueError, 'Release not published'):
            dsv41.release(self.save(), model_only=False)

    def test_unpublished_model_rejected_before_transfer(self):
        for status in [None, False, True, 'unpublished', 'Published']:
            with self.subTest(status=status), mock.patch('dsv41.download_model') as download:
                self.record['model']['status'] = status
                with self.assertRaisesRegex(ValueError, 'model artifact is not published'):
                    dsv41.main(['download-model', '--release', str(self.save()),
                                '--destination', str(self.directory / 'model')])
                download.assert_not_called()

    def test_model_only_keeps_all_pin_validation(self):
        changes = [{'repository': 'https://example/model'}, {'revision': 'main'},
                   {'revision': 'B' * 40}, {'files': []},
                   {'files': self.record['model']['files'] * 2}]
        for field, value in [('path', '../secret'), ('size', True), ('size', -1),
                             ('size', 1.0), ('sha256', 'c' * 63)]:
            row = dict(self.record['model']['files'][0], **{field: value})
            changes.append({'files': [row]})
        for change in changes:
            record = copy.deepcopy(self.record)
            record['model'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                dsv41.release(self.save(record), model_only=True)
        self.record['source_manifest_sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'source fingerprint'):
            dsv41.release(self.save(), model_only=True)

    def test_model_publication_cannot_enable_pull_plan_or_start(self):
        config = self.directory / 'config.json'
        # Explicit false must not select the legacy local SHA-ID bypass.
        config.write_text(json.dumps({'local_prebuilt': False, 'image': 'sha256:' + 'a' * 64}))
        with mock.patch('dsv41.cluster.ssh') as ssh, mock.patch('dsv41.subprocess.run') as run:
            for status in ['unpublished', 'published']:
                self.record['status'] = status
                for action in ['pull', 'plan', 'start']:
                    with self.subTest(status=status, action=action), self.assertRaises(ValueError):
                        dsv41.main([action, '--config', str(config), '--release', str(self.save())])
            ssh.assert_not_called()
            run.assert_not_called()

    def test_download_requires_destination(self):
        with mock.patch('dsv41.download_model') as download, self.assertRaises(SystemExit) as error:
            dsv41.main(['download-model', '--release', str(self.save())])
        self.assertEqual(error.exception.code, 2)
        download.assert_not_called()

    def test_malformed_release_and_pin_types_fail_closed(self):
        records = [None, [], True]
        for key in ['repository', 'revision']:
            record = copy.deepcopy(self.record)
            record['model'][key] = 42
            records.append(record)
        for record in records:
            with self.subTest(record=record):
                self.path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    dsv41.release(self.path, model_only=True)

    def test_malformed_manifest_fails_with_value_error(self):
        changes = [None, {}, 'LICENSE', [None], ['LICENSE'],
                   [{'path': 1, 'size': 1, 'sha256': 'c' * 64}],
                   [{'path': 'LICENSE', 'size': 1, 'sha256': 12}]]
        for files in changes:
            with self.subTest(files=files):
                record = copy.deepcopy(self.record)
                record['model']['files'] = files
                with self.assertRaises(ValueError):
                    dsv41.release(self.save(record), model_only=True)
