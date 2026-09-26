"""Offline integration checks for the standalone Kaggle notebook.

Run: python -m unittest discover -s tests -p test_kaggle_notebook.py -v
Synthetic videos are temporary test fixtures, never real training data.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import pandas as pd
import torch

from src.data.transforms import eval_transform
from src.models.cnn_lstm import CNNLSTM
from src.infer import load_detector, run_inference_frames

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/kaggle_train_cnn_lstm.ipynb'


class KaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.sources = {c['id']: ''.join(c['source']) for c in cls.notebook['cells']
                       if c['cell_type'] == 'code'}
        cls.temp = tempfile.TemporaryDirectory(prefix='webcam-notebook-test-')
        cls.base = Path(cls.temp.name)
        rows = []
        rng = np.random.default_rng(42)
        for split in ('train', 'val', 'test'):
            for label in (0, 1):
                path = cls.base / f'{split}_{label}.avi'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 20, (128, 96))
                if not writer.isOpened():
                    raise RuntimeError('MJPG codec required for notebook tests')
                for _ in range(48):
                    writer.write(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8))
                writer.release()
                rows.append(dict(path=path.name, label=label, group_id=f'{split}_session',
                                 split=split, domain='webcam'))
        cls.csv = cls.base / 'manifest.csv'
        pd.DataFrame(rows).to_csv(cls.csv, index=False)
        cls.ns = {'__name__': '__notebook_test__'}
        exec(compile(cls.sources['imports'], 'imports', 'exec'), cls.ns)
        # Execute config but replace user-editable assignments through AST,
        # without ever creating /kaggle directories or downloading weights.
        config = ast.parse(cls.sources['config'])
        replace = {'MANIFEST_PATH': str(cls.csv), 'OUT_DIR': str(cls.base / 'output')}
        for statement in config.body:
            if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name == 'DATA_MODE':
                    statement.value = ast.Constant('manifest')
                if name in replace:
                    statement.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                                               args=[ast.Constant(replace[name])], keywords=[])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            exec(compile(ast.fix_missing_locations(config), 'config', 'exec'), cls.ns)
        cfg = cls.ns['CFG']
        cfg['data']['image_size'] = 32
        cfg['model'].update(backbone='resnet18', pretrained=False, lstm_hidden=8, lstm_layers=1)
        cfg['train'].update(epochs=2, warmup_epochs=1, num_workers=0, batch_size=2)
        cls.ns['DEVICE'] = torch.device('cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_all_cells_compile_and_run(self):
        for name, source in self.sources.items():
            compile(source, name, 'exec')
            if name in {'imports', 'config'}:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                exec(compile(source, name, 'exec'), self.ns)
        output = self.base / 'output'
        for filename in ('best.pt', 'metrics.json', 'manifest_resolved.csv',
                         'test_window_predictions.csv', 'validation_thresholds.csv',
                         'deployment_config.json'):
            self.assertTrue((output / filename).is_file(), filename)
        checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
        self.assertFalse(checkpoint['cfg']['model']['pretrained'])
        self.assertEqual(checkpoint['cfg']['infer']['threshold'], self.ns['threshold'])
        self.assertEqual(len(self.ns['history']), 2)

    def test_02_manifest_rejects_leakage_bad_labels_and_duplicates(self):
        validate = self.ns['validate_manifest']
        original = pd.read_csv(self.csv)
        invalid_csv = self.base / 'invalid.csv'
        scenarios = []
        leak = original.copy()
        leak.loc[leak.split == 'val', 'group_id'] = 'train_session'
        scenarios.append(leak)
        bad_label = original.copy()
        bad_label.loc[0, 'label'] = 3
        scenarios.append(bad_label)
        duplicate = original.copy()
        duplicate.loc[1, 'path'] = duplicate.loc[0, 'path']
        scenarios.append(duplicate)
        copy_path = self.base / 'duplicate_content.avi'
        copy_path.write_bytes((self.base / original.loc[0, 'path']).read_bytes())
        duplicate_content = original.copy()
        duplicate_content.loc[1, 'path'] = copy_path.name
        scenarios.append(duplicate_content)
        no_class = original[~((original.split == 'val') & (original.label == 1))]
        scenarios.append(no_class)
        for invalid in scenarios:
            invalid.to_csv(invalid_csv, index=False)
            with warnings.catch_warnings(), self.assertRaises(ValueError):
                warnings.simplefilter('ignore', UserWarning)
                validate(invalid_csv)
        original_path = self.ns['MANIFEST_PATH']
        missing = self.base / 'missing_webcam_manifest.csv'
        self.ns['MANIFEST_PATH'] = missing
        try:
            with self.assertRaises(FileNotFoundError):
                self.ns['prepare_manifest']()
            self.assertFalse(missing.exists())
        finally:
            self.ns['MANIFEST_PATH'] = original_path

    def test_03_sampling_and_preprocess_match_web(self):
        indices = self.ns['clip_indices'](20, 48, 0)
        np.testing.assert_array_equal(indices, np.arange(0, 48, 3))
        with self.assertRaises(ValueError):
            self.ns['clip_indices'](20, 20, 0)
        starts = self.ns['eval_starts'](5.0)
        self.assertAlmostEqual(starts[-1] + self.ns['CLIP_SPAN_SECONDS'], 5.0)
        frames = [np.full((96, 128, 3), i * 10, dtype=np.uint8) for i in range(16)]
        transport_frames = [self.ns['webcam_transport'](f) for f in frames]
        expected = eval_transform(32)(transport_frames)
        actual = self.ns['to_tensor_clip'](frames)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_04_export_loads_in_existing_backend(self):
        model, cfg, device = load_detector(self.base / 'output/best.pt')
        model.cpu()
        frames = [np.full((96, 128, 3), 100, dtype=np.uint8) for _ in range(16)]
        result = run_inference_frames(model, cfg, torch.device('cpu'), frames, with_boxes=False)
        self.assertIn(result['pred_class'], ['non_violence', 'violence'])
        self.assertTrue(0 <= result['violence_prob_mean'] <= 1)
        self.assertEqual(result['threshold'], self.ns['threshold'])
        local = CNNLSTM(cfg).eval()
        local.load_state_dict(self.ns['model'].state_dict(), strict=True)
        x = self.ns['to_tensor_clip'](frames).unsqueeze(0)
        with torch.no_grad():
            torch.testing.assert_close(local(x)[0], self.ns['model'].cpu().eval()(x)[0])

    def test_05_validation_only_threshold_and_domain_weights(self):
        predictions = pd.DataFrame({'label': [0, 0, 1, 1],
                                    'violence_prob': [0.1, 0.2, 0.3, 0.4]})
        selected, table = self.ns['choose_threshold'](predictions)
        self.assertAlmostEqual(selected['macro_f1'], 1.0)
        self.assertTrue(0.2 < selected['threshold'] <= 0.3)
        df = pd.DataFrame({'domain': ['webcam'] * 4 + ['external'] * 2,
                           'label': [0, 0, 0, 1, 0, 1]})
        weights = self.ns['sampling_weights'](df)
        self.assertAlmostEqual(sum(weights[:4]), 0.7)
        self.assertAlmostEqual(sum(weights[4:]), 0.3)
        self.assertAlmostEqual(sum(weights[:3]), weights[3])

    def test_06_warmup_freezes_cnn_and_keeps_batchnorm_in_eval(self):
        model = self.ns['model']
        self.ns['set_train_stage'](model, 1)
        self.assertFalse(any(p.requires_grad for p in model.spatial.parameters()))
        self.ns['set_train_stage'](model, 2)
        self.assertTrue(all(p.requires_grad for p in model.spatial.features[7].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.spatial.features[:7].parameters()))
        self.assertFalse(any(m.training for m in model.spatial.modules()
                             if isinstance(m, torch.nn.BatchNorm2d)))

    def test_07_airtlab_end_to_end_group_split_and_short_pairs(self):
        data_root = self.base / 'airtlab_input'
        output = self.base / 'airtlab_output'
        rng = np.random.default_rng(123)
        for class_name in ('non-violent', 'violent'):
            for camera in ('cam1', 'cam2'):
                folder = data_root / class_name / camera
                folder.mkdir(parents=True)
                for clip_id in range(1, 5):
                    writer = cv2.VideoWriter(str(folder / f'{clip_id}.avi'),
                                             cv2.VideoWriter_fourcc(*'MJPG'), 20, (128, 96))
                    self.assertTrue(writer.isOpened())
                    # Fourth pair: one camera too short, both must be excluded.
                    count = 40 if clip_id == 4 and camera == 'cam1' else 48
                    for _ in range(count):
                        writer.write(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8))
                    writer.release()
        ns = {'__name__': '__airtlab_notebook_test__'}
        exec(compile(self.sources['imports'], 'imports', 'exec'), ns)
        config = ast.parse(self.sources['config'])
        replace = {'DATA_ROOT': str(data_root), 'OUT_DIR': str(output)}
        for statement in config.body:
            if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name in replace:
                    statement.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                                               args=[ast.Constant(replace[name])], keywords=[])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            exec(compile(ast.fix_missing_locations(config), 'config', 'exec'), ns)
        ns['CFG']['data']['image_size'] = 32
        ns['CFG']['model'].update(backbone='resnet18', pretrained=False, lstm_hidden=8, lstm_layers=1)
        ns['CFG']['train'].update(epochs=2, warmup_epochs=1, num_workers=0, batch_size=2)
        ns['DEVICE'] = torch.device('cpu')
        for name, source in self.sources.items():
            if name not in {'imports', 'config'}:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    exec(compile(source, name, 'exec'), ns)
        manifest = pd.read_csv(output / 'manifest.csv')
        self.assertEqual(len(manifest), 12)
        self.assertTrue((manifest.groupby('group_id').split.nunique() == 1).all())
        self.assertTrue((manifest.groupby('group_id').size() == 2).all())
        self.assertEqual(set(manifest.domain), {'external'})
        self.assertEqual(len(pd.read_csv(output / 'excluded_airtlab.csv')), 4)
        self.assertFalse((data_root / 'manifest.csv').exists())
        self.assertIn('test_external', ns['reports'])
        self.assertNotIn('test_webcam', ns['reports'])
        exported_model, cfg, _ = load_detector(output / 'best.pt')
        self.assertEqual(cfg['evaluation_domain'], 'external')
        self.assertIsInstance(exported_model, CNNLSTM)
        # Reproduce the reported missing Output CSV after a session reset/removal.
        manifest_path = output / 'manifest.csv'
        manifest_path.unlink()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            recreated = ns['prepare_manifest']()
        pd.testing.assert_frame_equal(manifest, recreated[manifest.columns], check_dtype=False)
        original_bytes = manifest_path.read_bytes()
        original_mtime = manifest_path.stat().st_mtime_ns
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            ns['prepare_manifest']()
        self.assertEqual(original_bytes, manifest_path.read_bytes())
        self.assertEqual(original_mtime, manifest_path.stat().st_mtime_ns)
        again = ns['build_airtlab_manifest'](data_root, output / 'repeat.csv', ns['CFG']['seed'])
        pd.testing.assert_frame_equal(manifest, again[manifest.columns].reset_index(drop=True), check_dtype=False)
        (data_root / 'violent/cam2/1.avi').unlink()
        with self.assertRaisesRegex(ValueError, 'camera'):
            ns['build_airtlab_manifest'](data_root, output / 'invalid.csv', ns['CFG']['seed'])
        manifest_path.unlink()
        with self.assertRaisesRegex(ValueError, 'camera'):
            ns['prepare_manifest']()
        self.assertFalse(manifest_path.exists())


if __name__ == '__main__':
    unittest.main()
