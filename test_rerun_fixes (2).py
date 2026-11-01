"""Focused checks for pre-rerun code changes; no raw data or pyvinecopulib needed."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm
import torch

ROOT = Path(__file__).resolve().parent


def function_from_source(module, name, symbols):
    """Compile one function in isolation when optional production packages are unavailable.

    Parameters:
    module (str): Module name (file next to this test).
    name (str): Name of the function.
    symbols (dict): Names the function needs in its global scope.

    Returns:
    callable: The compiled function.
    """
    tree = ast.parse((ROOT / f"{module}.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(symbols)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / f"{module}.py"), 'exec'), scope)
    return scope[name]


class TestPenaltyCorrections(unittest.TestCase):
    """Checks of the marginal-uniformity penalty and the global minibatch counter."""
    def test_independent_random_shifts(self):
        """The penalty grid is shifted by two independent random draws, one per axis."""
        penalty = function_from_source('codine', 'margin_uniformity_penalty', {'torch': torch})
        pts = []

        def density(u, w):
            """Test density that records the evaluation points.

            Parameters:
            u (torch.Tensor): Evaluation points.
            w (torch.Tensor): Covariate values.

            Returns:
            torch.Tensor: A smooth positive density.
            """
            pts.append(u.detach().clone())
            return 1 + 0.1 * (u[:, 0] - 0.5) * (u[:, 1] - 0.5)

        with patch.object(torch, 'rand', side_effect=[torch.tensor([0.2]), torch.tensor([0.8])]) as rand:
            value = penalty(density, torch.tensor([0.5]), n_grid=4)
        self.assertEqual(rand.call_count, 2)
        p = pts[0].reshape(4, 4, 2)
        torch.testing.assert_close(p[:, 0, 0], (torch.arange(4) + 0.2) / 4)
        torch.testing.assert_close(p[0, :, 1], (torch.arange(4) + 0.8) / 4)
        self.assertTrue(torch.isfinite(value))

    def test_global_minibatch_counter(self):
        """The training loops count minibatches over all epochs and apply the penalty at every margin_every-th step."""
        for name in ('codine', 'conditional_copula_flow'):
            text = (ROOT / f'{name}.py').read_text()
            self.assertIn('batch_step = 0\n    for epoch in pbar:', text)
            self.assertIn('batch_step % margin_every == 0', text)
            self.assertIn('batch_step += 1', text)
        m, epochs, every = 10, 4, 4
        self.assertEqual(every * sum(t % every == 0 for t in range(m * epochs)) / (m * epochs), 1.0)


class TestAcarSerialDependencePilot(unittest.TestCase):
    """Check of the serial-dependence pilot of the Acar GLRT."""
    def test_constant_copula_with_ar1_pairs(self):
        """The simulated pairs have the requested contemporaneous correlation and serial persistence."""
        simulator = function_from_source('acar_glrt', 'gaussian_ar1_pairs', {'np': np, 'norm': norm})
        u = simulator(np.random.default_rng(3), 5000, rho=0.4, phi=0.8)
        z = norm.ppf(u)
        self.assertEqual(u.shape, (5000, 2))
        self.assertAlmostEqual(np.corrcoef(z.T)[0, 1], 0.4, delta=0.06)
        self.assertAlmostEqual(np.corrcoef(z[:-1, 0], z[1:, 0])[0, 1], 0.8, delta=0.06)


class TestCalibrationGuards(unittest.TestCase):
    """Checks of the guards on the calibration file and on the calibrated values."""
    def test_loader_requires_current_production_calibration(self):
        """The penalty loader rejects a missing, outdated or invalid calibration file."""
        read = function_from_source('fold_pipeline', '_read_calibration', {'os': os, 'np': np})
        func = function_from_source('fold_pipeline', '_load_margin_penalties',
                                    {'os': os, 'Path': Path, 'np': np, 'ARTIFACT_DIR': None, '_read_calibration': read})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            func.__globals__['ARTIFACT_DIR'] = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                func()
            f = Path(tmp) / 'margin_penalty_calibrated.json'
            f.write_text(json.dumps(dict(codine=3, flow=10, smoke=False, real_binned_ks=0.09)))
            with self.assertRaisesRegex(ValueError, 'Outdated'):
                func()
            f.write_text(json.dumps(dict(schema_version=2, codine=3, flow=10, smoke=False,
                                         real_binned_ks=0.09)))
            with self.assertRaisesRegex(ValueError, 'Outdated'):
                func()
            cal = dict(schema_version=4, penalty_codine=3, penalty_flow_affine=10, penalty_flow_spline=30, smoke=False, real_binned_ks=0.09,
                       wd_codine=0.0, wd_flow_affine=1e-3, wd_flow_spline=1e-2)
            f.write_text(json.dumps(cal))
            self.assertEqual(func(), (3.0, 10.0, 30.0))
            with patch.dict(os.environ, {'THESIS_MARGIN_PENALTY': 'nan'}):
                with self.assertRaises(ValueError):
                    func()
            with patch.dict(os.environ, {'THESIS_MARGIN_PENALTY': '0'}):
                self.assertEqual(func(), (0.0, 0.0, 0.0))

    def test_weight_decays_come_from_the_calibration_file(self):
        """The weight decays are read from the calibration file and invalid values are rejected."""
        read = function_from_source('fold_pipeline', '_read_calibration', {'os': os, 'np': np})
        func = function_from_source('fold_pipeline', '_load_weight_decays',
                                    {'os': os, 'Path': Path, 'np': np, 'ARTIFACT_DIR': None, '_read_calibration': read,
                                     'WD_FALLBACK': (0.0, 1e-6, 1e-6)})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            func.__globals__['ARTIFACT_DIR'] = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                func()
            with patch.dict(os.environ, {'THESIS_ALLOW_UNCALIBRATED_IMPORT': '1'}):
                self.assertEqual(func(), (0.0, 1e-6, 1e-6))
            f = Path(tmp) / 'margin_penalty_calibrated.json'
            f.write_text(json.dumps(dict(schema_version=4, penalty_codine=3, penalty_flow_affine=10, penalty_flow_spline=30, smoke=False,
                                         real_binned_ks=0.09, wd_codine=0.0, wd_flow_affine=1e-3, wd_flow_spline=1e-2)))
            self.assertEqual(func(), (0.0, 1e-3, 1e-2))
            f.write_text(json.dumps(dict(schema_version=4, penalty_codine=3, penalty_flow_affine=10, penalty_flow_spline=30, smoke=False,
                                         real_binned_ks=0.09, wd_codine=0.0, wd_flow_affine=-1.0, wd_flow_spline=1e-2)))
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                func()

    def test_every_conditional_network_has_a_group_weight_decay(self):
        """Every conditional network variant gets the weight decay of its group."""
        import ast
        tree = ast.parse((ROOT / 'fold_pipeline.py').read_text())
        src = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertIn('wd_of_model', src)
        scope = {'CCODINE_VARIANTS': {'a': 1}, 'CCOPFLOW_VARIANTS': {'b': ('x', 'affine'), 'c': ('x', 'spline')},
                 'WD_CCODINE': 0.0, 'WD_CCOPFLOW_AFFINE': 1e-3, 'WD_CCOPFLOW_SPLINE': 1e-2}
        exec(compile(ast.Module(body=[src['wd_of_model'], src['fixed_wd_by_model']], type_ignores=[]), 'fp', 'exec'), scope)
        self.assertEqual(scope['fixed_wd_by_model'](), {'a': 0.0, 'b': 1e-3, 'c': 1e-2})
        self.assertEqual(scope['fixed_wd_by_model'](0.5), {'a': 0.5, 'b': 0.5, 'c': 0.5})

    def test_calibration_selection_uses_unclipped_excess(self):
        """The calibration chooses the penalty weight on the unclipped excess CVPE."""
        import pandas as pd
        from types import SimpleNamespace
        n = json.loads((ROOT / '03_marginal_uniformity_penalty_calibration.ipynb').read_text())
        cell = next(''.join(c['source']) for c in n['cells'] if 'FLOOR = 1e-5' in ''.join(c['source']))
        designs = ['design1', 'design2']
        names = [('C-CODINE', 'C-CODINE (w only)'), ('C-CopFlow, affine', 'C-CopFlow (w only, affine)'),
                 ('C-CopFlow, spline coupling', 'C-CopFlow (w only, spline coupling)')]
        rows = []
        for fam, model in names:
            for d in designs:
                rows.extend([dict(family=fam, model=model, dgp=d, n=300, lam=lam,
                                  excess_cvpe=score) for lam, score in [(0., 0.000001), (3., -0.01), (10., -0.001)]])
        scope = {'pd': pd, 'np': np, 'LONG': pd.DataFrame(rows), 'LAMBDAS': (0., 3., 10.),
                 'DESIGNS': designs, 'GROUP': {m: g for g, m in names}, 'RT': SimpleNamespace(write_table=lambda *a, **k: None),
                 'display': lambda *a: None}
        exec(cell, scope)
        self.assertTrue((scope['best'].best_lam == 3.).all())
        self.assertEqual(scope['CHOSEN'], {g: 3. for g, _ in names})

    def test_calibrated_values_are_reused_everywhere(self):
        """No notebook other than 03 sets the calibrated values itself, and synthetic_study reads them from fold_pipeline."""
        # Weight decay and penalty weight are tuned once (notebook 03) and read from the calibration file by every other notebook.
        for path in ROOT.glob('*.ipynb'):
            if path.name.startswith('03_'):
                continue
            text = path.read_text()
            for forbidden in ('margin_penalty=', 'weight_decay=', 'THESIS_CAL_', 'LAMBDAS'):
                self.assertNotIn(forbidden, text, (path.name, forbidden))
        self.assertIn('FP.fixed_wd_by_model()', (ROOT / 'synthetic_study.py').read_text())
        self.assertIn('FP.default_hyper()', (ROOT / 'synthetic_study.py').read_text())

    def test_missing_real_folds_fail_without_override(self):
        """Calibration without the real folds raises unless an explicit delta is given."""
        from types import SimpleNamespace
        from scipy.stats import kstest
        n = json.loads((ROOT / '03_marginal_uniformity_penalty_calibration.ipynb').read_text())
        cell = ''.join(n['cells'][4]['source'])
        f = SimpleNamespace(load_all_folds=lambda: (_ for _ in ()).throw(FileNotFoundError('no folds')))
        scope = dict(np=np, os=os, S=SimpleNamespace(W_RANGE=(0, 1)), FP=f, kstest=kstest, DELTA_MIN=0.10)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'requires all real-data folds'):
                exec(cell, scope)
        with patch.dict(os.environ, {'THESIS_CAL_DELTA': '0.4'}):
            exec(cell, scope)
        self.assertEqual(scope['DELTA'], 0.4)
        self.assertTrue(np.isnan(scope['REAL_KS']))


class TestNotebookRunner(unittest.TestCase):
    """Check of the resumable notebook runner."""
    def test_retries_failure_and_skips_unchanged_success(self):
        """A failed notebook is retried, and an unchanged successful notebook is skipped."""
        import run_all
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / 'marker.status'
            calls = []

            def run(_):
                """Fake notebook run that fails on the first call and succeeds afterwards.

                Parameters:
                _ (str): Notebook path; not used.

                Returns:
                bool: Success flag.
                """
                calls.append(True)
                return len(calls) != 1

            with patch.object(run_all, '_marker', return_value=status), \
                 patch.object(run_all, '_run_fingerprint', return_value='version-a'), \
                 patch.object(run_all, 'find', side_effect=lambda x: ['00_preprocessing.ipynb'] if x == '00' else []), \
                 patch.object(run_all, 'synthetic_prefixes', return_value=[]), \
                 patch.object(run_all, 'ORDER', ['00']), patch.object(run_all, 'run', side_effect=run):
                run_all.run_everything()
                self.assertEqual(json.loads(status.read_text())['status'], 'FAILED')
                run_all.run_everything()
                self.assertEqual(json.loads(status.read_text())['status'], 'ok')
                run_all.run_everything()
                self.assertEqual(len(calls), 2)
                with patch.object(run_all, '_run_fingerprint', return_value='version-b'):
                    run_all.run_everything()
                self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()


class TestTableWriters(unittest.TestCase):
    """Checks of the LaTeX table writers."""
    def test_number_formats_and_bold(self):
        """The number formats and the bold minimum of a row."""
        import reporting_tables as RT
        self.assertEqual(RT.tex_p(2e-5), r'$<0.0001$')
        self.assertEqual(RT.tex_p(0.0234), '0.023')
        self.assertEqual(RT.tex_num(-0.012, 2), r'$-$0.01')
        self.assertEqual(RT.tex_num(-0.001, 2), '0.00')
        self.assertEqual(RT.tex_bold_min([0.3, 0.1, np.nan], lambda x: f'{x:.1f}'), ['0.3', r'\textbf{0.1}', '--'])

    def test_weight_decay_table_bolds_row_minimum(self):
        """The weight-decay table marks the smallest validation CVPE of each row in bold."""
        import tempfile
        import pandas as pd
        import reporting_tables as RT
        old = RT.LATEX_DIR
        with tempfile.TemporaryDirectory() as d:
            RT.LATEX_DIR = d
            try:
                wd = pd.DataFrame([dict(model='C-CODINE (w only)', wd=w, best_val_cvpe=v)
                                   for w, v in zip((0.0, 1e-6, 1e-2), (0.1503, 0.1501, 0.1536))])
                tex = RT.wd_sensitivity_table(wd)
            finally:
                RT.LATEX_DIR = old
        self.assertIn(r'\textbf{0.1501}', tex)
        self.assertIn(r'$10^{-6}$', tex)
