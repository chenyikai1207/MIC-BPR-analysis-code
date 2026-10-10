"""SYNTHETIC_TEST_ONLY: counterfactual invariants and direct library oracles.

All signals/labels/identifiers are generated here. Tests do not load original
data or validate a raw-to-manuscript workflow. Temporary artifacts are retained
outside this repository; no deletion is performed.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[key] = '2'
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits
from mic_intervention_api import *
from meg_source import fit_source_arrays


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        # SYNTHETIC_TEST_ONLY, no study assignment or efficacy seed.
        self.rng = np.random.default_rng(730031)
        self.true = np.repeat([0, 1], 12)
        self.codes = np.tile([0, 0, 1, 1], 6)[None, :]
        self.codes = np.concatenate([self.codes, 1 - self.codes])
        self.half = np.zeros(24, dtype=int)
        self.time = np.arange(24, dtype=float)
        self.plan = strict_factorial_plan(self.codes, self.true, self.half, self.time)
        self.xp, self.xi = self.rng.normal(size=(24, 16)), self.rng.normal(size=(24, 16))

    def test_strict_support_and_nearest_tie(self):
        self.assertTrue(self.plan['eligible'].all())
        self.assertEqual(self.plan['k'][0, 0], 1)
        self.assertEqual(self.plan['l'][0, 0], 2)
        tied = self.time.copy(); tied[2] = 2.; tied[3] = -2.
        self.assertEqual(strict_factorial_plan(self.codes, self.true, self.half, tied)['l'][0, 0], 2)

    def test_unsupported_reason_without_fallback(self):
        half = self.half.copy(); half[0:3] = 1
        labels = self.codes.copy(); labels[:, 0:3] = np.array([[1, 0, 0], [0, 1, 1]])
        plan = strict_factorial_plan(labels, self.true, half, self.time)
        self.assertFalse(plan['eligible'][0, 0])
        self.assertEqual(plan['reason_code'][0, 0], 1)
        self.assertEqual(plan['k'][0, 0], -1)
        half[12:16] = 1
        labels[:, 12:16] = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        plan = strict_factorial_plan(labels, self.true, half, self.time)
        self.assertEqual(plan['reason_code'][0, 12], 2)
        self.assertEqual(plan['l'][0, 12], -1)

    def test_reject_bad_labels_metadata_and_frozen_donor(self):
        with self.assertRaises(ValueError): strict_factorial_plan(self.codes.astype(float), self.true, self.half, self.time)
        with self.assertRaises(ValueError): strict_factorial_plan(self.codes, self.true, self.half, self.time * np.nan)
        wrong = self.plan['k'].copy(); wrong[0, 0] = 5
        with self.assertRaises(ValueError): strict_factorial_plan(self.codes, self.true, self.half, self.time, wrong)

    def test_four_training_cells_share_counts_partner_and_fixed_labels(self):
        before = self.codes.copy()
        for q in [0, 1]:
            for j in [0, 9, 23]:
                records = []
                for arm in ARMS:
                    y, deleted = training_cell(self.codes, self.plan['k'], self.plan['l'], q, j, arm)
                    keep = np.arange(24) != deleted
                    records.append(np.bincount(y[keep], minlength=2))
                    self.assertEqual(keep.sum(), 23)
                    if arm.endswith('X'):
                        self.assertEqual(np.sum(y != before[q]), 2)
                        self.assertNotEqual(y[self.plan['l'][q, j]], before[q, self.plan['l'][q, j]])
                for counts in records[1:]: assert_array_equal(counts, records[0])
        assert_array_equal(self.codes, before)

    def test_all_four_decisions_against_direct_sklearn(self):
        ev = score_four_cell(self.xp, self.xi, self.codes, self.plan)
        for q in [0, 1]:
            for j in [0, 11, 23]:
                for ai, arm in enumerate(ARMS):
                    y, deleted = training_cell(self.codes, self.plan['k'], self.plan['l'], q, j, arm)
                    keep = np.arange(24) != deleted
                    ref = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(self.xp[keep], y[keep])
                    expected = ref.decision_function(self.xi[j:j+1])[0]
                    assert_allclose(ev[q, j, ai], expected, atol=1e-8, rtol=1e-9)
                    self.assertEqual(ev[q, j, ai] > 0, expected > 0)

    def test_imagery_does_not_change_fitted_models(self):
        w, a = masked_binary_lda(self.xp, self.codes, np.array([1, 2]))
        changed_i = self.xi + 9000
        w2, a2 = masked_binary_lda(self.xp, self.codes, np.array([1, 2]))
        assert_array_equal(w, w2); assert_array_equal(a, a2)
        self.assertFalse(np.array_equal(self.xi @ w[0], changed_i @ w2[0]))

    def test_identical_same_code_inputs_have_zero_input_effect(self):
        xp = self.xp.copy(); xp[1] = xp[0]
        ev = score_four_cell(xp, self.xi, self.codes, self.plan)
        assert_allclose(ev[0, 0, 0], ev[0, 0, 1], atol=1e-12)
        assert_allclose(ev[0, 0, 2], ev[0, 0, 3], atol=1e-12)

    def test_class_balanced_accuracy_threshold_and_support(self):
        y = np.array([[0, 0, 0, 1]])
        ev = np.array([[[0.], [-1.], [1.], [1.]]])
        ba = balanced_code_accuracies(y, np.ones_like(y, bool), ev)
        assert_allclose(ba, [[5/6]])
        with self.assertRaises(ValueError): balanced_code_accuracies(y, np.array([[True, True, True, False]]), ev)

    def test_code_then_person_mean_and_J_sign(self):
        # SYNTHETIC_TEST_ONLY analytic values, not real results.
        ba = np.array([[[.8, .5, .4, .6]], [[.6, .4, .5, .5]]])
        ep = factorial_person_endpoints(ba, ['B4'])
        self.assertAlmostEqual(ep['B4_J_INTERACTION_pp'], 35.)
        self.assertAlmostEqual(ep['B4_E_C_pp'], 25.)
        self.assertAlmostEqual(ep['B4_E_X_pp'], -10.)

    def test_paired_person_bootstrap_against_count_weights(self):
        values = np.array([[1., 11.], [2., 12.], [8., 18.]])
        idx = np.array([[0, 0, 1], [2, 2, 2], [1, 0, 2], [1, 1, 1]])
        got = participant_bootstrap(values, idx)
        weights = np.stack([np.bincount(row, minlength=3) for row in idx])
        independent = weights @ values / 3
        assert_allclose(got['CI95'], np.quantile(independent, [.025, .975], axis=0))
        assert_allclose(participant_bootstrap(values[:, 1]-values[:, 0], idx)['CI95'], [10., 10.])

    def test_bootstrap_rejects_bad_order_nan_and_indices(self):
        with self.assertRaises(ValueError): participant_bootstrap([1., np.nan], np.array([[0, 1], [1, 0]]))
        with self.assertRaises(ValueError): participant_bootstrap([1., 2.], np.array([[0, 2], [0, 1]]))
        with self.assertRaises(ValueError): participant_bootstrap([1., 2.], np.array([[0], [1]]))

    def test_complete_source_variance_against_pseudovalues(self):
        full = np.array([1., 2., 5., 8.])
        deleted = full[None, :] + np.array([-.2, .1, .4, -.1])[:, None]
        got = source_target_sensitivity(full, deleted)
        pseudo = 4 * full.mean() - 3 * deleted.mean(1)
        self.assertAlmostEqual(got['source_variance_pp2'], np.var(pseudo, ddof=1) / 4)
        self.assertAlmostEqual(got['target_variance_pp2'], np.var(full, ddof=1) / 4)
        self.assertFalse(got['independent_replication'])
        with self.assertRaises(ValueError): source_target_sensitivity(full, deleted[:, :3])

    def test_source_paired_subtraction_before_sensitivity(self):
        a = np.array([[1., 2., 3.], [4., 5., 6.]])
        b = a - 2.
        assert_allclose(paired_endpoint_difference(a, b), np.full((2, 3), 2.))
        with self.assertRaises(ValueError): paired_endpoint_difference(a, b[:, :2])

    def test_direction_zero_equals_original_library(self):
        sd = self.rng.normal(size=16)
        w, a, wd, ad = direction_calibrators(self.xp, self.true, sd, [0., .5, 1.])
        for j in [0, 11, 23]:
            keep = np.arange(24) != j
            ref = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(self.xp[keep], self.true[keep])
            assert_allclose(self.xi @ w[j, 0] + a[j, 0], ref.decision_function(self.xi), atol=1e-8, rtol=1e-9)
        cov, means, counts = loo_statistics(self.xp, self.true)
        assert_allclose(w[:, 2], np.linalg.solve(cov[0], np.broadcast_to(sd, (24, 16))[..., None])[..., 0])
        assert_allclose(wd, (means[0, :, 1]-means[0, :, 0]) / np.diagonal(cov[0], axis1=1, axis2=2))

    def test_scored_own_P_and_label_cannot_change_own_model(self):
        sd = self.rng.normal(size=16)
        before = direction_calibrators(self.xp, self.true, sd, [0., .5, 1.])
        p, y = self.xp.copy(), self.true.copy(); p[0] += 1000; y[0] = 1-y[0]
        after = direction_calibrators(p, y, sd, [0., .5, 1.])
        for x, z in zip(before, after): assert_allclose(x[0], z[0], atol=1e-12, rtol=0)

    def test_equal_person_source_direction(self):
        means = np.stack([np.zeros((2, 16)), np.zeros((2, 16))])
        means[0, 1] = 2.; means[1, 1] = 10.
        m = dict(scaler_mean=np.zeros(16), scaler_scale=np.ones(16), pca_mean=np.zeros(16),
                 pca_components=np.eye(16), p_mean=np.zeros(16), B4_P_weights=np.eye(16))
        assert_allclose(source_direction(means, m), np.full(16, 6.))

    def test_source_only_selection_fixed_grid_ties_and_invalid(self):
        ba = np.full((3, 5), .5); ba[:, 1] += 5e-13
        self.assertEqual(select_source_lambda(ba)['chosen_lambda'], 0.)
        self.assertEqual(select_source_lambda(ba)['target_scores_used'], 0)
        with self.assertRaises(ValueError): select_source_lambda(ba, [0., .1, .5, .75, 1.])
        ba[0, 0] = np.nan
        with self.assertRaises(ValueError): select_source_lambda(ba)

    def test_complete_source_refit_on_small_synthetic_arrays(self):
        owner = np.repeat(np.arange(35), 4); y = np.tile([0, 1, 0, 1], 35)
        p = self.rng.normal(size=(140, 64)); i = p + self.rng.normal(scale=.4, size=p.shape)
        uid = np.array([f'SYNTHETIC_TEST_ONLY_UID_{j}' for j in range(140)])
        ids = [f'SYNTHETIC_TEST_ONLY_SOURCE_{j}' for j in range(35)]
        full = fit_source_arrays(p, i, y, ids, uid)
        keep = owner != 0
        deleted = fit_source_arrays(p[keep], i[keep], y[keep], ids[1:], uid[keep])
        self.assertNotIn(ids[0], deleted['source_ids'])
        self.assertTrue(set(deleted['source_original_uids']).isdisjoint(set(uid[~keep])))
        self.assertEqual(len(deleted['source_ids']), 34)
        self.assertFalse(np.array_equal(full['scaler_mean'], deleted['scaler_mean']))

    def test_cli_plan_score_and_fixed_scoring_labels(self):
        self.cli_fixture(full_direction=False)

    def test_cli_direction_source_selection_and_target_score(self):
        self.cli_fixture(full_direction=True)

    def cli_fixture(self, full_direction):
        scratch = os.environ.get('MIC_SYNTHETIC_SCRATCH')
        if scratch:
            Path(scratch).mkdir(parents=True, exist_ok=True)
        d = Path(tempfile.mkdtemp(prefix='MIC_SYNTHETIC_TEST_ONLY_', dir=scratch))
        cli = Path(__file__).with_name('mic_interventions_cli.py')
        q = {'status': 'QUALIFIED_LOCALLY', 'modality': 'MEG', 'scope': 'SYNTHETIC_TEST_ONLY',
             **{key: True for key in ['snapshot_identity_verified', 'eligibility_verified', 'event_uid_row_order_verified',
                  'feature_contract_verified', 'frozen_roles_verified', 'source_LOPO_exclusion_verified',
                  'complete_source_refits_verified', 'same_target_support_verified']},
             'local_qualification_evidence': 'SYNTHETIC_TEST_ONLY generated fixture; no real qualification'}
        (d/'q.json').write_text(json.dumps(q), encoding='utf-8')
        rawp, rawi = np.zeros((24, 24480)), np.zeros((24, 24480))
        rawp[:, :16], rawi[:, :16] = self.xp, self.xi
        uid = np.array([f'SYNTHETIC_TEST_ONLY_TARGET_UID_{j}' for j in range(24)])
        np.savez(d/'target.npz', P=rawp, I=rawi, y=self.true, original_uid=uid,
                 owner=np.full(24, 'SYNTHETIC_TEST_ONLY_TARGET'), pause_halfblock=self.half, onset_log_sec=self.time)
        pc = np.zeros((24480, 16)); pc[:16] = np.eye(16)
        suid = np.array([f'SYNTHETIC_TEST_ONLY_SOURCE_UID_{j}' for j in range(24)])
        sids = np.array(['SYNTHETIC_TEST_ONLY_SOURCE_A', 'SYNTHETIC_TEST_ONLY_SOURCE_B'])
        np.savez(d/'model.npz', scaler_mean=np.zeros(24480), scaler_scale=np.ones(24480), pca_mean=np.zeros(24480),
                 pca_components=pc, p_mean=np.zeros(16), i_mean=np.zeros(16), B4_P_weights=np.eye(16), B4_I_weights=np.eye(16),
                 source_ids=sids, source_original_uids=suid, modality=np.array('MEG'))
        def run(command, input_name, output_name, *extra, success=True):
            result = subprocess.run([sys.executable, '-B', str(cli), command, '--qualification', str(d/'q.json'),
                   '--input', str(d/input_name), '--output', str(d/output_name), *map(str, extra)], capture_output=True, text=True)
            if success: self.assertEqual(result.returncode, 0, result.stderr)
            else: self.assertNotEqual(result.returncode, 0)
        if not full_direction:
            np.savez(d/'codes.npz', labels=self.codes, original_uid=uid)
            run('plan', 'target.npz', 'plan.npz', '--codes', d/'codes.npz')
            run('four-cell', 'target.npz', 'score.npz', '--model', d/'model.npz', '--plan', d/'plan.npz')
            with np.load(d/'score.npz', allow_pickle=False) as z:
                assert_array_equal(z['labels'], self.codes)
                self.assertEqual(z['decision_evidence'].shape, (2, 24, 2, 4))
                self.assertEqual(str(z['scope']), 'SYNTHETIC_TEST_ONLY')
            run('plan', 'target.npz', 'plan.npz', '--codes', d/'codes.npz', success=False)
        else:
            sy = np.tile(np.repeat([0, 1], 6), 2)
            np.savez(d/'source.npz', P=rawp, I=rawi, y=sy, original_uid=suid, owner=np.repeat(sids, 12))
            run('direction-prior', 'source.npz', 'prior.npz', '--model', d/'model.npz')
            run('direction-grid', 'target.npz', 'grid.npz', '--model', d/'model.npz', '--prior', d/'prior.npz')
            with np.load(d/'grid.npz', allow_pickle=False) as z: grid_ba = z['source_fold_BA']
            np.savez(d/'selection_input.npz', source_person_order=sids, source_person_BA=np.stack([grid_ba, grid_ba]), lambda_grid=LAMBDA_GRID)
            run('select-direction', 'selection_input.npz', 'selection.json')
            run('direction-score', 'target.npz', 'directions.npz', '--model', d/'model.npz', '--prior', d/'prior.npz', '--selection', d/'selection.json')
            with np.load(d/'directions.npz', allow_pickle=False) as z:
                assert_array_equal(z['evaluation_true_labels'], self.true)
                self.assertEqual(z['decision_evidence'].shape, (24, 4))


if __name__ == '__main__':
    with threadpool_limits(2):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'scope': 'SYNTHETIC_TEST_ONLY', 'tests_run': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors),
                      'real_data_loaded': False, 'artifacts_retained_outside_repository': True}))
    sys.exit(0 if result.wasSuccessful() else 1)
