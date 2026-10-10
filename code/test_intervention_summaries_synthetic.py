"""SYNTHETIC_TEST_ONLY black-box aggregation and qualification rejection tests."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose
from mic_intervention_api import source_target_sensitivity


class SummaryCliTests(unittest.TestCase):
    def setUp(self):
        scratch = os.environ.get('MIC_SYNTHETIC_SCRATCH')
        if scratch: Path(scratch).mkdir(parents=True, exist_ok=True)
        self.d = Path(tempfile.mkdtemp(prefix='MIC_SYNTHETIC_TEST_ONLY_SUMMARY_', dir=scratch))
        self.q = {'status':'QUALIFIED_LOCALLY', 'modality':'MEG', 'scope':'SYNTHETIC_TEST_ONLY',
                  **{k:True for k in ['snapshot_identity_verified','eligibility_verified','event_uid_row_order_verified',
                    'feature_contract_verified','frozen_roles_verified','complete_source_refits_verified','same_target_support_verified']},
                  'local_qualification_evidence':'SYNTHETIC_TEST_ONLY generated fixture, no real data'}
        (self.d/'q.json').write_text(json.dumps(self.q), encoding='utf-8')

    def run_cli(self, command, file='input.npz', out='out.json', success=True):
        r = subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('mic_interventions_cli.py')), command,
             '--qualification',str(self.d/'q.json'),'--input',str(self.d/file),'--output',str(self.d/out)], capture_output=True,text=True)
        if success: self.assertEqual(r.returncode,0,r.stderr)
        else: self.assertNotEqual(r.returncode,0)
        return r

    def test_factorial_summary_preserves_person_pairs_and_code_means(self):
        # Analytic synthetic examples; none are paper result instances.
        ba = np.array([[[[.8,.5,.4,.6]],[[.6,.4,.5,.5]]],
                       [[[.9,.5,.4,.6]],[[.7,.4,.5,.5]]],
                       [[[.8,.5,.5,.6]],[[.6,.4,.6,.5]]]])
        idx = np.array([[0,0,1],[1,1,1],[2,2,2],[0,1,2]])
        np.savez(self.d/'input.npz', per_person_code_BA=ba, participant_row_resamples=idx,
                 person_order=np.array(['SYNTHETIC_TEST_ONLY_T0','SYNTHETIC_TEST_ONLY_T1','SYNTHETIC_TEST_ONLY_T2']),readout_order=np.array(['B4']))
        self.run_cli('aggregate-factorial')
        result=json.loads((self.d/'out.json').read_text())
        j=100*(ba[:,:,0,0]-ba[:,:,0,1]-ba[:,:,0,2]+ba[:,:,0,3]).mean(1)
        ep=result['endpoints']['B4_J_INTERACTION_pp']
        assert_allclose(ep['mean'],j.mean())
        assert_allclose(ep['CI95'],np.quantile(j[idx].mean(1),[.025,.975]))
        self.assertEqual(result['data_scope'],'SYNTHETIC_TEST_ONLY')
        self.assertTrue(result['equal_person_weighting'])

    def test_source_summary_subtracts_paired_endpoints_before_variance(self):
        full=np.array([[9.,7.],[12.,10.],[8.,6.]])
        deleted=np.tile(full,(3,1,1))
        deleted[:,:,0]+=np.array([-.2,.1,.4])[:,None]
        np.savez(self.d/'input.npz', full_endpoint_pairs_pp=full,deleted_endpoint_pairs_pp=deleted,
                 source_order=np.array(['SYNTHETIC_TEST_ONLY_S0','SYNTHETIC_TEST_ONLY_S1','SYNTHETIC_TEST_ONLY_S2']),
                 target_order=np.array(['SYNTHETIC_TEST_ONLY_T0','SYNTHETIC_TEST_ONLY_T1','SYNTHETIC_TEST_ONLY_T2']))
        self.run_cli('source-sensitivity')
        result=json.loads((self.d/'out.json').read_text())
        ref=source_target_sensitivity(full[:,0]-full[:,1],deleted[:,:,0]-deleted[:,:,1])
        assert_allclose(result['approx_source_plus_target_CI95_pp'],ref['approx_source_plus_target_CI95_pp'])
        self.assertFalse(result['independent_replication'])

    def test_real_qualification_cannot_be_inferred_from_synthetic_label(self):
        self.q['scope']='USER_SUPPLIED_ADMITTED_DATA'
        self.q['data_access_legal_and_license_verified']=False
        (self.d/'q-real.json').write_text(json.dumps(self.q), encoding='utf-8')
        cli=Path(__file__).with_name('mic_interventions_cli.py')
        r=subprocess.run([sys.executable,'-B',str(cli),'select-direction','--qualification',str(self.d/'q-real.json'),
                          '--input',str(self.d/'missing.npz'),'--output',str(self.d/'not-written.json')],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('Identity/license/legality',r.stderr)
        self.assertFalse((self.d/'not-written.json').exists())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SummaryCliTests))
    print(json.dumps({'scope':'SYNTHETIC_TEST_ONLY','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'real_data_loaded':False,'no_files_deleted':True}))
    sys.exit(0 if result.wasSuccessful() else 1)
