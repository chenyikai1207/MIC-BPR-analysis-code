"""SYNTHETIC_TEST_ONLY: recovered blocks, restored dependencies and input contracts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='2'
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from meg_source import fit_source_arrays
from meg_binary import donor_map, mismatch
from meg_epoch_reader import _PicklePlaceholder
from analysis_adapters import score_meg_recall, score_meg_manipulation, score_meg_correspondence
from mic_cli import features, validate_model, validate_assignment, external_path, ROOT


def rejected(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError('Malformed input was accepted')


def main():
    started=time.perf_counter();rng=np.random.default_rng(91737)
    with threadpool_limits(2):
        holder=_PicklePlaceholder();holder.__setstate__({'SYNTHETIC_TEST_ONLY':True})
        assert holder.SYNTHETIC_TEST_ONLY
        p=rng.normal(size=(60,64));i=rng.normal(size=(60,64));y=np.repeat([0,1],30)
        m=fit_source_arrays(p,i,y)
        true=np.repeat([0,1],12).astype(np.int8);half=np.tile(np.repeat([0,1],6),2);onset=np.arange(24)*200
        labels=np.vstack([true]+[rng.permutation(true) for _ in range(199)])
        donor,_=donor_map(labels,true,half,onset)
        tp=rng.normal(size=(24,64));ti=rng.normal(size=(24,64))
        recall=score_meg_recall(m,tp,ti,labels,donor)
        evaluation=labels.copy();evaluation[0]=1-true
        manipulation=score_meg_manipulation(m,tp,ti,labels,evaluation,donor)
        assert recall['reference_checks']==manipulation['reference_checks']==6
        # Independent arm comparison: same P fits, complemented real evaluation only.
        assert np.allclose(recall['decision_evidence'][:,:,:,:2],manipulation['decision_evidence'],atol=1e-8,rtol=1e-9)
        assert np.allclose(manipulation['per_label_BA'][0],1-recall['per_label_BA'][0,:,:2])
        assert np.allclose(manipulation['per_label_BA'][1:],recall['per_label_BA'][1:,:,:2])
        # Original mismatch solver uses floating timing costs (infinity on diagonal).
        tt=onset.astype(np.float64)
        near=mismatch(true,half,tt);far=mismatch(true,half,tt,True)
        evidence=recall['decision_evidence'][:,:,:,2]
        result=score_meg_correspondence(true,labels[1:],evidence[0]>0,evidence[1:]>0,near,far)
        assert result['endpoint_values'].shape==(10,) and result['mean_values'].shape==(8,)
        # Scalar class-by-class oracle, independent from batch_ba/vector.
        expected=[]
        for mapping in (np.arange(24),near,far):
            matrix=np.array([[np.mean([np.mean((evidence[b+1,:,r]>0)[labels[b+1,mapping]==c]==c) for c in (0,1)]) for r in (0,1)] for b in range(199)])
            expected.append(matrix)
        for key,oracle in zip(('correct_null_BA','near_null_BA','far_null_BA'),expected):
            assert np.allclose(result[key],oracle,atol=1e-15)
        uid=np.array([f'SYNTHETIC_TEST_ONLY_{j}' for j in range(24)])
        z={'P':np.zeros((24,24480),np.float32),'I':np.ones((24,24480),np.float32),
           'y':true,'original_uid':uid,'owner':np.array(['SYNTHETIC_TEST_ONLY_TARGET']*24),
           'pause_halfblock':half,'onset_log_sec':onset.astype(np.float64)/100}
        assert features(z,'MEG')==24
        assignment={'labels':labels,'donor':donor,'original_uid':uid}
        validate_assignment(assignment,z,'MEG')
        bad=dict(assignment,original_uid=uid[::-1]);rejected(lambda:validate_assignment(bad,z,'MEG'))
        bad=dict(assignment,donor=np.tile(np.arange(24),(200,1)));rejected(lambda:validate_assignment(bad,z,'MEG'))
        rejected(lambda:features(dict(z,P=z['P'].T),'MEG'))
        mm=dict(m,modality=np.array('MEG'),source_ids=np.array(['SYNTHETIC_TEST_ONLY_TARGET']))
        rejected(lambda:validate_model(mm,z,'MEG'))
        rejected(lambda:external_path(ROOT/'unreleased_rows.npz'))
    print(json.dumps({'status':'PASS_SYNTHETIC_TEST_ONLY_INTERFACES','seconds':time.perf_counter()-started,
      'restored_mapping_executed':True,'restored_correspondence_keys_executed':True,
      'MEG_recall_direct_sklearn_checks':6,'MEG_manipulation_direct_sklearn_checks':6,
      'manipulation_true_goal_complement_and_random_labels_verified':True,
      'correspondence_independent_scalar_oracle':True,'negative_contract_checks':5,
      'CPU_threads':2,'actual_participant_data_read':False,'real_model_fits':0}))


if __name__=='__main__':main()
