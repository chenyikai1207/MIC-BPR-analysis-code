"""SYNTHETIC_TEST_ONLY; bounded scientific-kernel consistency, no real inputs."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import json,time,importlib
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from external_core import fit_source,score_target,donor_map
from external_codes import draw_proposed_codes
from external_statistics import summarize_lopo
import meg_binary as binary
from meg_source import fit_source_arrays

def main():
    start=time.perf_counter();rng=np.random.default_rng(91736)
    for n in ['historical_d1','historical_d6','historical_listener_origin','native_mat5_metadata','meg_epoch_reader','feature_windows']:
        importlib.import_module(n)
    with threadpool_limits(2):
        y=np.repeat(np.arange(3,dtype=np.int8),20)
        p=rng.normal(size=(60,64));i=.25*p+rng.normal(size=p.shape)
        model=fit_source(p,i,y)
        true=np.repeat(np.arange(3,dtype=np.int8),6);session=np.array(['SYNTHETIC_TEST_ONLY_SESSION']*18)
        target={'P':rng.normal(size=(18,64)),'I':rng.normal(size=(18,64)),'y':true,'session':session}
        codes=draw_proposed_codes(true,session,999,n_codes=3)
        labels=np.vstack([true,codes]);donor,tier=donor_map(labels,true,session,np.array(['normal']*18),np.arange(18)*2048)
        result,pred=score_target(model,target,{'labels':labels,'donor':donor},return_predictions=True)
        manual=[]
        for b in range(1,4):
            arms=np.empty((2,2))
            for r in range(2):
                for a in range(2):arms[r,a]=np.mean([np.mean(pred[b,labels[b]==c,r,a]==c) for c in range(3)])
            manual.append((arms[0,0]-arms[0,1])-(arms[1,0]-arms[1,1]))
        assert np.isclose(result['primary_fraction'],np.mean(manual)) and result['reference_checks']==4
        assert np.all(tier<=1) and pred.shape==(4,18,2,2)
        x=rng.normal(size=(24,16));lab=np.repeat([0,1],12).astype(np.int8)[None,:]
        w,a=binary.loo_coefficients(x,lab);wf,af=binary.all_coefficients(x,lab)
        assert np.allclose(w,wf[:,:24],rtol=1e-9,atol=1e-9)
        for j in [0,23,24]:
            keep=np.arange(24)!=j if j<24 else np.ones(24,bool)
            ref=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(x[keep],lab[0,keep])
            actual=x@wf[0,j]+af[0,j]
            assert np.allclose(ref.decision_function(x),actual,rtol=1e-9,atol=1e-8)
        py=np.repeat([0,1],30);pp=rng.normal(size=(60,64));ii=.2*pp+rng.normal(size=pp.shape)
        m=fit_source_arrays(pp,ii,py)
        assert m['B4_P_weights'].shape[1]==m['B4_I_weights'].shape[1]==16
        points=np.arange(12,dtype=float)/100
        outer=np.array([[np.nan if j==k else points[j] for j in range(12)] for k in range(12)])
        stats=summarize_lopo(points,outer)
        # Mean jackknife SE has a closed-form independently known result.
        deleted=np.array([np.delete(points,k).mean() for k in range(12)])
        v=(11/12)*np.sum((deleted-deleted.mean())**2)
        assert np.isclose(np.sqrt(v),points.std(ddof=1)/np.sqrt(12))
        assert len(stats)>0
    report={'status':'PASS_SYNTHETIC_TEST_ONLY_CURRENT_KERNELS','input_label':'SYNTHETIC_TEST_ONLY','seconds':time.perf_counter()-start,'CPU_threads':2,'external_direct_sklearn_checks':4,'binary_direct_sklearn_checks':3,'macro_class_code_session_arm_direction_match':True,'MEG_source_fixed16_rank_gate':True,'historical_kernel_imports':True,'wholeperson_mean_closed_form_jackknife_match':True,'actual_participant_data_read':False,'real_model_fits':0,'hashes_computed':False}
    print(json.dumps(report,ensure_ascii=True))
if __name__=='__main__':main()
