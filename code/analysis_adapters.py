"""Explicit-array adapters around unchanged current producer numerical blocks.
These are qualified-feature entry points, not raw archive qualification.
No experiment runs on import. All generated outcomes stay local.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from meg_binary import projection, all_coefficients, loo_coefficients, ba_many, batch_ba, vector

NKEYS=['REAL_B4_INCLUDE_MINUS_EXCLUDE','REAL_PCA16_INCLUDE_MINUS_EXCLUDE','REAL_B4_MINUS_PCA_DIFFERENCE_OF_DIFFERENCES',
 'RANDOM_B4_INCLUDE_MINUS_EXCLUDE','RANDOM_PCA16_INCLUDE_MINUS_EXCLUDE','RANDOM_B4_MINUS_PCA_DIFFERENCE_OF_DIFFERENCES']

EKEYS=['PRIMARY_FIXED_EVAL','PRIMARY_COUPLED_SPECIFICITY','PRIMARY_ABSOLUTE_LABEL_SPECIFICITY','REAL_INCREMENT_B4_MINUS_PCA16',
 'NULL_FIXED_INCREMENT','NULL_COUPLED_INCREMENT','NULL_COUPLED_B4_MINUS_CHANCE','REAL_B4_MINUS_CHANCE','REAL_PCA16_MINUS_CHANCE']

def score_meg_recall(m,p,i,labels,donor,budget=lambda:None):
    true=labels[0];fitcount=0;checks=0;maxerr=0.
    pp=projection(p,m);ii=projection(i,m)
    reps=[((pp-m['p_mean'])@m['B4_P_weights'],(ii-m['i_mean'])@m['B4_I_weights']),(pp[:,:16],ii[:,:16])]
    ev=np.empty((200,len(true),2,3));scores=np.empty((200,2,3));fixed=np.empty((200,2))
    for model,(xp,xi) in enumerate(reps):
        for lo in range(0,200,10):
            hi=min(lo+10,200);w,a=all_coefficients(xp,labels[lo:hi]);fitcount+=(hi-lo)*(len(true)+1)
            q=np.arange(hi-lo)[:,None];index=donor[lo:hi]
            ev[lo:hi,:,model,0]=np.einsum('bnd,nd->bn',w[q,index],xi)+a[q,index]
            ev[lo:hi,:,model,1]=np.einsum('bnd,nd->bn',w[:,:-1],xi)+a[:,:-1]
            ev[lo:hi,:,model,2]=np.einsum('bd,nd->bn',w[:,-1],xi)+a[:,-1,None]
            scores[lo:hi,model,:]=ba_many(labels[lo:hi],ev[lo:hi,:,model,:]>0)
            fixed[lo:hi,model]=ba_many(np.tile(true,(hi-lo,1)),(ev[lo:hi,:,model,2]>0)[...,None])[:,0]
            for qcheck,deleted in ((0,0),(1,len(true)),(199,len(true)-1)):
                if lo<=qcheck<hi:
                    keep=np.ones(len(true),bool)
                    if deleted<len(true):keep[deleted]=False
                    reference=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(xp[keep],labels[qcheck,keep])
                    expected=reference.decision_function(xi);actual=xi@w[qcheck-lo,deleted]+a[qcheck-lo,deleted]
                    err=float(np.max(np.abs(expected-actual)));maxerr=max(maxerr,err);checks+=1
                    assert np.allclose(expected,actual,rtol=1e-9,atol=1e-8) and np.array_equal(expected>0,actual>0)
            del w,a;budget()
    delta=scores[:,:,0]-scores[:,:,1];r=delta[0];z=delta[1:].mean(0)
    nvalues=[r[0],r[1],r[0]-r[1],z[0],z[1],z[0]-z[1]]
    real=scores[0,:,2];null=scores[1:,:,2].mean(0);f=fixed[1:].mean(0);rd=real[0]-real[1];cd=null[0]-null[1];fd=f[0]-f[1]
    evalues=[rd-fd,rd-cd,real[0]-null[0],rd,fd,cd,null[0]-.5,real[0]-.5,real[1]-.5]
    return {'decision_evidence':ev,'per_label_BA':scores,
            'fixed_true_evaluation_full_cal_BA':fixed,
            'endpoint_names':np.array(NKEYS+EKEYS),
            'endpoint_values':np.array(nvalues+evalues),
            'reference_checks':np.array(checks),'max_reference_error':np.array(maxerr)}

def score_meg_manipulation(m,p,i,labels,evaluation,donor,budget=lambda:None):
    assert np.array_equal(evaluation[0],1-labels[0])
    assert np.array_equal(evaluation[1:],labels[1:])
    fitcount=0;checks=0;maxerr=0.
    pp=projection(p,m);ii=projection(i,m);reps=[((pp-m['p_mean'])@m['B4_P_weights'],(ii-m['i_mean'])@m['B4_I_weights']),(pp[:,:16],ii[:,:16])]
    evidence=np.empty((200,len(p),2,2));scores=np.empty((200,2,2))
    for model,(xp,xi) in enumerate(reps):
        for lo in range(0,200,10):
            hi=min(lo+10,200);w,a=loo_coefficients(xp,labels[lo:hi]);fitcount+=(hi-lo)*len(p)
            q=np.arange(hi-lo)[:,None];idx=donor[lo:hi]
            evidence[lo:hi,:,model,0]=np.einsum('bnd,nd->bn',w[q,idx],xi)+a[q,idx]
            evidence[lo:hi,:,model,1]=np.einsum('bnd,nd->bn',w,xi)+a
            scores[lo:hi,model,:]=ba_many(evaluation[lo:hi],evidence[lo:hi,:,model,:]>0)
            for l,j in ((0,0),(1,len(p)//2),(199,len(p)-1)):
                if lo<=l<hi:
                    keep=np.arange(len(p))!=j;ref=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(xp[keep],labels[l,keep])
                    expected=ref.decision_function(xi);actual=xi@w[l-lo,j]+a[l-lo,j];error=float(np.max(np.abs(expected-actual)));maxerr=max(maxerr,error);checks+=1
                    assert np.allclose(expected,actual,rtol=1e-9,atol=1e-8) and np.array_equal(expected>0,actual>0)
            del w,a;budget()
    d=scores[:,:,0]-scores[:,:,1];r=d[0];z=d[1:].mean(0);values=[r[0],r[1],r[0]-r[1],z[0],z[1],z[0]-z[1]]
    return {'decision_evidence':evidence,'per_label_BA':scores,
            'endpoint_names':np.array(NKEYS),'endpoint_values':np.array(values),
            'reference_checks':np.array(checks),'max_reference_error':np.array(maxerr)}

def score_meg_correspondence(y,null,realpred,pred,near,far):
    # Original run_E4.py score-only block; caller retains original UID order.
    correct=batch_ba(null,pred);ns=batch_ba(null[:,near],pred);fs=batch_ba(null[:,far],pred);true=batch_ba(y[None,:],realpred[None,:,:])[0]
    assert np.max(np.abs(true-batch_ba(y[near][None,:],realpred[None,:,:])[0]))==0 and np.max(np.abs(true-batch_ba(y[far][None,:],realpred[None,:,:])[0]))==0
    values,means=vector(true,correct,ns,fs)
    return {'endpoint_names':np.array(list(values)), 'endpoint_values':np.array(list(values.values())),
            'mean_names':np.array(list(means)), 'mean_values':np.array(list(means.values())),
            'true_BA':true,'correct_null_BA':correct,'near_null_BA':ns,'far_null_BA':fs}
