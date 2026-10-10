"""Current binary MEG readout functions, copied verbatim from actual producers.
No data paths, integrity gate, file writes or model execution on import.
"""
import numpy as np
from sklearn.preprocessing._data import _is_constant_feature
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import t
from scipy.optimize import linear_sum_assignment

def projection(x,m):
    return (((x.astype(np.float64)-m['scaler_mean'])/m['scaler_scale'])-m['pca_mean'])@m['pca_components']

def loo_coefficients(x,labels):
    """Each label row gets N freshly refitted N-1 models; no I input.

    Covariance follows StandardScaler -> LedoitWolf -> rescale exactly.
    Repeated arm training sets reuse the same newly refitted deletion model.
    """
    x=np.asarray(x,dtype=np.float64);labels=np.asarray(labels,dtype=np.int8)
    if labels.ndim==1: labels=labels[None,:]
    batch,n,d=labels.shape[0],*x.shape
    assert d==16 and np.all(np.isin(labels,(0,1)))
    include=~np.eye(n,dtype=bool)
    pooled=np.zeros((batch,n,d,d),np.float64)
    means=[];counts=[]
    for cls in (0,1):
        mask=(labels[:,None,:]==cls)&include[None,:,:]
        count=mask.sum(-1);assert count.min()>=2
        mean=np.einsum('bkn,nd->bkd',mask,x)/count[...,None]
        centered=(x[None,None,:,:]-mean[:,:,None,:])*mask[...,None]
        var=np.einsum('bknd,bknd->bkd',centered,centered)/count[...,None]
        constant=_is_constant_feature(var,mean,count[...,None])
        scale=np.sqrt(var);scale[constant]=1.
        standardized=centered/scale[:,:,None,:]
        # sklearn LedoitWolf centers the standardized points a second time.
        standardized-=(standardized.sum(2)/count[...,None])[:,:,None,:]*mask[...,None]
        emp=np.einsum('bkni,bknj->bkij',standardized,standardized)/count[...,None,None]
        trace=np.trace(emp,axis1=-2,axis2=-1);mu=trace/d
        delta_sum=np.einsum('bkij,bkij->bk',emp,emp)
        squared_norm=np.einsum('bknd,bknd->bkn',standardized,standardized)
        beta_sum=np.einsum('bkn,bkn->bk',squared_norm,squared_norm)
        beta=(beta_sum/count-delta_sum)/(d*count)
        delta=(delta_sum-2*mu*trace+d*mu**2)/d
        beta=np.minimum(beta,delta)
        shrink=np.divide(beta,delta,out=np.zeros_like(beta),where=beta!=0)
        cov=emp*(1-shrink[...,None,None])
        diagonal=np.arange(d);cov[:,:,diagonal,diagonal]+=shrink[...,None]*mu[...,None]
        cov*=scale[...,None]*scale[...,None,:]
        pooled+=cov*count[...,None,None]/(n-1)
        means.append(mean);counts.append(count)
    means=np.stack(means,axis=-2)
    # Well conditioned auto-shrinkage matrices: same solution as lsqr's lstsq.
    coef=np.linalg.solve(pooled,np.swapaxes(means,-1,-2))
    coef=np.swapaxes(coef,-1,-2)
    intercept=-.5*np.einsum('bkcd,bkcd->bkc',means,coef)+np.log(np.stack(counts,axis=-1)/(n-1))
    w=coef[:,:,1,:]-coef[:,:,0,:];a=intercept[:,:,1]-intercept[:,:,0]
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(a))
    return w,a

def donor_map(labels,true,half,onset):
    """Predefined deterministic same-supervision-label nearest donor.

    Order: same true melody/task/halfblock; same true melody/task;
    same task/halfblock; same task. All tasks here are recall.
    Within the first supported tier minimize absolute onset displacement,
    then original row index. With replacement across j is allowed here:
    each j indexes a separate input-deletion experiment, not label shuffling.
    """
    labels=np.asarray(labels);n=len(true);out=np.empty_like(labels,dtype=np.int16)
    tier=np.empty_like(labels,dtype=np.int8)
    for b,y in enumerate(labels):
        for j in range(n):
            valid=(y==y[j])&(np.arange(n)!=j)
            for level,restriction in enumerate(((true==true[j])&(half==half[j]),true==true[j],half==half[j],np.ones(n,bool))):
                cand=np.flatnonzero(valid&restriction)
                if len(cand):
                    k=cand[np.lexsort((cand,np.abs(onset[cand]-onset[j])))[0]]
                    out[b,j]=k;tier[b,j]=level;break
            else: raise AssertionError('No same-supervision-class donor; freeze requires all originals supported')
    assert np.all(out!=np.arange(n)[None,:])
    assert np.array_equal(labels,np.take_along_axis(labels,out,axis=1))
    return out,tier

def ba_many(label,pred):
    # label (batch,N); pred (batch,N,arms).
    return .5*np.stack([np.mean(pred[q,label[q]==0,:]==0,axis=0)+np.mean(pred[q,label[q]==1,:]==1,axis=0) for q in range(len(label))])

def all_coefficients(x,labels):
    """Fresh N deleted-row LDA fits + full-N fit, same auto shrinkage."""
    x=np.asarray(x,np.float64);labels=np.asarray(labels,np.int8);batch,n,d=labels.shape[0],*x.shape
    include=np.vstack([~np.eye(n,dtype=bool),np.ones((1,n),bool)]);folds=n+1;amount=include.sum(-1)
    pooled=np.zeros((batch,folds,d,d));means=[];counts=[]
    for cls in (0,1):
        mask=(labels[:,None,:]==cls)&include[None,:,:];count=mask.sum(-1);assert count.min()>=2
        mean=np.einsum('bkn,nd->bkd',mask,x)/count[...,None]
        centered=(x[None,None,:,:]-mean[:,:,None,:])*mask[...,None]
        var=np.einsum('bknd,bknd->bkd',centered,centered)/count[...,None]
        scale=np.sqrt(var);scale[_is_constant_feature(var,mean,count[...,None])]=1.
        standardized=centered/scale[:,:,None,:]
        standardized-=(standardized.sum(2)/count[...,None])[:,:,None,:]*mask[...,None]
        emp=np.einsum('bkni,bknj->bkij',standardized,standardized)/count[...,None,None]
        trace=np.trace(emp,axis1=-2,axis2=-1);mu=trace/d;ds=np.einsum('bkij,bkij->bk',emp,emp)
        sq=np.einsum('bknd,bknd->bkn',standardized,standardized);bs=np.einsum('bkn,bkn->bk',sq,sq)
        beta=(bs/count-ds)/(d*count);delta=(ds-2*mu*trace+d*mu**2)/d;beta=np.minimum(beta,delta)
        shrink=np.divide(beta,delta,out=np.zeros_like(beta),where=beta!=0)
        cov=emp*(1-shrink[...,None,None]);di=np.arange(d);cov[:,:,di,di]+=shrink[...,None]*mu[...,None]
        cov*=scale[...,None]*scale[...,None,:];pooled+=cov*count[...,None,None]/amount[None,:,None,None]
        means.append(mean);counts.append(count)
    means=np.stack(means,axis=-2);coef=np.swapaxes(np.linalg.solve(pooled,np.swapaxes(means,-1,-2)),-1,-2)
    intercept=-.5*np.einsum('bkcd,bkcd->bkc',means,coef)+np.log(np.stack(counts,axis=-1)/amount[None,:,None])
    return coef[:,:,1,:]-coef[:,:,0,:],intercept[:,:,1]-intercept[:,:,0]

def fit_predict(p,i,y):
    model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(p,y)
    ev=model.decision_function(i)
    return ev,(ev>0).astype(np.int8)

def infer(full,deleted,n=35):
    mean=full.mean(0);vt=full.var(axis=0,ddof=1)/n;vs=4/5*((deleted-deleted.mean(0))**2).sum(0)
    se=np.sqrt(vt+vs);cse=np.sqrt(vt);q=t.ppf(.975,4);cq=t.ppf(.975,n-1)
    p=lambda m,s,df:float(t.sf(m/s,df)) if s>0 else (0. if m>0 else 1.)
    return [{'mean_BA':float(mean[j]),'conditional_ci95_low':float(mean[j]-cq*cse[j]),'conditional_ci95_high':float(mean[j]+cq*cse[j]),
             'conditional_p_one_sided_t':p(mean[j],cse[j],n-1),'source_target_approx_ci95_low':float(mean[j]-q*se[j]),
             'source_target_approx_ci95_high':float(mean[j]+q*se[j]),'source_target_approx_p_t4':p(mean[j],se[j],4),
             'variance_target':float(vt[j]),'variance_source_group_jackknife':float(vs[j]),'target_N':n,'source_blocks':5} for j in range(len(mean))]

def mismatch(y,half,tt,far=False):
    out=np.arange(len(y))
    for h in (0,1):
        for c in (0,1):
            ix=np.flatnonzero((half==h)&(y==c));assert len(ix)>=2,'No alternate grouping/fallback allowed.'
            d=np.abs(tt[ix,None]-tt[ix][None,:]);cost=-d if far else d;np.fill_diagonal(cost,np.inf)
            rr,cc=linear_sum_assignment(cost);assert np.array_equal(rr,np.arange(len(ix)));out[ix]=ix[cc]
    assert len(np.unique(out))==len(y) and not np.any(out==np.arange(len(y)))
    assert np.array_equal(y,y[out]) and np.array_equal(half,half[out]);return out

def batch_ba(labels,pred):
    lab=labels.astype(bool);zero=~lab
    assert np.all(zero.sum(1)>0) and np.all(lab.sum(1)>0)
    return .5*((~pred&zero[:,:,None]).sum(1)/zero.sum(1)[:,None]+(pred&lab[:,:,None]).sum(1)/lab.sum(1)[:,None])

def vector(true,correct,near,far):
    rd=true[0]-true[1];cd=correct[:,0]-correct[:,1];nd=near[:,0]-near[:,1];fd=far[:,0]-far[:,1]
    v=[rd-cd.mean(),true[0]-correct[:,0].mean(),(correct[:,0]-near[:,0]).mean(),(cd-nd).mean(),
       (correct[:,0]-far[:,0]).mean(),(cd-fd).mean(),(near[:,0]-far[:,0]).mean(),(nd-fd).mean(),nd.mean(),fd.mean()]
    means=[*true,*correct.mean(0),*near.mean(0),*far.mean(0)]
    return dict(zip(KEYS,v)),dict(zip(MEANKEYS,means))

# Restored verbatim producer constants required by vector().
KEYS=['TRUE_INCREMENT_MINUS_SYNCHRONIZED_NULL_INCREMENT','TRUE_B4_MINUS_SYNCHRONIZED_NULL_B4',
      'NULL_B4_CORRECT_MINUS_NEAR','NULL_INCREMENT_CORRECT_MINUS_NEAR',
      'NULL_B4_CORRECT_MINUS_FAR','NULL_INCREMENT_CORRECT_MINUS_FAR',
      'NULL_B4_NEAR_MINUS_FAR','NULL_INCREMENT_NEAR_MINUS_FAR',
      'NULL_NEAR_INCREMENT','NULL_FAR_INCREMENT']

MEANKEYS=['real_B4_BA','real_PCA16_BA','correct_null_B4_BA','correct_null_PCA16_BA','near_null_B4_BA','near_null_PCA16_BA','far_null_B4_BA','far_null_PCA16_BA']
