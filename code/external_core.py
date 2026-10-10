"""Actual frozen external EEG scientific core; obsolete unsupported coding removed.
Functions are verbatim; no paths, raw data, hashes or run-on-import behavior.
"""
import numpy as np
from scipy.linalg import lstsq,eigh,cho_factor,cho_solve
from scipy.signal import detrend,resample_poly
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing._data import _is_constant_feature
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

CLASSES=('flower','penguin','guitar')

HELMERT=np.array([[1/np.sqrt(2),1/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(6)],[0,-2/np.sqrt(6)]])

def remove_line_band_own_window(x):
    """Frozen v2: project out constant, trend and 49-51Hz Fourier quadratures.

    Metadata specifies 50Hz. Use the FIVE discrete 2s Fourier frequencies
    49,49.5,50,50.5,51Hz, the predeclared diagnostic band. This removes them within each own
    2s window without using adjacent P/I samples or fitting across windows.
    It does NOT qualify ocular/muscle cleaning or all off-grid line leakage.
    """
    x=np.asarray(x,np.float64);assert x.shape==(124,2048) and np.isfinite(x).all()
    t=np.arange(2048)/1024
    frequencies=np.arange(49,51.01,.5)
    design=np.column_stack([np.ones(2048),t-t.mean()]+[fun(2*np.pi*f*t) for f in frequencies for fun in (np.sin,np.cos)])
    q=np.linalg.qr(design,mode='reduced')[0]
    return x-(x@q)@q.T

def phase_local_v2(x):
    return resample_poly(remove_line_band_own_window(x),5,128,axis=1,padtype='line').reshape(-1).astype(np.float32)

def loo_coefficients(x,labels):
    """All N independent N-1 LDA lsqr/auto fits, THREE coefficient rows.

    Caller limits label batches to 8. Each row is freshly refitted; classwise
    scaling/Ledoit-Wolf match sklearn1.9.1. Rank-deficient cases use its lstsq.
    """
    x=np.asarray(x,np.float64);labels=np.asarray(labels,np.int8)
    if labels.ndim==1:labels=labels[None,:]
    batch,n=labels.shape;d=x.shape[1]
    assert x.shape==(n,16) and batch<=8 and np.all(np.isin(labels,(0,1,2)))
    include=~np.eye(n,dtype=bool);pooled=np.zeros((batch,n,d,d));means=[];counts=[]
    for cls in range(3):
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
        cov*=scale[...,None]*scale[...,None,:];pooled+=cov*count[...,None,None]/(n-1)
        means.append(mean);counts.append(count)
    means=np.stack(means,axis=-2);coef=np.empty_like(means)
    # Use sklearn's scipy.linalg.lstsq driver on every small matrix. A rank
    # decision based on hand-picked condition thresholds would be unnecessary.
    condition=np.linalg.cond(pooled)
    direct_intercepts={}
    for b in range(batch):
        for j in range(n):
            if condition[b,j]>1e8:
                # Numerical qualification, not an efficacy tuning parameter:
                # tiny covariance summation differences are amplified in an
                # ill-conditioned system. Reuse the declared sklearn fit.
                keep=include[j]
                ref=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(x[keep],labels[b,keep])
                coef[b,j]=ref.coef_;direct_intercepts[b,j]=ref.intercept_
            else:coef[b,j]=lstsq(pooled[b,j],means[b,j].T,check_finite=False)[0].T
    intercept=-.5*np.einsum('bkcd,bkcd->bkc',means,coef)+np.log(np.stack(counts,axis=-1)/(n-1))
    for (b,j),value in direct_intercepts.items():intercept[b,j]=value
    assert np.isfinite(coef).all() and np.isfinite(intercept).all()
    return coef,intercept

def donor_map(labels,true,session,pitch,onset):
    """Same arbitrary/real code AND true category, within the SAME person.

    Tiers: same session+pitch; same session; same person. Nearest absolute
    native time within a session, then row index; cross-session tier uses
    session token and original row (onsets across sessions not comparable).
    """
    labels=np.asarray(labels);true=np.asarray(true);session=np.asarray(session);pitch=np.asarray(pitch);onset=np.asarray(onset)
    n=len(true);out=np.empty(labels.shape,np.int32);tier=np.empty(labels.shape,np.int8)
    for b,y in enumerate(labels):
        for j in range(n):
            valid=(y==y[j])&(true==true[j])&(np.arange(n)!=j)
            restrictions=((session==session[j])&(pitch==pitch[j]),session==session[j],np.ones(n,bool))
            for level,r in enumerate(restrictions):
                cand=np.flatnonzero(valid&r)
                if len(cand):
                    order=(np.lexsort((cand,np.abs(onset[cand]-onset[j]))) if level<2 else np.lexsort((cand,session[cand])))
                    out[b,j]=cand[order[0]];tier[b,j]=level;break
            else:raise ValueError('No donor within true category/code/person; stop before scoring')
    assert np.all(out!=np.arange(n)[None,:]) and np.array_equal(labels,np.take_along_axis(labels,out,axis=1))
    assert np.array_equal(np.tile(true,(len(labels),1)),true[out])
    return out,tier

def person_primary(labels,pred,session):
    """SYNTHETIC-testable aggregator; labels QxN, pred QxNx2reprx2arms.

    Class macro-average within session, sessions equal, codes equal; returns
    ONE person DoD (fraction, multiply100 for percentage points).
    """
    labels=np.asarray(labels);pred=np.asarray(pred);session=np.asarray(session)
    assert pred.shape==(*labels.shape,2,2)
    session_values=[]
    for s in np.unique(session):
        keep=session==s;labs=labels[:,keep];p=pred[:,keep]
        ba=[]
        for c in range(3):
            hit=labs==c;den=hit.sum(1);assert den.min()>0,'Each code/session must support all classes'
            ba.append(((p==c)*hit[...,None,None]).sum(1)/den[:,None,None])
        mean=np.mean(ba,axis=0);delta=mean[:,:,0]-mean[:,:,1]
        session_values.append(delta[:,0]-delta[:,1])
    return float(np.mean(session_values))

def fit_source(p,i,y):
    """Fixed 16D source maps: 2 Helmert columns + 14 paired-common PCs."""
    p=np.asarray(p,np.float64);i=np.asarray(i,np.float64);y=np.asarray(y)
    assert p.shape==i.shape and np.all(np.isin(y,(0,1,2))) and set(y)=={0,1,2}
    n=len(y);x=np.vstack([p,i]);sc=StandardScaler(copy=False).fit(x);sc.transform(x,copy=False)
    pm=x.mean(0);x-=pm;gram=x@x.T/(len(x)-1)
    value,vec=eigh(gram,check_finite=False,overwrite_a=True,driver='evd')
    value=np.maximum(value[::-1],0);vec=vec[:,::-1];rank=int(np.searchsorted(np.cumsum(value)/value.sum(),.95)+1)
    assert rank>=16 and value[rank-1]>value[0]*1e-12,'Insufficient fixed 16D support'
    components=x.T@vec[:,:rank]/np.sqrt(value[:rank]*(len(x)-1))[None,:]
    for j in range(rank):
        if components[np.argmax(np.abs(components[:,j])),j]<0:components[:,j]*=-1
    z=x@components;pp=z[:n];ii=z[n:];pmean=pp.mean(0);imean=ii.mean(0);pc=pp-pmean;ic=ii-imean
    common=(pc+ic)/2;common-=common.mean(0)
    val,u=eigh(common.T@common,subset_by_index=[rank-14,rank-1],check_finite=False,driver='evr')
    val=val[::-1];u=u[:,::-1];assert val[-1]>val[0]*1e-10
    for j in range(14):
        if u[np.argmax(np.abs(u[:,j])),j]<0:u[:,j]*=-1
    h=np.column_stack([HELMERT[y],common@u]);wp,wi=paired_weights(pc,ic,h,y)
    assert np.linalg.matrix_rank(wp)==np.linalg.matrix_rank(wi)==16
    return {'scaler_mean':sc.mean_,'scaler_scale':sc.scale_,'pca_mean':pm,'pca_components':components,
      'p_mean':pmean,'i_mean':imean,'B4_P_weights':wp,'B4_I_weights':wi,'PCA_dimension':rank,
      'source_original_N':n}

def paired_weights(pc,ic,h,y):
    """Exact minimizer of P/I ridge fits plus class-centroid agreement."""
    r=pc.shape[1];pf=cho_factor(pc.T@pc+.01*np.eye(r),lower=True,check_finite=False)
    inf=cho_factor(ic.T@ic+.01*np.eye(r),lower=True,check_finite=False)
    base=np.vstack([cho_solve(pf,pc.T@h,check_finite=False),cho_solve(inf,ic.T@h,check_finite=False)])
    cent=np.array([np.r_[pc[y==c].mean(0),-ic[y==c].mean(0)] for c in range(3)])
    inv=np.vstack([cho_solve(pf,cent[:,:r].T,check_finite=False),cho_solve(inf,cent[:,r:].T,check_finite=False)])
    weights=base-inv@np.linalg.solve(np.eye(3)/.1+cent@inv,cent@base)
    return weights[:r],weights[r:]

def project(x,m):return (((np.asarray(x,np.float64)-m['scaler_mean'])/m['scaler_scale'])-m['pca_mean'])@m['pca_components']

def balanced_session_means(labels,pred,session):
 result=[]
 for s in np.unique(session):
  keep=session==s;lab=labels[:,keep];pp=pred[:,keep];ba=[]
  for c in range(3):
   hit=lab==c;count=hit.sum(1);assert count.min()>0
   ba.append(((pp==c)*hit[:,:,None,None]).sum(1)/count[:,None,None])
  result.append(np.mean(ba,axis=0))
 return np.mean(result,axis=0)

def score_target(model,person,assignment,budget=lambda:None,return_predictions=False):
 p=project(person['P'],model);i=project(person['I'],model)
 labels=assignment['labels'];donor=assignment['donor'];n=len(person['y']);bcount=len(labels)
 assert labels.shape==donor.shape==(bcount,n) and np.all(donor!=np.arange(n)[None,:])
 assert np.array_equal(labels,np.take_along_axis(labels,donor,axis=1))
 reps=[((p-model['p_mean'])@model['B4_P_weights'],(i-model['i_mean'])@model['B4_I_weights']),(p[:,:16],i[:,:16])]
 pred=np.empty((bcount,n,2,2),np.int8);maxerr=0.;checks=0
 for r,(xp,xi) in enumerate(reps):
  for lo in range(0,bcount,8):
   hi=min(lo+8,bcount);w,a=loo_coefficients(xp,labels[lo:hi]);q=np.arange(hi-lo)[:,None]
   included=np.einsum('bncd,nd->bnc',w[q,donor[lo:hi]],xi)+a[q,donor[lo:hi]]
   excluded=np.einsum('bncd,nd->bnc',w,xi)+a
   pred[lo:hi,:,r,0]=np.argmax(included,axis=-1);pred[lo:hi,:,r,1]=np.argmax(excluded,axis=-1)
   # Independent direct sklearn check for first and last code/deletion.
   for b,j in ((0,0),(bcount-1,n-1)):
    if lo<=b<hi:
     keep=np.arange(n)!=j;ref=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(xp[keep],labels[b,keep])
     expected=ref.decision_function(xi);actual=xi@w[b-lo,j].T+a[b-lo,j]
     err=float(np.max(np.abs(expected-actual)/(1+np.abs(expected))));maxerr=max(maxerr,err);checks+=1
     assert np.allclose(expected,actual,atol=1e-8,rtol=1e-8) and np.array_equal(ref.predict(xi),np.argmax(actual,axis=1))
   budget();del w,a,included,excluded
 ba=balanced_session_means(labels,pred,person['session']);delta=ba[:,:,0]-ba[:,:,1]
 primary=person_primary(labels[1:],pred[1:],person['session'])
 assert np.isclose(primary,np.mean(delta[1:,0]-delta[1:,1]))
 out={'primary_fraction':primary,'real_B4_inclusion_fraction':float(delta[0,0]),'real_PCA16_inclusion_fraction':float(delta[0,1]),
    'real_DoD_fraction':float(delta[0,0]-delta[0,1]),'arbitrary_B4_inclusion_fraction':float(delta[1:,0].mean()),
    'arbitrary_PCA16_inclusion_fraction':float(delta[1:,1].mean()),'reference_checks':checks,'maximum_relative_decision_error':maxerr,
    'N':n,'sessions':list(np.unique(person['session'])),'readout_order':['B4','PCA16'],'arm_order':['includeOwnP','excludeOwnP']}
 return (out,pred) if return_predictions else out
