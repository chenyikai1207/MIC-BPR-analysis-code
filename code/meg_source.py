"""Current MEG complete-source numerical fit, with array-based IO adapter.
The StandardScaler/dual-PCA/B4 numerical block is verbatim; historical cache
loading and saved-model gates are replaced by explicit input arrays.
"""
import gc
import numpy as np
from scipy.linalg import eigh,cho_factor,cho_solve
from sklearn.preprocessing import StandardScaler

def fit_source_arrays(p,i,y,source_ids=(),original_uids=None,budget=lambda:None):
    p=np.asarray(p,np.float64);i=np.asarray(i,np.float64);y=np.asarray(y,int)
    assert p.shape==i.shape and len(p)==len(y) and set(y)=={0,1}
    n=len(y);x=np.vstack([p,i]);yy=y.tolist();ids=list(source_ids)
    uu=np.arange(n) if original_uids is None else np.asarray(original_uids)
    sc=StandardScaler(copy=False);sc.fit(x);sc.transform(x,copy=False);pm=x.mean(0);x-=pm
    gram=x@x.T;gram/=len(x)-1
    value,vec=eigh(gram,check_finite=False,overwrite_a=True,driver='evd');del gram
    value=np.maximum(value[::-1],0);vec=vec[:,::-1];rank=int(np.searchsorted(np.cumsum(value)/value.sum(),.95)+1);assert rank>31
    components=(x.T@vec[:,:rank])/np.sqrt(value[:rank]*(len(x)-1))[None,:]
    for j in range(rank):
        if components[np.argmax(np.abs(components[:,j])),j]<0:components[:,j]*=-1
    z=x@components;del x,value,vec;gc.collect();budget()
    p=z[:n];i=z[n:];y=np.array(yy,int);pmean=p.mean(0);imean=i.mean(0);pc=p-pmean;ic=i-imean
    common=(pc+ic)*.5;common-=common.mean(0)
    val,u=eigh(common.T@common,subset_by_index=[rank-31,rank-1],check_finite=False,driver='evr');val=val[::-1];u=u[:,::-1]
    assert val[-1]>val[0]*1e-10
    for j in range(31):
        if u[np.argmax(np.abs(u[:,j])),j]<0:u[:,j]*=-1
    target=np.column_stack([np.where(y==1,1.,-1.),(common@u)[:,:15]])
    pf=cho_factor(pc.T@pc+.01*np.eye(rank),lower=True,check_finite=False);inf=cho_factor(ic.T@ic+.01*np.eye(rank),lower=True,check_finite=False)
    base=np.vstack([cho_solve(pf,pc.T@target,check_finite=False),cho_solve(inf,ic.T@target,check_finite=False)])
    cent=np.array([np.concatenate([pc[y==c].mean(0),-ic[y==c].mean(0)]) for c in (0,1)])
    inv=np.vstack([cho_solve(pf,cent[:,:rank].T,check_finite=False),cho_solve(inf,cent[:,rank:].T,check_finite=False)])
    weights=base-inv@np.linalg.solve(np.eye(2)/.1+cent@inv,cent@base)
    assert np.linalg.matrix_rank(weights[:rank])==np.linalg.matrix_rank(weights[rank:])==16
    m={'scaler_mean':sc.mean_,'scaler_scale':sc.scale_,'pca_mean':pm,'pca_components':components,'p_mean':pmean,'i_mean':imean,
       'B4_P_weights':weights[:rank],'B4_I_weights':weights[rank:],'source_ids':np.array(ids),'source_original_uids':np.array(uu)}
    return m
