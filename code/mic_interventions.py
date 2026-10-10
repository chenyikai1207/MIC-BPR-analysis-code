"""Portable MIC interventions on qualified arrays; no IO or fitting on import.

The two batched label-intervention kernels below are extracted unchanged from
the executed implementation. Singular covariance raises an error; no rescue
model is substituted. Numerical profile is requirements-analysis.txt.
"""
import numpy as np
from scipy.stats import t
from sklearn.preprocessing._data import _is_constant_feature
from meg_binary import loo_coefficients, projection

ARMS = ('OWN_C', 'OTHER_C', 'OWN_X', 'OTHER_X')
LAMBDA_GRID = np.array([0., .25, .5, .75, 1.])

def masked_binary_lda(x,labels,deleted):
    """Exact original binary lsqr/auto covariance for one deletion per label row.

    Only the batch axes differ from historical loo_coefficients. No evaluation
    data or evaluation labels are accepted. Fitted means/covariance are recomputed.
    """
    x=np.asarray(x,np.float64);labels=np.asarray(labels,np.int8);deleted=np.asarray(deleted,int)
    batch,n=labels.shape;assert x.shape==(n,16) and deleted.shape==(batch,)
    assert batch<=32 and np.all(np.isin(labels,(0,1))) and deleted.min()>=0 and deleted.max()<n
    keep=np.arange(n)[None,:]!=deleted[:,None]
    pooled=np.zeros((batch,16,16),np.float64);means=[];counts=[]
    for cls in (0,1):
        mask=(labels==cls)&keep;count=mask.sum(1);assert count.min()>=2
        mean=np.einsum('bn,nd->bd',mask,x)/count[:,None]
        centered=(x[None,:,:]-mean[:,None,:])*mask[:,:,None]
        var=np.einsum('bnd,bnd->bd',centered,centered)/count[:,None]
        constant=_is_constant_feature(var,mean,count[:,None]);scale=np.sqrt(var);scale[constant]=1.
        standardized=centered/scale[:,None,:]
        standardized-=(standardized.sum(1)/count[:,None])[:,None,:]*mask[:,:,None]
        emp=np.einsum('bni,bnj->bij',standardized,standardized)/count[:,None,None]
        trace=np.trace(emp,axis1=-2,axis2=-1);mu=trace/16
        delta_sum=np.einsum('bij,bij->b',emp,emp)
        squared_norm=np.einsum('bnd,bnd->bn',standardized,standardized)
        beta_sum=np.einsum('bn,bn->b',squared_norm,squared_norm)
        beta=(beta_sum/count-delta_sum)/(16*count)
        delta=(delta_sum-2*mu*trace+16*mu**2)/16;beta=np.minimum(beta,delta)
        shrink=np.divide(beta,delta,out=np.zeros_like(beta),where=beta!=0)
        cov=emp*(1-shrink[:,None,None]);di=np.arange(16);cov[:,di,di]+=shrink[:,None]*mu[:,None]
        cov*=scale[:,None,:]*scale[:,:,None]
        pooled+=cov*count[:,None,None]/(n-1);means.append(mean);counts.append(count)
    means=np.stack(means,axis=-2);coef=np.linalg.solve(pooled,np.swapaxes(means,-1,-2));coef=np.swapaxes(coef,-1,-2)
    intercept=-.5*np.einsum('bcd,bcd->bc',means,coef)+np.log(np.stack(counts,axis=-1)/(n-1))
    w=coef[:,1,:]-coef[:,0,:];a=intercept[:,1]-intercept[:,0]
    assert np.isfinite(w).all() and np.isfinite(a).all()
    return w,a

def intervention_batches(labels,k,l,eligible,batchsize=32):
    """Yield original code/row, input condition, new training labels/deletion.

    Both X arms share partner l; score targets are never mutated here.
    """
    qs,js=np.nonzero(eligible);q=np.repeat(qs,2);j=np.repeat(js,2);arm=np.tile(np.array([0,1],np.int8),len(qs))
    for first in range(0,len(q),batchsize):
        qb=q[first:first+batchsize];jb=j[first:first+batchsize];ab=arm[first:first+batchsize];row=np.arange(len(qb))
        kb=k[qb,jb];lb=l[qb,jb];slot=np.where(ab==0,jb,kb);deleted=np.where(ab==0,kb,jb)
        train=labels[qb].copy();before=train.copy()
        train[row,slot]=before[row,lb];train[row,lb]=before[row,slot]
        keep=np.arange(labels.shape[1])[None,:]!=deleted[:,None]
        assert np.all((slot!=deleted)&(lb!=deleted)&(slot!=lb))
        assert np.all(((train!=before)&keep).sum(1)==2)
        for c in (0,1):assert np.array_equal(((train==c)&keep).sum(1),((before==c)&keep).sum(1))
        yield qb,jb,ab,train,deleted


def loo_statistics(x, labels):
    """Each label row gets N freshly refitted N-1 models; no I input.

    Covariance follows StandardScaler -> LedoitWolf -> rescale exactly.
    Repeated arm training sets reuse the same newly refitted deletion model.
    """
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim == 1:
        labels = labels[None, :]
    batch, n, d = (labels.shape[0], *x.shape)
    assert d == 16 and np.all(np.isin(labels, (0, 1)))
    include = ~np.eye(n, dtype=bool)
    pooled = np.zeros((batch, n, d, d), np.float64)
    means = []
    counts = []
    for cls in (0, 1):
        mask = (labels[:, None, :] == cls) & include[None, :, :]
        count = mask.sum(-1)
        assert count.min() >= 2
        mean = np.einsum('bkn,nd->bkd', mask, x) / count[..., None]
        centered = (x[None, None, :, :] - mean[:, :, None, :]) * mask[..., None]
        var = np.einsum('bknd,bknd->bkd', centered, centered) / count[..., None]
        constant = _is_constant_feature(var, mean, count[..., None])
        scale = np.sqrt(var)
        scale[constant] = 1.0
        standardized = centered / scale[:, :, None, :]
        standardized -= (standardized.sum(2) / count[..., None])[:, :, None, :] * mask[..., None]
        emp = np.einsum('bkni,bknj->bkij', standardized, standardized) / count[..., None, None]
        trace = np.trace(emp, axis1=-2, axis2=-1)
        mu = trace / d
        delta_sum = np.einsum('bkij,bkij->bk', emp, emp)
        squared_norm = np.einsum('bknd,bknd->bkn', standardized, standardized)
        beta_sum = np.einsum('bkn,bkn->bk', squared_norm, squared_norm)
        beta = (beta_sum / count - delta_sum) / (d * count)
        delta = (delta_sum - 2 * mu * trace + d * mu ** 2) / d
        beta = np.minimum(beta, delta)
        shrink = np.divide(beta, delta, out=np.zeros_like(beta), where=beta != 0)
        cov = emp * (1 - shrink[..., None, None])
        diagonal = np.arange(d)
        cov[:, :, diagonal, diagonal] += shrink[..., None] * mu[..., None]
        cov *= scale[..., None] * scale[..., None, :]
        pooled += cov * count[..., None, None] / (n - 1)
        means.append(mean)
        counts.append(count)
    means = np.stack(means, axis=-2)
    return (pooled, means, np.stack(counts, axis=-1))
