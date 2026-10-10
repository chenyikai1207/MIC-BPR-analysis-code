"""Historical matched-listener origin kernels; input dictionaries supplied locally."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

TASKS=('IMAGERY_RECALL','IMAGERY_MANIPULATION')

DIRECTIONS=(('RECALL_TO_MANIPULATION',TASKS[0],TASKS[1]),('MANIPULATION_TO_RECALL',TASKS[1],TASKS[0]))

SEED=20261018

def source_indices(d,task,b):
    a=d['available'][task];y=d[task]['y'];return np.concatenate([a[y[a]==c][:b] for c in (0,1)])

def other(pid,pids): return pids[(pids.index(pid)+1)%len(pids)]

def project(a,sc,pc): return pc.transform(sc.transform(a.transpose(0,2,1).reshape(-1,306))).reshape(len(a),20,16)

def fit_source(s,task,dd,b,budget):
    budget.check();d=dd[s];ix=source_indices(d,task,b);v=d[task];P=v['P'][ix];I=v['I'][ix];y=v['y'][ix];assert len(ix)==2*b
    sc=StandardScaler();pc=PCA(n_components=16,svd_solver='full').fit(sc.fit_transform(P.transpose(0,2,1).reshape(-1,306)));sp=project(P,sc,pc);si=project(I,sc,pc);intended=y if task==TASKS[0] else 1-y
    m=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(si[:,2:18].reshape(2*b,-1),intended);budget.fits+=1
    return {'s':s,'task':task,'sc':sc,'pc':pc,'sp':sp,'physical':y,'model':m,'source_uid':v['uid'][ix]}

def rotation(bundle,d,controls=False):
    cp=project(d[TASKS[0]]['P'][d['cal']],bundle['sc'],bundle['pc']);cy=d[TASKS[0]]['y'][d['cal']];sp=bundle['sp'];sy=bundle['physical']
    ref=np.stack([sp[sy==c].mean(0) for c in (0,1)])[cy]
    def solve(cp):
        x=cp[:,2:18,:4].reshape(-1,4).astype('float64');y=ref[:,2:18,:4].reshape(-1,4).astype('float64');x-=x.mean(0);y-=y.mean(0);u,sv,vt=np.linalg.svd(x.T@y,full_matrices=False);R=np.eye(16);R[:4,:4]=u@vt;return R,sv
    R,sv=solve(cp);control={}
    if controls:
        A=np.linalg.qr(np.random.default_rng(SEED).normal(size=(4,4)))[0];injected=cp.astype('float64').copy();injected[:,:,:4]=injected[:,:,:4]@A;RR,_=solve(injected)
        recovery=float(np.linalg.norm(A@RR[:4,:4]-R[:4,:4])/2);norm=float(np.max(np.abs(np.linalg.norm(cp.astype('float64')@R,axis=2)-np.linalg.norm(cp.astype('float64'),axis=2))));orth=float(np.linalg.norm(R.T@R-np.eye(16)))
        assert recovery<1e-9 and norm<1e-10 and orth<1e-10;control={'recovery':recovery,'norm':norm,'orthogonality':orth,'calibration_target_I_used':False}
    return R,sv,control

def ba(y,p): return float(.5*(np.mean(p[y==0]==0)+np.mean(p[y==1]==1)))

def score(bundle,pid,dirspec,origin,dd,controls=False):
    direction,train,test=dirspec;assert bundle['task']==train;d=dd[pid];v=d[test];ix=d['test'][test];uids=v['uid'][ix]
    assert not set(map(str,bundle['source_uid'])).intersection(map(str,uids));R,sv,cc=rotation(bundle,d,controls);x=project(v['I'][ix],bundle['sc'],bundle['pc'])[:,2:18].astype('float64')@R;e=bundle['model'].decision_function(x.reshape(len(ix),-1));assert np.isfinite(e).all();pred=(e>0).astype(int);y=v['y'][ix];intended=y if test==TASKS[0] else 1-y
    row={'participant':pid,'direction':direction,'origin':origin,'source':bundle['s'],'n_test_originals':len(ix),'n_test_class0':int(sum(y==0)),'n_test_class1':int(sum(y==1)),'BA_intended':ba(intended,pred),'BA_physical':ba(y,pred),'predicted_class0':int(sum(pred==0)),'predicted_class1':int(sum(pred==1)),'calibration_min_singular':float(min(sv))}
    private={'uids':uids,'physical':y,'intended':intended,'evidence':e,'prediction':pred,'R':R}
    return row,private,cc
