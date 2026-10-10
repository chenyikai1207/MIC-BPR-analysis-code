"""Historical spatial-PCA16/four-coordinate D1 control; distinct from current MEG."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

TAKE=slice(2,18)

ROUTES=('I_IDENTITY','I_FIXED4_P_ROTATION','P_NEURAL_IDENTITY')

CONDITIONS=('IMAGERY_RECALL','IMAGERY_MANIPULATION')

def sources_for(pid,pids,people):
    j=pids.index(pid);ordered=[pids[(j+delta)%len(pids)] for delta in range(1,len(pids))];sources=[s for s in ordered if people[s]['eligible']][:10];assert len(sources)==10 and pid not in sources;return sources

def project(a,scaler,pca):return pca.transform(scaler.transform(a.transpose(0,2,1).reshape(-1,306))).reshape(len(a),20,16)

def fit_R(sp,sy,cp,cy):
    ref=np.stack([sp[sy==c].mean(0) for c in [0,1]])[cy]
    x=cp[:,TAKE,:4].reshape(-1,4).astype('float64');y=ref[:,TAKE,:4].reshape(-1,4).astype('float64');x-=x.mean(0);y-=y.mean(0);u,s,vt=np.linalg.svd(x.T@y,full_matrices=False);R=np.eye(16);R[:4,:4]=u@vt;return R,s

def lda(x,y):return LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(x,y)

def fit_bundle(pid,sources,people,control=False):
    ds=[people[s] for s in sources];SP=np.concatenate([d['P'][d['src']] for d in ds]);SI=np.concatenate([d['I'][d['src']] for d in ds]);sy=np.concatenate([d['y'][d['src']] for d in ds]);d=people[pid]
    sourceuids=np.concatenate([d['uids'][d['src']] for d in ds]);assert len(SP)==len(SI)==320 and sum(sy==0)==sum(sy==1)==160 and pid not in sources
    assert not set(map(str,sourceuids))&set(map(str,d['uids']))
    sc=StandardScaler();pc=PCA(n_components=16,svd_solver='full').fit(sc.fit_transform(SP.transpose(0,2,1).reshape(-1,306)))
    sp=project(SP,sc,pc);si=project(SI,sc,pc);cp=project(d['P'][d['cal']],sc,pc);cy=d['y'][d['cal']];R,s=fit_R(sp,sy,cp,cy)
    controls={}
    if control:
        A=np.linalg.qr(np.random.default_rng(20261008).normal(size=(4,4)))[0];injected=cp.astype('float64').copy();injected[:,:,:4]=injected[:,:,:4]@A
        RR,_=fit_R(sp,sy,injected,cy);err=float(np.linalg.norm(A@RR[:4,:4]-R[:4,:4])/2)
        normerr=float(np.max(np.abs(np.linalg.norm(cp.astype('float64')@R,axis=2)-np.linalg.norm(cp.astype('float64'),axis=2))))
        assert err<1e-9 and normerr<1e-10 and np.linalg.norm(R.T@R-np.eye(16))<1e-10
        controls={'rotation_recovery_error':err,'norm_preservation_error':normerr,'target_I_used':False}
    vi=si[:,TAKE].reshape(320,-1);vp=sp[:,TAKE].reshape(320,-1)
    return {'pid':pid,'sources':sources,'sc':sc,'pc':pc,'sp':sp,'si':si,'cp':cp,'sy':sy,'cy':cy,'vi':vi,'vp':vp,'R':R,'I_model':lda(vi,sy),'P_model':lda(vp,sy),'controls':controls,'sourceuids':sourceuids,'R_cross_singular':s}

def evaluation_input(bundle,d):
    out={}
    for cond,key,ix,y,uid in [('IMAGERY_RECALL','I',d['test'],d['y'],d['uids']),('IMAGERY_MANIPULATION','MI',np.arange(len(d['my'])),d['my'],d['muids'])]:
        x=project(d[key][ix],bundle['sc'],bundle['pc'])[:,TAKE];physical=y[ix];intended=physical if cond=='IMAGERY_RECALL' else 1-physical
        assert set(physical)=={0,1};out[cond]={'x':x,'physical':physical,'intended':intended,'uid':uid[ix]}
    return out

def ba(y,p):return float(.5*(np.mean(p[y==0]==0)+np.mean(p[y==1]==1)))

def metrics(e,ctx):
    pred=(e>0).astype(int);physical=ctx['physical'];intended=ctx['intended'];se=float(.5*(e[physical==1].mean()-e[physical==0].mean()))
    return {'n_score':len(pred),'physical_class0':int(sum(physical==0)),'physical_class1':int(sum(physical==1)),'BA_intended':ba(intended,pred),'BA_physical':ba(physical,pred),'signed_physical_balanced':se,'signed_intended_balanced':se if np.array_equal(intended,physical) else -se,'predicted_class0':int(sum(pred==0)),'predicted_class1':int(sum(pred==1))}

def evaluate(bundle,ctxs,private=False):
    rows=[];preds=[]
    for cond,ctx in ctxs.items():
        x=ctx['x'];xx=x.reshape(len(x),-1)
        evidence={'I_IDENTITY':bundle['I_model'].decision_function(xx),'I_FIXED4_P_ROTATION':bundle['I_model'].decision_function((x@bundle['R']).reshape(len(x),-1)),'P_NEURAL_IDENTITY':bundle['P_model'].decision_function(xx)}
        for route,e in evidence.items():
            m=metrics(e,ctx);rows.append({'participant':bundle['pid'],'condition':cond,'route':route,**m})
            if private:
                for uid,p,i,ev in zip(ctx['uid'],ctx['physical'],ctx['intended'],e):preds.append({'participant':bundle['pid'],'original_trial_uid':str(uid),'condition':cond,'route':route,'physical_label':int(p),'intended_label':int(i),'decision_evidence':float(ev),'prediction':int(ev>0)})
    for row in rows:
        base=next(r for r in rows if r['condition']==row['condition'] and r['route']=='I_IDENTITY');row['intended_BA_gain_over_I_identity']=row['BA_intended']-base['BA_intended']
    return rows,preds

def pseudo_labels(people,rng):
    # Whole original recall trial is the exchange unit; source P/I twins and its calibration role share assignment.
    assigned={}
    for pid,d in people.items():
        yy=d['y'].copy();cal=d['cal'];yy[cal]=rng.permutation(yy[cal]);assert sum(yy[cal]==0)==sum(yy[cal]==1)==4
        if d['eligible']:
            rem=np.asarray([j for j in d['src'] if j not in set(cal)]);yy[rem]=rng.permutation(yy[rem]);assert sum(yy[d['src']]==0)==sum(yy[d['src']]==1)==16
        assigned[pid]=yy
    return assigned

def aggregate(rows):
    return [{ 'condition':cond,'route':route,'participant_N':len(rr),'mean_BA_intended':float(np.mean([r['BA_intended'] for r in rr])),'mean_BA_physical':float(np.mean([r['BA_physical'] for r in rr])),'mean_signed_intended_balanced':float(np.mean([r['signed_intended_balanced'] for r in rr])),'mean_signed_physical_balanced':float(np.mean([r['signed_physical_balanced'] for r in rr])),'mean_gain_over_I_identity':float(np.mean([r['intended_BA_gain_over_I_identity'] for r in rr])),'single_class_participant_count':sum(r['predicted_class0']==0 or r['predicted_class1']==0 for r in rr)} for cond in CONDITIONS for route in ROUTES for rr in [[r for r in rows if r['condition']==cond and r['route']==route]]]
