"""Historical paired-versus-disjoint D6 control kernels."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import t as st
import historical_d1 as old
TASKS=old.CONDITIONS

COUNTS=(10,20)

BUDGETS=(2,4)

PAIRING=('DISJOINT_ORIGINAL','PAIRED_ORIGINAL')

def donors(pid,pids,people,count):
    j=pids.index(pid);out=[pids[(j+k)%len(pids)] for k in range(1,len(pids)) if people[pids[(j+k)%len(pids)]]['eligible']][:count];assert len(out)==count and pid not in out;return out

def cal_indices(d,b,pairing):
    allowed=d['pool'] if pairing==PAIRING[0] else d['test'];ix=np.concatenate([allowed[d['y'][allowed]==c][:b] for c in [0,1]]);assert len(ix)==2*b
    if pairing==PAIRING[0]:assert not set(ix)&set(d['test'])
    else:assert set(ix)<=set(d['test'])
    return ix

def fit_source(pid,srcs,people,count,save=None):
    n=160//count;assert n*count==160;pp=[];ii=[];yy=[];uid=[]
    for src in srcs:
        d=people[src];ix=np.concatenate([np.flatnonzero(d['y']==c)[:n] for c in [0,1]]);assert len(ix)==2*n;pp.append(d['P'][ix]);ii.append(d['I'][ix]);yy.append(d['y'][ix]);uid.extend(map(str,d['uids'][ix]))
    P=np.concatenate(pp);I=np.concatenate(ii);y=np.concatenate(yy);assert len(P)==len(I)==len(uid)==len(set(uid))==320;sc=StandardScaler().fit(P.transpose(0,2,1).reshape(-1,306));pc=PCA(n_components=16,svd_solver='full').fit(sc.transform(P.transpose(0,2,1).reshape(-1,306)));sp=old.project(P,sc,pc);si=old.project(I,sc,pc);model=old.lda(si[:,2:18].reshape(320,256),y);bundle={'sc':sc,'pc':pc,'sp':sp,'sy':y,'model':model,'source_people':srcs,'source_uids':uid,'count':count}
    if save is not None:np.savez_compressed(save,sc_mean=sc.mean_,sc_scale=sc.scale_,pc_mean=pc.mean_,pc_components=pc.components_,sp=sp,si=si,sy=y,source_uids=np.array(uid),source_people=np.array(srcs),coef=model.coef_,intercept=model.intercept_)
    return bundle

def evaluate(bundle,d,save=None):
    rows=[];payload={};contexts=old.evaluation_input({'sc':bundle['sc'],'pc':bundle['pc']},d)
    for cond,ctx in contexts.items():
        x=ctx['x'];e=bundle['model'].decision_function(x.reshape(len(x),256));m=old.metrics(e,ctx);rows.append({'source_count':bundle['count'],'cal_per_class':0,'pairing':'IDENTITY','task':cond,**m});payload[cond+'_IDENTITY_decision']=e;payload[cond+'_score_UIDs']=ctx['uid'];payload[cond+'_physical']=ctx['physical'];payload[cond+'_intended']=ctx['intended']
    for b in BUDGETS:
        for pairing in PAIRING:
            ix=cal_indices(d,b,pairing);cp=old.project(d['P'][ix],bundle['sc'],bundle['pc']);rot,s=old.fit_R(bundle['sp'],bundle['sy'],cp,d['y'][ix]);key=str(b)+'_'+pairing;payload[key+'_R']=rot;payload[key+'_cal_UIDs']=d['uids'][ix];payload[key+'_singular']=s
            for cond,ctx in contexts.items():
                x=ctx['x'];e=bundle['model'].decision_function((x@rot).reshape(len(x),256));m=old.metrics(e,ctx);rows.append({'source_count':bundle['count'],'cal_per_class':b,'pairing':pairing,'task':cond,**m});payload[cond+'_'+key+'_decision']=e
    if save is not None:np.savez_compressed(save,**payload)
    return rows

def endpoints(rows,pids):
    out=[]
    for pid in pids:
        mm={(r['source_count'],r['cal_per_class'],r['pairing'],r['task']):r['BA_intended'] for r in rows if r['participant']==pid};g=[]
        for task in TASKS:g.append(np.mean([mm[n,b,PAIRING[1],task]-mm[n,b,PAIRING[0],task] for n in COUNTS for b in BUDGETS]))
        out.append([g[0],g[0]-g[1]])
    return np.array(out)

def infer(X):
    rng=np.random.default_rng(20261016);m=X.mean(0);sg=rng.choice([-1,1],(10000,len(X)));p=(1+np.sum(sg@X/len(X)>=m,axis=0))/10001;idx=rng.integers(0,len(X),(5000,len(X)));ci=np.quantile(X[idx].mean(1),[.025,.975],axis=0).T;return {'N':len(X),'endpoint_order':['paired_minus_disjoint_recall_BA_average_fixedcount_budget','thatgain_minus_manipulation_gain'],'means':m.tolist(),'conditional_CI95':ci.tolist(),'conditional_one_sided_p':p.tolist(),'IUT_p':float(p.max()),'candidate_supported':bool(m.min()>0 and p.max()<.05),'symmetry_required':True,'paired_calibration_labels_reveal_some_recall_test_intended_labels':True,'not_original_trial_heldout_in_paired_branch':True}
