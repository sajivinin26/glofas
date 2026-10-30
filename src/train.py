"""Compare models for 3-day flood-onset early warning. Tune on valid (time-ordered), report once on test."""
import json, time, warnings, itertools, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, f1_score,
                             precision_score, recall_score, matthews_corrcoef, confusion_matrix)
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier
warnings.filterwarnings("ignore"); SEED=42; import pathlib; ROOT=str(pathlib.Path(__file__).resolve().parent.parent)
FEATS=["q_over_thr","lag1_over_thr","lag2_over_thr","lag3_over_thr","lag7_over_thr","mean7_over_thr","mean30_over_thr","max7_over_thr","rise_1d","rise_3d","q_over_mean30"]   # scale-free; season removed: with <2 yrs of training it acts as a date identifier and trees memorise events: no raw flow size, so models cannot memorise individual rivers
df=pd.read_csv(f"{ROOT}/data/onset_table.csv",parse_dates=["date"])
for a,b in (("q_lag1","lag1_over_thr"),("q_lag2","lag2_over_thr"),("q_lag3","lag3_over_thr"),("q_lag7","lag7_over_thr"),("q_mean7","mean7_over_thr"),("q_mean30","mean30_over_thr"),("q_max7","max7_over_thr")): df[b]=df[a]/df["thr"]
tr,va,te=[df[df.split==s] for s in ("train","valid","test")]
Xtr,ytr,Xva,yva,Xte,yte=tr[FEATS],tr.y,va[FEATS],va.y,te[FEATS],te.y
print({k:(len(v),int(v.y.sum())) for k,v in dict(train=tr,valid=va,test=te).items()},flush=True)
pos_w=(ytr==0).sum()/(ytr==1).sum()
rng=np.random.default_rng(SEED)                      # GP: stratified subsample (kernel methods scale ~n^3)
idx_pos=np.where(ytr.values==1)[0]; idx_neg=np.where(ytr.values==0)[0]
gp_idx=np.sort(np.concatenate([idx_pos, rng.choice(idx_neg,2000,replace=False)]))

def f_rule(**kw): return None
specs={
 "Rule: flow >= x * high-flow level (baseline)": None,
 "Logistic Regression (lecture baseline)": (lambda **p: make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=2000,**p)), [dict(C=c) for c in (0.01,0.1,1,10)], None),
 "Random Forest (lecture baseline)": (lambda **p: RandomForestClassifier(n_estimators=300,class_weight="balanced_subsample",random_state=SEED,n_jobs=4,**p), [dict(max_features=m,min_samples_leaf=l) for m,l in itertools.product((0.3,0.6),(2,10))], None),
 "LightGBM (gradient boosting)": (lambda **p: LGBMClassifier(random_state=SEED,verbose=-1,n_jobs=4,scale_pos_weight=pos_w,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,**p), [dict(n_estimators=n,learning_rate=lr,num_leaves=nl,min_child_samples=30) for n,lr,nl in itertools.product((150,400),(0.03,0.08),(8,16))], None),
 "Explainable Boosting Machine": (lambda **p: ExplainableBoostingClassifier(random_state=SEED,n_jobs=4,outer_bags=4,**p), [dict(learning_rate=0.03,max_leaves=3,interactions=0),dict(learning_rate=0.03,max_leaves=3,interactions=5),dict(learning_rate=0.05,max_leaves=5,interactions=5)], None),
 "Gaussian Process Classifier (subsampled)": (lambda **p: make_pipeline(StandardScaler(),GaussianProcessClassifier(kernel=C(1.0)*RBF(length_scale=p["ls"]),random_state=SEED)), [dict(ls=l) for l in (1.0,3.0)], gp_idx),
}
def best_thr(y,p):
    pr,rc,th=precision_recall_curve(y,p); f=2*pr*rc/(pr+rc+1e-12); i=np.nanargmax(f[:-1]); return float(th[i])
def scores(y,p,thr):
    yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    return dict(pr_auc=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)),precision=float(precision_score(y,yh,zero_division=0)),
        recall=float(recall_score(y,yh)),f1=float(f1_score(y,yh)),mcc=float(matthews_corrcoef(y,yh)),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))
rows,fitted,test_probs=[],{},{}
for name,spec in specs.items():
    t0=time.time()
    if spec is None:                                 # rule baseline: alert when flow already near the level
        grid=[(x,np.clip(va.q_over_thr.values/x,0,10)) for x in (0.5,0.6,0.7,0.8,0.9)]
        x,_=max(grid,key=lambda g:average_precision_score(yva,np.clip(g[1],0,1)) )
        pv=np.clip(va.q_over_thr.values,0,1.5); pt=np.clip(te.q_over_thr.values,0,1.5)
        thr=best_thr(yva,pv); r=dict(model=name,params={"rule":"q/thr >= %.2f"%thr},thr=thr,valid=scores(yva,pv,thr),test=scores(yte,pt,thr),seconds=0.0)
        rows.append(r); test_probs[name]=pt; print(name,r["valid"]["pr_auc"],r["test"]["pr_auc"],flush=True); continue
    mk,grid,sub=spec; best=None
    for p in grid:
        m=mk(**p); Xf,yf=(Xtr.iloc[sub],ytr.iloc[sub]) if sub is not None else (Xtr,ytr)
        m.fit(Xf,yf); pv=m.predict_proba(Xva)[:,1]; ap=average_precision_score(yva,pv)
        print("  ",name,p,"valid PR-AUC=%.4f"%ap,flush=True)
        if best is None or ap>best[0]: best=(ap,p,m,pv)
    ap,p,m,pv=best; thr=best_thr(yva,pv); pt=m.predict_proba(Xte)[:,1]
    r=dict(model=name,params={k:(v.item() if hasattr(v,"item") else v) for k,v in p.items()},thr=thr,valid=scores(yva,pv,thr),test=scores(yte,pt,thr),seconds=round(time.time()-t0,1))
    rows.append(r); fitted[name]=m; test_probs[name]=pt
    print(f"{name:45s} valid PR-AUC={r['valid']['pr_auc']:.4f} | test PR-AUC={r['test']['pr_auc']:.4f} ROC={r['test']['roc_auc']:.4f} P={r['test']['precision']:.3f} R={r['test']['recall']:.3f} F1={r['test']['f1']:.3f} ({r['seconds']}s)",flush=True)
best_name=max([r for r in rows if r["model"] in fitted and "baseline" not in r["model"]],key=lambda r:r["valid"]["pr_auc"])["model"]   # best NEW algorithm, chosen on VALID only (lecture models are baselines)
print("BEST (by valid PR-AUC, ML models):",best_name,flush=True)
json.dump(dict(features=FEATS,best=best_name,results=rows,counts={k:(len(v),int(v.y.sum())) for k,v in dict(train=tr,valid=va,test=te).items()}),open(f"{ROOT}/outputs/results.json","w"),indent=2)
joblib.dump(dict(model=fitted[best_name],features=FEATS,thr=[r for r in rows if r["model"]==best_name][0]["thr"],name=best_name),f"{ROOT}/models/best_model.joblib")
for n,m in fitted.items(): joblib.dump(m,f"{ROOT}/models/{n.split(' (')[0].replace(' ','_').replace(':','')}.joblib")
te.assign(**{"p_"+n.split(" (")[0]:v for n,v in test_probs.items()}).to_csv(f"{ROOT}/outputs/test_predictions.csv",index=False)
# figures
fig,ax=plt.subplots(figsize=(7,5))
for n,pt in test_probs.items():
    pr,rc,_=precision_recall_curve(yte,pt); ax.plot(rc,pr,label=f"{n.split(' (')[0][:34]} (AP={average_precision_score(yte,pt):.2f})")
ax.axhline(yte.mean(),color="grey",ls=":",label="no-skill"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves, test period (2026)"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{ROOT}/figures/pr_curves.png",dpi=160); plt.close()
names=[r["model"].split(" (")[0] for r in rows]; vals=[r["test"]["pr_auc"] for r in rows]
fig,ax=plt.subplots(figsize=(8,4)); ax.barh(names,vals,color="#4C78A8"); ax.invert_yaxis(); ax.set_xlabel("Test PR-AUC (higher is better)")
for i,v in enumerate(vals): ax.text(v+.005,i,f"{v:.3f}",va="center",fontsize=9)
ax.set_xlim(0,max(vals)*1.15); ax.set_title("Model comparison, test PR-AUC"); plt.tight_layout(); plt.savefig(f"{ROOT}/figures/model_comparison.png",dpi=160); plt.close()
