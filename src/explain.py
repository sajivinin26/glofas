"""Explainability: SHAP (global + local), partial dependence, and EBM cross-check for the selected model."""
import json, numpy as np, pandas as pd, joblib, shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import pathlib; ROOT=str(pathlib.Path(__file__).resolve().parent.parent)
res=json.load(open(f"{ROOT}/outputs/results.json")); FEATS=res["features"]; best=res["best"]; print("explaining:",best)
bundle=joblib.load(f"{ROOT}/models/best_model.joblib"); model=bundle["model"]
df=pd.read_csv(f"{ROOT}/data/onset_table.csv",parse_dates=["date"])
for a,b in (("q_lag1","lag1_over_thr"),("q_lag2","lag2_over_thr"),("q_lag3","lag3_over_thr"),("q_lag7","lag7_over_thr"),("q_mean7","mean7_over_thr"),("q_mean30","mean30_over_thr"),("q_max7","max7_over_thr")): df[b]=df[a]/df["thr"]
te=df[df.split=="test"].reset_index(drop=True); X=te[FEATS]
NICE={"q_over_thr":"Flow today / high-flow level","lag1_over_thr":"Flow 1 day ago / level","lag2_over_thr":"Flow 2 days ago / level","lag3_over_thr":"Flow 3 days ago / level","lag7_over_thr":"Flow 7 days ago / level",
 "mean7_over_thr":"7-day mean flow / level","mean30_over_thr":"30-day mean flow / level","max7_over_thr":"7-day max flow / level","rise_1d":"1-day relative rise","rise_3d":"3-day relative rise","q_over_mean30":"Flow today / 30-day mean"}
Xn=X.rename(columns=NICE)
expl=shap.TreeExplainer(model); sv=expl(X)
vals=sv.values if sv.values.ndim==2 else sv.values[:,:,1]
base=float(np.ravel(sv.base_values)[0])
exp=shap.Explanation(values=vals,base_values=np.full(len(X),base),data=Xn.values,feature_names=list(Xn.columns))
shap.plots.beeswarm(exp,show=False,max_display=11); plt.title("SHAP summary (test period)"); plt.tight_layout(); plt.savefig(f"{ROOT}/figures/shap_beeswarm.png",dpi=160,bbox_inches="tight"); plt.close()
shap.plots.bar(exp,show=False,max_display=11); plt.title("Mean |SHAP| (log-odds units)"); plt.tight_layout(); plt.savefig(f"{ROOT}/figures/shap_bar.png",dpi=160,bbox_inches="tight"); plt.close()
imp=pd.Series(np.abs(vals).mean(0),index=[NICE[c] for c in FEATS]).sort_values(ascending=False); imp.to_csv(f"{ROOT}/outputs/shap_importance.csv",header=["mean_abs_shap"]); print(imp.round(3))
# local explanation: highest-risk test case that was a true onset
p=model.predict_proba(X)[:,1]; cand=np.where((te.y.values==1))[0]; i=int(cand[np.argmax(p[cand])])
shap.plots.waterfall(exp[i],show=False,max_display=9); plt.title(f"Local explanation: {te.point_id[i]} on {te.date[i].date()}  (P={p[i]:.2f})",fontsize=9); plt.tight_layout(); plt.savefig(f"{ROOT}/figures/shap_waterfall.png",dpi=160,bbox_inches="tight"); plt.close()
# dependence for the top-2 features
top=list(imp.index[:2]); inv={v:k for k,v in NICE.items()}
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,nm in zip(axs,top):
    j=FEATS.index(inv[nm]); ax.scatter(X.iloc[:,j],vals[:,j],s=6,alpha=.4,color="#4C78A8"); ax.axhline(0,color="grey",lw=.8); ax.set_xlabel(nm); ax.set_ylabel("SHAP value"); 
    lo,hi=np.percentile(X.iloc[:,j],[1,99]); ax.set_xlim(lo,hi)
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/shap_dependence.png",dpi=160); plt.close()
# partial dependence (model-agnostic) for top-4 features
top4=[inv[n] for n in imp.index[:4]]
disp=PartialDependenceDisplay.from_estimator(model,X,[FEATS.index(c) for c in top4],feature_names=[NICE[c] for c in FEATS],grid_resolution=30,percentiles=(0.02,0.98),n_cols=2,line_kw={"color":"#4C78A8"})
disp.figure_.set_size_inches(9,6); disp.figure_.suptitle("Partial dependence (model output is not calibrated: class weighting inflates probabilities)",y=1.0,fontsize=10); disp.figure_.tight_layout(h_pad=3); disp.figure_.savefig(f"{ROOT}/figures/pdp_top4.png",dpi=160,bbox_inches="tight"); plt.close("all")
# cross-check with the Explainable Boosting Machine's own term importances
ebm=joblib.load(f"{ROOT}/models/Explainable_Boosting_Machine.joblib"); ex=ebm.explain_global().data()
eimp=pd.Series(ex["scores"],index=ex["names"]).sort_values(ascending=False); eimp.to_csv(f"{ROOT}/outputs/ebm_importance.csv",header=["importance"])
print("\nEBM top terms:\n",eimp.head(6).round(3))
from scipy.stats import spearmanr
common=[c for c in FEATS if c in eimp.index]; rho=spearmanr([imp[NICE[c]] for c in common],[eimp[c] for c in common])[0]; print("Spearman(SHAP rank vs EBM rank)=%.2f"%rho)
json.dump(dict(top_shap=imp.head(5).round(4).to_dict(),spearman_shap_ebm=float(rho),local_case=dict(point=te.point_id[i],date=str(te.date[i].date()),p=float(p[i]))),open(f"{ROOT}/outputs/xai_summary.json","w"),indent=2)
