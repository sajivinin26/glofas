"""Streamlit front-end: 3-day flood-onset early warning for Sri Lankan rivers (GloFAS data).
Shows LightGBM, Logistic Regression and their blend side by side."""
import json, pathlib, numpy as np, pandas as pd, joblib, shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
st.set_page_config(page_title="River flood early warning · Sri Lanka", page_icon="🌊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.6rem;max-width:1250px}
.card{border-radius:14px;padding:16px 18px;border:1px solid rgba(128,128,128,.28);height:100%}
.card h4{margin:0 0 2px 0;font-size:.95rem;opacity:.8;font-weight:600}
.big{font-size:2.1rem;font-weight:700;line-height:1.15}
.pill{display:inline-block;padding:3px 12px;border-radius:99px;font-weight:700;font-size:.85rem;margin-top:6px}
.alert{background:#fde2dd;color:#a3260f}.ok{background:#dcf3e1;color:#17662b}
.small{font-size:.82rem;opacity:.75}
.win{border:2px solid #2E933C}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load():
    b = joblib.load(ROOT / "models" / "blend_model.joblib")
    res = json.load(open(ROOT / "outputs" / "blend_results.json"))
    df = pd.read_csv(ROOT / "data" / "onset_table.csv", parse_dates=["date"])
    for a, c in (("q_lag1","lag1_over_thr"),("q_lag2","lag2_over_thr"),("q_lag3","lag3_over_thr"),("q_lag7","lag7_over_thr"),
                 ("q_mean7","mean7_over_thr"),("q_mean30","mean30_over_thr"),("q_max7","max7_over_thr")):
        df[c] = df[a] / df["thr"]
    pts = pd.read_csv(ROOT / "outputs" / "points.csv")
    return b, res, df, pts, shap.TreeExplainer(b["lgbm"])

B, RES, df, pts, explainer = load()
FEATS, NICE, W = B["features"], B["nice"], B["w"]
NAMES = ["Logistic Regression", "LightGBM", "Blend (LightGBM + Logistic)"]
THR = {"Logistic Regression": B["thr_lr"], "LightGBM": B["thr_lgb"], "Blend (LightGBM + Logistic)": B["thr"]}

def predict(row):
    pl = float(B["lr"].predict_proba(row[FEATS])[:, 1][0]); pg = float(B["lgbm"].predict_proba(row[FEATS])[:, 1][0])
    return {"Logistic Regression": pl, "LightGBM": pg, "Blend (LightGBM + Logistic)": W * pg + (1 - W) * pl}

def cards(p):
    cols = st.columns(3)
    for c, n in zip(cols, NAMES):
        al = p[n] >= THR[n]
        c.markdown(f"""<div class="card {'win' if n.startswith('Blend') else ''}"><h4>{n}{' · used for the alert' if n.startswith('Blend') else ''}</h4>
        <div class="big">{p[n]:.0%}</div><div class="small">chance of high flow within 3 days · alert level {THR[n]:.0%}</div>
        <span class="pill {'alert' if al else 'ok'}">{'ALERT' if al else 'No alert'}</span></div>""", unsafe_allow_html=True)

def lgb_explain(row):
    sv = explainer(row[FEATS]); v = sv.values[0] if sv.values.ndim == 2 else sv.values[0, :, 1]
    e = shap.Explanation(values=v, base_values=float(np.ravel(sv.base_values)[0]), data=row[FEATS].values[0], feature_names=[NICE[c] for c in FEATS])
    fig = plt.figure(figsize=(6.4, 4)); shap.plots.waterfall(e, show=False, max_display=8); plt.tight_layout(); return fig

def lr_explain(row):
    sc, lr = B["lr"][0], B["lr"][-1]; z = sc.transform(row[FEATS])[0]; contrib = pd.Series(lr.coef_[0] * z, index=[NICE[c] for c in FEATS])
    contrib = contrib.reindex(contrib.abs().sort_values().index).tail(8)
    fig, ax = plt.subplots(figsize=(6.4, 4)); ax.barh(contrib.index, contrib.values, color=np.where(contrib.values > 0, "#E4572E", "#4C78A8")); ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("push on log-odds (red = raises risk)"); ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); return fig

def explain_block(row):
    a, b = st.columns(2)
    with a: st.markdown("**Why LightGBM says this** (SHAP)"); st.pyplot(lgb_explain(row))
    with b: st.markdown("**Why Logistic Regression says this** (coefficient × feature)"); st.pyplot(lr_explain(row))
    st.caption("Both charts show which inputs pushed the prediction up (red) or down (blue). The blended probability is the weighted average of the two models' probabilities.")

st.title("🌊 River flood early warning for Sri Lanka")
st.caption(f"Predicts whether a river that is **below** its high-flow level today will go **above** it within 3 days · {B['name']} · "
           "GloFAS v4 reanalysis (a model simulation, not gauge readings) · learning/demo tool, not an official warning")
t0, t1, t2, t3, t4 = st.tabs(["Overview", "Explore a river and date", "Enter flows manually", "Compare models", "How to read this"])

with t0:
    r = RES["results"]; te = {n: r[n]["test"] for n in NAMES}
    st.subheader("Headline results on the final test period (2026)")
    k = st.columns(4)
    k[0].metric("Blend recall", f"{te[NAMES[2]]['recall']:.0%}", help="Share of real flood starts that the blend alerted on")
    k[1].metric("Blend precision", f"{te[NAMES[2]]['precision']:.0%}", help="Share of the blend's alerts that were real")
    k[2].metric("Blend accuracy", f"{te[NAMES[2]]['accuracy']:.1%}", help="Misleading on its own: never alerting scores %.1f%%" % (RES["never_alert_accuracy"] * 100))
    k[3].metric("Blend PR-AUC", f"{te[NAMES[2]]['pr_auc']:.2f}", help="Main score for rare events (higher is better)")
    st.info(f"Only {RES['test_onsets']} of {RES['test_days']:,} test days are real flood starts, so a model that **never** alerts scores {RES['never_alert_accuracy']:.1%} accuracy while detecting nothing. "
            "That is why recall, precision and PR-AUC matter more than accuracy here.")
    st.image(str(ROOT / "figures" / "nb_03_metrics_and_accuracy.png"))

with t1:
    c1, c2 = st.columns(2)
    pid = c1.selectbox("River point", sorted(df.point_id.unique()), help="One of 33 GloFAS river cells with the largest average flow.")
    sub = df[df.point_id == pid].sort_values("date")
    d = c2.selectbox("Date (only days still below the high-flow level)", sub.date.dt.date.tolist()[::-1])
    row = sub[sub.date.dt.date == d].iloc[[0]]; pr = pts[pts.point_id == pid].iloc[0]
    st.caption(f"Point {pid}: lat {pr.lat:.3f}, lon {pr.lon:.3f} · average flow {pr.mean_q:.0f} m³/s · high-flow level {float(row.thr.iloc[0]):.0f} m³/s · flow on this day {float(row.q.iloc[0]):.0f} m³/s")
    p = predict(row); cards(p)
    out = int(row.y.iloc[0]); st.markdown(f"**What actually happened in the data:** {'🔴 high flow occurred within 3 days' if out else '🟢 no high flow within 3 days'} · data period: *{row.split.iloc[0]}*")
    explain_block(row)

with t2:
    st.write("Enter a river's recent flows (m³/s) and its high-flow level.")
    c = st.columns(3); lvl = c[0].number_input("High-flow level", 1.0, 5000.0, 100.0); q0 = c[1].number_input("Flow today", 0.0, 10000.0, 60.0); q1 = c[2].number_input("Flow 1 day ago", 0.0, 10000.0, 50.0)
    c = st.columns(3); q2 = c[0].number_input("Flow 2 days ago", 0.0, 10000.0, 45.0); q3 = c[1].number_input("Flow 3 days ago", 0.0, 10000.0, 40.0); q7 = c[2].number_input("Flow 7 days ago", 0.0, 10000.0, 30.0)
    c = st.columns(3); m7 = c[0].number_input("7-day mean flow", 0.0, 10000.0, 45.0); x7 = c[1].number_input("7-day max flow", 0.0, 10000.0, 60.0); m30 = c[2].number_input("30-day mean flow", 0.0, 10000.0, 35.0)
    row = pd.DataFrame([{"q_over_thr": q0/lvl, "lag1_over_thr": q1/lvl, "lag2_over_thr": q2/lvl, "lag3_over_thr": q3/lvl, "lag7_over_thr": q7/lvl, "mean7_over_thr": m7/lvl,
                         "mean30_over_thr": m30/lvl, "max7_over_thr": x7/lvl, "rise_1d": (q0-q1)/(q1+1e-6), "rise_3d": (q0-q3)/(q3+1e-6), "q_over_mean30": q0/(m30+1e-6)}])
    if q0 >= lvl: st.warning("Flow today is already at or above the high-flow level; the models are built for days that are still below it.")
    cards(predict(row)); explain_block(row)

with t3:
    cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    for s, title in (("test", "Final test period (2026)"), ("valid", "Validation period (used to choose settings)")):
        st.subheader(title)
        tb = pd.DataFrame({n: {c: RES["results"][n][s][c] for c in cols} for n in NAMES}).T.round(3)
        tb.columns = ["Accuracy", "Balanced acc.", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]
        st.dataframe(tb, width="stretch")
    st.caption("Settings, the blend weight and alert levels were chosen on the validation period only. The test period has few flood starts, so small differences between models are within noise.")
    for f, cap in (("nb_04_pr_roc_curves.png", "Precision-recall and ROC curves"), ("nb_05_confusion_matrices.png", "Confusion matrices at each model's alert level"),
                   ("nb_06_probability_over_time.png", "Predicted probability over time for one river point"), ("nb_11_partial_dependence.png", "How each model's risk changes with a feature")):
        st.markdown(f"**{cap}**"); st.image(str(ROOT / "figures" / f))

with t4:
    st.markdown("""
**River point (P01, P02, …)**: a GloFAS grid cell on a large river. The labels are mine, not official gauge names.

**High-flow level**: the flow a river exceeds only about 1 day in 10 (90th percentile of its training-period flow).

**Onset**: the river is below its high-flow level today but goes above it within the next 3 days. That is what the models predict.

**Probability and alert level**: the model outputs a probability. An alert is raised when it passes the alert level, which was chosen on validation data to balance false alarms and missed events.

**Precision / recall**: precision = share of alerts that were real; recall = share of real events that were caught.

**Accuracy**: shown for completeness, but misleading because flood starts are only about 1% of days.

**SHAP**: explains each LightGBM prediction by showing how much each input pushed it up or down.

**Limits**: GloFAS is a simulation, not gauge data; no rainfall input; short record; use as decision support only.
""")
