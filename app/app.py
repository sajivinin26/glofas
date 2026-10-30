"""Streamlit front-end: 3-day flood-onset early warning for Sri Lankan rivers (GloFAS data)."""
import json, pathlib, numpy as np, pandas as pd, joblib, shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Flood-onset early warning", page_icon="🌊", layout="wide")

@st.cache_resource
def load():
    b = joblib.load(ROOT / "models" / "blend_model.joblib")
    res = json.load(open(ROOT / "outputs" / "results.json"))
    df = pd.read_csv(ROOT / "data" / "onset_table.csv", parse_dates=["date"])
    for a, c in (("q_lag1","lag1_over_thr"),("q_lag2","lag2_over_thr"),("q_lag3","lag3_over_thr"),("q_lag7","lag7_over_thr"),
                 ("q_mean7","mean7_over_thr"),("q_mean30","mean30_over_thr"),("q_max7","max7_over_thr")):
        df[c] = df[a] / df["thr"]
    pts = pd.read_csv(ROOT / "outputs" / "points.csv")
    return b, res, df, pts, shap.TreeExplainer(b["model"])

bundle, res, df, pts, explainer = load()
model, FEATS, THR = bundle["model"], bundle["features"], bundle["thr"]
NICE = {"q_over_thr":"Flow today / level","lag1_over_thr":"Flow 1 d ago / level","lag2_over_thr":"Flow 2 d ago / level","lag3_over_thr":"Flow 3 d ago / level",
        "lag7_over_thr":"Flow 7 d ago / level","mean7_over_thr":"7-day mean / level","mean30_over_thr":"30-day mean / level","max7_over_thr":"7-day max / level",
        "rise_1d":"1-day relative rise","rise_3d":"3-day relative rise","q_over_mean30":"Flow today / 30-day mean"}

def explain(row: pd.DataFrame):
    sv = explainer(row[FEATS])
    v = sv.values[0] if sv.values.ndim == 2 else sv.values[0, :, 1]
    base = float(np.ravel(sv.base_values)[0])
    e = shap.Explanation(values=v, base_values=base, data=row[FEATS].values[0], feature_names=[NICE[c] for c in FEATS])
    fig = plt.figure(figsize=(7, 4)); shap.plots.waterfall(e, show=False, max_display=8); plt.tight_layout()
    return fig

def verdict(p):
    st.metric("Probability of high flow within 3 days", f"{p:.0%}")
    if p >= THR: st.error(f"ALERT: probability is above the alert threshold ({THR:.2f}).")
    else: st.success(f"No alert: probability is below the alert threshold ({THR:.2f}).")

st.title("🌊 Flood-onset early warning for Sri Lankan rivers")
st.caption(f"Model: {bundle['name']} · trained on GloFAS v4 reanalysis (model simulation, not gauge readings) · learning/demo tool, not an official warning")
tab1, tab2, tab3 = st.tabs(["Explore a river and date", "Enter flows manually", "Model comparison"])

with tab1:
    c1, c2 = st.columns(2)
    pid = c1.selectbox("River point", sorted(df.point_id.unique()), help="Grid cells with the largest average flow. Coordinates are shown below.")
    sub = df[df.point_id == pid].sort_values("date")
    d = c2.selectbox("Date (days that are not already above the high-flow level)", sub.date.dt.date.tolist()[::-1])
    r = sub[sub.date.dt.date == d].iloc[[0]]
    p = float(model.predict_proba(r[FEATS])[:, 1][0])
    pr = pts[pts.point_id == pid].iloc[0]
    st.write(f"Point {pid}: lat {pr.lat:.3f}, lon {pr.lon:.3f}, mean flow {pr.mean_q:.0f} m³/s · high-flow level {float(r.thr.iloc[0]):.0f} m³/s")
    a, b = st.columns([1, 2])
    with a:
        verdict(p)
        st.write("Actual outcome in the data:", "**high flow occurred**" if int(r.y.iloc[0]) == 1 else "no high flow")
        st.write("Data split:", r.split.iloc[0])
    with b:
        st.pyplot(explain(r))
        st.caption("Bars show how each feature pushed the prediction up (red) or down (blue), in log-odds, from the average prediction.")

with tab2:
    st.write("Enter the river's recent flows (m³/s) and its high-flow level.")
    c = st.columns(3)
    lvl = c[0].number_input("High-flow level (m³/s)", 1.0, 5000.0, 100.0)
    q0 = c[1].number_input("Flow today", 0.0, 10000.0, 60.0); q1 = c[2].number_input("Flow 1 day ago", 0.0, 10000.0, 50.0)
    c = st.columns(3)
    q2 = c[0].number_input("Flow 2 days ago", 0.0, 10000.0, 45.0); q3 = c[1].number_input("Flow 3 days ago", 0.0, 10000.0, 40.0); q7 = c[2].number_input("Flow 7 days ago", 0.0, 10000.0, 30.0)
    c = st.columns(3)
    m7 = c[0].number_input("7-day mean flow", 0.0, 10000.0, 45.0); x7 = c[1].number_input("7-day max flow", 0.0, 10000.0, 60.0); m30 = c[2].number_input("30-day mean flow", 0.0, 10000.0, 35.0)
    row = pd.DataFrame([{"q_over_thr": q0/lvl, "lag1_over_thr": q1/lvl, "lag2_over_thr": q2/lvl, "lag3_over_thr": q3/lvl, "lag7_over_thr": q7/lvl,
                         "mean7_over_thr": m7/lvl, "mean30_over_thr": m30/lvl, "max7_over_thr": x7/lvl,
                         "rise_1d": (q0-q1)/(q1+1e-6), "rise_3d": (q0-q3)/(q3+1e-6), "q_over_mean30": q0/(m30+1e-6)}])
    if q0 >= lvl: st.warning("Flow today is already at or above the high-flow level; this model is built for onset, i.e. days that are still below it.")
    p = float(model.predict_proba(row[FEATS])[:, 1][0]); a, b = st.columns([1, 2])
    with a: verdict(p)
    with b: st.pyplot(explain(row))

with tab3:
    rows = [{"Model": r["model"], "Valid PR-AUC": r["valid"]["pr_auc"], "Test PR-AUC": r["test"]["pr_auc"], "Test ROC-AUC": r["test"]["roc_auc"],
             "Test precision": r["test"]["precision"], "Test recall": r["test"]["recall"], "Test F1": r["test"]["f1"]} for r in res["results"]]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.caption("Selected model = best new algorithm by validation PR-AUC. Test period (2026) has only 100 onset days from about two events, so differences between models are uncertain.")
    for f in ("model_comparison.png", "pr_curves.png"): st.image(str(ROOT / "figures" / f))
