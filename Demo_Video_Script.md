# Demo video script (about 4 minutes)

**0:00–0:30 Problem.** Floods in Sri Lanka's river basins are driven by the two monsoons. I use GloFAS river-discharge data (Copernicus), 33 river points, 2024 to Oct 2026, to predict whether a river that is currently below its own high-flow level (its 90th percentile in the training period) will exceed it within the next 3 days.

**0:30–1:10 Data and split.** Features are today's flow, lags of 1, 2, 3 and 7 days, 7- and 30-day means, 7-day maximum and recent rises, all relative to each point's high-flow level. The split is by time, not random: train on the earliest period, validate next, test on the latest. Show the split on the slide: 352 / 162 / 100 onset events.

**1:10–2:10 Models.** Baselines from the lectures: Logistic Regression and Random Forest. New algorithms: LightGBM, Explainable Boosting Machine, Gaussian Process. Then show the notebook: I blended LightGBM with Logistic Regression (0.4 LightGBM + 0.6 Logistic, weight chosen on validation only). Show `nb_03` (scores and accuracy) and `nb_04` (PR and ROC curves). Test results: LightGBM PR-AUC 0.283 and recall 0.53; Logistic PR-AUC 0.241 and recall 0.34; Blend PR-AUC 0.282, ROC-AUC 0.905 (highest), recall 0.45 with fewer false alarms than LightGBM (182 vs 306). Accuracy is 97.5% for the blend but a model that never alerts scores 99.0%, so accuracy is misleading; judge by PR-AUC and recall. Be honest: with only 100 test onsets the differences between the three are within chance.

**2:10–3:00 Explainability.** Show the SHAP bar and beeswarm plots. Top drivers: flow relative to the high-flow level, flow relative to the 30-day mean, and the 1-day relative rise. This matches hydrology: a river already near its threshold and rising fast is most likely to cross it. Show the SHAP waterfall (nb_10) and the Logistic coefficient chart (nb_09).

**3:00–3:50 App.** Run `streamlit run app/app.py`. Show the Overview tab, then Explore: choose a river point and date and show the three model cards (Logistic, LightGBM, Blend), the alert, what actually happened, and both explanation charts.

**3:50–4:00 Limits.** GloFAS is a simulation, not gauge data; no rainfall input; short record. Use as decision support only.
