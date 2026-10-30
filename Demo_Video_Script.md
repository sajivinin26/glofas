# Demo video script (about 4 minutes)

**0:00–0:30 Problem.** Floods in Sri Lanka's river basins are driven by the two monsoons. I use GloFAS river-discharge data (Copernicus), 33 river points, 2024 to Oct 2026, to predict whether a river that is currently below its own high-flow level (its 90th percentile in the training period) will exceed it within the next 3 days.

**0:30–1:10 Data and split.** Features are today's flow, lags of 1, 2, 3 and 7 days, 7- and 30-day means, 7-day maximum and recent rises, all relative to each point's high-flow level. The split is by time, not random: train on the earliest period, validate next, test on the latest. Show the split on the slide: 352 / 162 / 100 onset events.

**1:10–2:10 Models.** Baselines from the lectures: Logistic Regression and Random Forest, plus a simple rule. New algorithms: LightGBM, Explainable Boosting Machine, Gaussian Process. Show the comparison figure. Best new algorithm on validation PR-AUC: LightGBM (valid 0.41, test 0.28). Be honest: test scores are lower than validation and precision is low, because onset events are rare (100 in the test period). Logistic Regression was strongest on validation (0.42), so the new algorithms do not clearly win.

**2:10–3:00 Explainability.** Show the SHAP bar and beeswarm plots. Top drivers: flow relative to the high-flow level, flow relative to the 30-day mean, and the 1-day relative rise. This matches hydrology: a river already near its threshold and rising fast is most likely to cross it. Show one waterfall explanation (point P28, 9 Oct 2026, probability 0.95).

**3:00–3:50 App.** Run `streamlit run app/app.py`. Choose a river point and date, show the predicted probability, the alert decision and the SHAP explanation.

**3:50–4:00 Limits.** GloFAS is a simulation, not gauge data; no rainfall input; short record. Use as decision support only.
