# Flood-onset early warning for Sri Lankan rivers (GloFAS + interpretable ML)

Predicts whether a river point's discharge will rise above its own high-flow level
(90th percentile of training-period flow) within the next 3 days, for days when it is
currently below that level. Compares Logistic Regression and Random Forest (lecture baselines)
with LightGBM, an Explainable Boosting Machine and a Gaussian Process classifier (new algorithms).
The best new algorithm (chosen on the validation period) is explained with SHAP and partial dependence.

## Main workflow
Open `notebooks/Flood_Onset_Analysis.ipynb` (Kernel > Restart & Run All). It trains Logistic Regression and LightGBM, blends them (weight chosen on validation), compares them with graphs and an accuracy check, runs SHAP / partial dependence, and saves `models/blend_model.joblib` used by the app.

## Folder layout
- `notebooks/Flood_Onset_Analysis.ipynb`  the full analysis with all graphs (start here)
- `raw/`      GloFAS v4 reanalysis NetCDF files (daily river discharge, Sri Lanka box N10 W79.4 S5.8 E82), downloaded from the Copernicus EWDS
- `src/prepare.py`  builds `data/onset_table.csv` (river points, features, target, time-based split)
- `src/train.py`    trains/tunes/compares models, saves `outputs/results.json`, `models/`, figures
- `src/explain.py`  SHAP, partial dependence, EBM cross-check, figures
- `app/app.py`      Streamlit front-end (bonus)
- `outputs/`, `figures/`  results used in the report

## Run
```
pip install -r requirements.txt
python src/prepare.py
python src/train.py        # about 6 minutes (Gaussian Process is the slowest)
python src/explain.py
jupyter notebook notebooks/Flood_Onset_Analysis.ipynb   # blend + comparison graphs; creates models/blend_model.joblib
streamlit run app/app.py
```
Note: `train.py` regenerates all model files; the zip ships only the selected model plus LightGBM, EBM and Logistic Regression to keep it small.

## Data
GloFAS v4 historical reanalysis (LISFLOOD), Copernicus Emergency Management Service, obtained via
https://ewds.climate.copernicus.eu under the CEMS-FLOODS datasets licence. 2 Jan 2024 to 25 Oct 2026 (one missing day, 1 Jan 2025, filled by interpolation).
No personal data. The data are model simulations, not gauge observations.
