# ai-models
this repo is used for generating new AI Models
# Mobile Price Classification — Streamlit App

This folder contains a small Streamlit app to explore models trained on the Mobile Price Classification dataset.

Files added/modified
- `streamlit_app.py` — interactive Streamlit UI to train/evaluate models and show metrics.
- `requirements.txt` — updated to include `streamlit` and `joblib`.
- `run_models.py` — existing script was enhanced to generate human-readable observations per model.

Run locally

1. Create / activate the repository virtual environment (optional if already created):

```powershell
C:/pythonWorkspace/.venv/Scripts/python.exe -m pip install -r Sem1Assignments/ML/Assignment2/mobilePriceClassificationDataset/requirements.txt
C:/pythonWorkspace/.venv/Scripts/python.exe -m streamlit run Sem1Assignments/ML/Assignment2/mobilePriceClassificationDataset/streamlit_app.py
```

2. Use the sidebar to select a model, click `Train / Retrain selected model`, then `Evaluate` to see metrics, observation, and confusion matrix.

Deploy to Streamlit Community Cloud

1. Commit and push this repository to GitHub (public or private).
2. On Streamlit Community Cloud, create a new app and point it to:
   - Repository: your GitHub repo
   - File to run: `Sem1Assignments/ML/Assignment2/mobilePriceClassificationDataset/streamlit_app.py`
3. Ensure `Sem1Assignments/ML/Assignment2/mobilePriceClassificationDataset/requirements.txt` is present at repo root or in the same folder; Streamlit Cloud will install the packages.

Notes
- If `test.csv` has no labels, `run_models.py` will split `train.csv` (80/20).
- For large models or long training, consider persist models with `joblib` and loading them in the app.

# Mobile Price Classification - Models and Metrics

Files:
- `run_models.py` - trains models and computes metrics.
- `requirements.txt` - Python dependencies.

Usage:

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Place `train.csv` and (optionally) `test.csv` in this folder.
3. Run:

```bash
python run_models.py
```

Results are saved to `results_metrics.csv`.
