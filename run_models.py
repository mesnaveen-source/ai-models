"""
Script to train multiple classifiers on the Mobile Price Classification dataset and
compute evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC.

Expects dataset files at:
- mobilePriceClassificationDataset/train.csv
- mobilePriceClassificationDataset/test.csv

If `test.csv` does not contain labels, the script will split `train.csv`.

Outputs: `results_metrics.csv` saved in the same folder.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


DATA_DIR = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TARGET_COLUMN = None  # will infer


def load_data(train_path, test_path):
    if not os.path.exists(train_path):
        print(f"Missing {train_path}")
        sys.exit(1)
    df_train = pd.read_csv(train_path)

    if df_train.shape[0] < 10:
        print("Train file looks too small. Please check dataset files.")
        sys.exit(1)

    # Infer target column: common mobile price dataset uses 'price_range'
    global TARGET_COLUMN
    if TARGET_COLUMN is None:
        if "price_range" in df_train.columns:
            TARGET_COLUMN = "price_range"
        else:
            # fallback: assume last column
            TARGET_COLUMN = df_train.columns[-1]

    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
    else:
        df_test = None

    # If test exists but has no target, we'll split train
    if df_test is None or TARGET_COLUMN not in df_test.columns:
        print("Test file missing or has no labels — splitting train.csv into train/test (80/20).")
        X = df_train.drop(columns=[TARGET_COLUMN])
        y = df_train[TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train = df_train.drop(columns=[TARGET_COLUMN])
        y_train = df_train[TARGET_COLUMN]
        X_test = df_test.drop(columns=[TARGET_COLUMN])
        y_test = df_test[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def safe_auc(y_true, y_score, labels):
    # y_score expected shape (n_samples, n_classes) for multiclass
    try:
        if len(labels) > 2:
            y_true_bin = label_binarize(y_true, classes=labels)
            return roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
        else:
            # binary
            return roc_auc_score(y_true, y_score[:, 1])
    except Exception as e:
        print("AUC calculation failed:", e)
        return np.nan


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    metrics = {}

    metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    metrics["Precision"] = precision_score(y_test, y_pred, average="macro", zero_division=0)
    metrics["Recall"] = recall_score(y_test, y_pred, average="macro", zero_division=0)
    metrics["F1"] = f1_score(y_test, y_pred, average="macro", zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_test, y_pred)

    labels = np.unique(np.concatenate([y_test, y_pred]))

    # AUC: prefer predict_proba, fallback to decision_function
    y_score = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_score = pipeline.predict_proba(X_test)
        except Exception:
            y_score = None
    if y_score is None and hasattr(pipeline, "decision_function"):
        try:
            dec = pipeline.decision_function(X_test)
            # decision_function may return (n_samples,) for binary
            if dec.ndim == 1:
                dec = np.vstack([1 - dec, dec]).T
            y_score = dec
        except Exception:
            y_score = None

    if y_score is not None:
        metrics["AUC"] = safe_auc(y_test, y_score, labels)
    else:
        metrics["AUC"] = np.nan

    return metrics


def generate_observation(metrics: dict) -> str:
    """Return a short human-readable observation about model performance.

    Heuristics used:
    - Accuracy/F1/AUC thresholds to classify overall quality.
    - Precision vs Recall imbalance notes.
    - Low or NaN AUC warning.
    """
    acc = metrics.get("Accuracy", np.nan)
    f1 = metrics.get("F1", np.nan)
    auc = metrics.get("AUC", np.nan)
    prec = metrics.get("Precision", np.nan)
    rec = metrics.get("Recall", np.nan)
    mcc = metrics.get("MCC", np.nan)

    parts = []

    # Overall quality from F1 / Accuracy
    score = np.nanmean([v for v in [f1, acc] if not np.isnan(v)])
    if not np.isnan(score):
        if score >= 0.9:
            parts.append("Excellent overall performance")
        elif score >= 0.8:
            parts.append("Good performance")
        elif score >= 0.7:
            parts.append("Moderate performance")
        else:
            parts.append("Weak performance")
    else:
        parts.append("Insufficient metrics to assess overall quality")

    # AUC remark
    if np.isnan(auc):
        parts.append("AUC unavailable")
    else:
        if auc >= 0.9:
            parts.append("Strong ranking ability (high AUC)")
        elif auc >= 0.8:
            parts.append("Good ranking ability (AUC)")

    # Precision / Recall balance
    if not np.isnan(prec) and not np.isnan(rec):
        if prec - rec >= 0.15:
            parts.append("Higher precision than recall — conservative predictions")
        elif rec - prec >= 0.15:
            parts.append("Higher recall than precision — more positives captured, more false positives")

    # MCC warning for very low correlation
    if not np.isnan(mcc) and mcc < 0.2:
        parts.append("Low MCC — predictions have poor correlation with true labels")

    return "; ".join(parts)


def build_pipelines(random_state=42):
    # Use scaling for models that benefit from it
    scaler = StandardScaler()

    models = {}

    models["LogisticRegression"] = Pipeline([
        ("scaler", scaler),
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])

    models["DecisionTree"] = Pipeline([
        ("clf", DecisionTreeClassifier(random_state=random_state))
    ])

    models["KNN"] = Pipeline([
        ("scaler", scaler),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])

    models["GaussianNB"] = Pipeline([
        ("scaler", scaler),
        ("clf", GaussianNB())
    ])

    models["RandomForest"] = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state))
    ])

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([
            ("scaler", scaler),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state))
        ])
    else:
        print("xgboost not installed — skipping XGBoost model. Install via requirements.txt to enable it.")

    return models


def main():
    X_train, X_test, y_train, y_test = load_data(TRAIN_PATH, TEST_PATH)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    models = build_pipelines()
    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        # Fit
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Failed to fit {name}: {e}")
            continue

        print(f"Evaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test)
        metrics_row = {"Model": name}
        metrics_row.update(metrics)
        # Generate a concise observation about this model's performance
        try:
            obs = generate_observation(metrics)
        except Exception:
            obs = "Could not generate observation"
        metrics_row["Observation"] = obs
        print(f"Observation for {name}: {obs}")
        results.append(metrics_row)

    if len(results) == 0:
        print("No models were successfully trained.")
        return

    df_results = pd.DataFrame(results)
    out_path = os.path.join(DATA_DIR, "results_metrics.csv")
    df_results.to_csv(out_path, index=False)
    print(f"Saved metrics to {out_path}")
    print(df_results)


if __name__ == "__main__":
    main()
