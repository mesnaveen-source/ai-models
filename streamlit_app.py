import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from run_models import build_pipelines, load_data, generate_observation, TRAIN_PATH, TEST_PATH


@st.cache_data
def load_dataset():
    X_train, X_test, y_train, y_test = load_data(TRAIN_PATH, TEST_PATH)
    return X_train, X_test, y_train, y_test


@st.cache_data
def get_models():
    return build_pipelines()


def evaluate_and_report(model, X_test, y_test):
    # fit is done before calling this in UI flow
    y_pred = model.predict(X_test)
    try:
        y_score = model.predict_proba(X_test)
    except Exception:
        y_score = None

    # reuse evaluate_model from run_models by importing it would require circular import; use minimal metrics here
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    labels = np.unique(np.concatenate([y_test, y_pred]))
    if y_score is not None:
        try:
            if len(labels) > 2:
                y_true_bin = np.zeros((len(y_test), len(labels)))
                for i, lab in enumerate(labels):
                    y_true_bin[:, i] = (np.array(y_test) == lab).astype(int)
                metrics["AUC"] = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
            else:
                metrics["AUC"] = roc_auc_score(y_test, y_score[:, 1])
        except Exception:
            metrics["AUC"] = np.nan
    else:
        metrics["AUC"] = np.nan

    return metrics, y_pred


def main():
    st.title("Mobile Price Classification — Model Explorer")

    X_train, X_test, y_train, y_test = load_dataset()
    models = get_models()

    st.sidebar.header("Options")
    model_choice = st.sidebar.selectbox("Choose model", list(models.keys()))
    retrain = st.sidebar.button("Train / Retrain selected model")

    st.write("## Dataset")
    st.write("Train size:", X_train.shape, "Test size:", X_test.shape)

    model = models[model_choice]

    if retrain:
        with st.spinner(f"Training {model_choice}..."):
            model.fit(X_train, y_train)
        st.success("Training complete")

    st.write("## Selected Model: ", model_choice)

    if st.button("Evaluate selected model (requires prior training) "):
        try:
            metrics, y_pred = evaluate_and_report(model, X_test, y_test)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            return

        obs = generate_observation(metrics)

        st.write("### Metrics")
        st.json(metrics)
        st.write("**Observation:**", obs)

        st.write("### Confusion matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        st.pyplot(fig)

        if not np.isnan(metrics.get("AUC", np.nan)):
            st.write("AUC:", float(metrics["AUC"]))

    st.write("---")
    st.write("Hints: Use the sidebar to select a model and click 'Train / Retrain' then evaluate.")


if __name__ == "__main__":
    main()
