import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from run_models import build_pipelines, load_data, generate_observation, TRAIN_PATH, TEST_PATH, evaluate_model


@st.cache_data
def load_dataset():
    X_train, X_test, y_train, y_test = load_data(TRAIN_PATH, TEST_PATH)
    return X_train, X_test, y_train, y_test


def get_models():
    # return fresh pipelines so they can be trained independently
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
    train_all = st.sidebar.button("Train all models and evaluate")

    st.write("## DataSet used: Mobile price Classification")
    st.write("Train size:", X_train.shape, "Test size:", X_test.shape)

    # single-model training/evaluation removed — only batch evaluation available

    if train_all:
        st.info("Training all models — this may take a while")
        results = []
        progress = st.progress(0)
        total = len(models)
        i = 0
        for name, m in models.items():
            i += 1
            with st.spinner(f"Training {name} ({i}/{total})..."):
                try:
                    m.fit(X_train, y_train)
                except Exception as e:
                    st.warning(f"Failed to train {name}: {e}")
                    continue
            # evaluate using evaluate_model from run_models
            try:
                metrics = evaluate_model(m, X_test, y_test)
            except Exception as e:
                st.warning(f"Evaluation failed for {name}: {e}")
                metrics = {"Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan, "MCC": np.nan, "AUC": np.nan}

            obs = generate_observation(metrics)
            row = {"Model": name}
            row.update(metrics)
            row["Observation"] = obs
            results.append(row)
            progress.progress(int(i / total * 100))

        if len(results) == 0:
            st.warning("No models produced results.")
        else:
            df = pd.DataFrame(results)
            # reorder columns if present
            cols = [c for c in ["Model", "Accuracy", "Precision", "Recall", "F1", "MCC", "AUC", "Observation"] if c in df.columns]
            df = df[cols]
            st.write("## All models — metrics summary")
            st.dataframe(df.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}", "AUC": lambda v: "{:.4f}".format(v) if pd.notna(v) else "NaN"}))

            # per-model tabs
            tabs = st.tabs(list(df["Model"]))
            for tab, (_, r) in zip(tabs, df.iterrows()):
                with tab:
                    st.write(f"### {r['Model']}")
                    m = models[r['Model']]
                    # predictions
                    try:
                        y_pred = m.predict(X_test)
                    except Exception as e:
                        st.error(f"Could not predict for {r['Model']}: {e}")
                        continue
                    st.write("**Metrics**")
                    # Build metrics dict: convert numeric metrics to float, keep Observation as string
                    metrics_display = {}
                    for k, v in r.items():
                        if k == 'Model':
                            continue
                        if k == 'Observation':
                            metrics_display[k] = v
                        else:
                            metrics_display[k] = float(v) if (pd.notna(v)) else None
                    st.json(metrics_display)
                    st.write("**Observation**", r.get("Observation", ""))
                    st.write("**Confusion matrix**")
                    fig, ax = plt.subplots()
                    try:
                        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not draw confusion matrix: {e}")

            # save summary CSV
            out_path = TRAIN_PATH.replace('train.csv', 'results_metrics_streamlit.csv')
            df.to_csv(out_path, index=False)
            st.success(f"Saved summary to {out_path}")

    st.write("---")
    st.write("Hints: Use the sidebar to run batch training/evaluation for all models.")


if __name__ == "__main__":
    main()

