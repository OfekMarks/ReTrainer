import streamlit as st
from typing import Dict, Any


def render_model_config() -> Dict[str, Any]:
    """Renders the UI for configuring Model Architecture and Fine-Tuning."""
    st.header("2. Model & Trainer Configuration")

    task_type = st.selectbox(
        "ML Task Trainer", options=["SklearnClassificationTrainer"]
    )

    st.subheader("Model Source")
    strategy = st.radio(
        "Training Strategy", ["Train from Scratch", "Fine-Tune Existing Model"]
    )

    model_config = {
        "task_type": task_type,
        "strategy": strategy,
        "model_uri": None,
        "model_class_str": None,
        "params": {},
    }

    if strategy == "Fine-Tune Existing Model":
        model_config["model_uri"] = st.text_input(
            "MLflow Model URI",
            value="models:/survival-classifier/1",
            help="e.g. models:/<model_name>/<version> or runs:/<run_id>/model",
        )
    else:
        model_config["model_class_str"] = st.selectbox(
            "Model Architecture", options=["RandomForestClassifier"]
        )

        st.write("Hyperparameters")
        n_estimators = st.slider(
            "n_estimators", min_value=10, max_value=500, value=100, step=10
        )
        max_depth = st.slider("max_depth", min_value=1, max_value=50, value=10)

        model_config["params"] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
        }

    return model_config
