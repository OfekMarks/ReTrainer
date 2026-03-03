import streamlit as st
from typing import Dict, Any
from trainers import AVAILABLE_TRAINERS
from preprocessors import AVAILABLE_PREPROCESSORS
import class_registry


def render_model_config() -> Dict[str, Any]:
    """Renders the UI for configuring ML Architecture, Preprocessing natively."""
    st.header("2. ML Pipeline Components")

    # --- PREPROCESSOR ---
    st.subheader("Data Preprocessor")
    prep_name = st.selectbox(
        "Preprocessor Implementation", options=list(AVAILABLE_PREPROCESSORS.keys())
    )
    prep_cls = AVAILABLE_PREPROCESSORS[prep_name]
    prep_kwargs = class_registry.render_dynamic_params(
        prep_cls, key_prefix="preprocessor"
    )

    st.markdown("---")

    # --- TRAINER ---
    st.subheader("ML Task Trainer")
    trainer_name = st.selectbox(
        "Trainer Implementation", options=list(AVAILABLE_TRAINERS.keys())
    )
    trainer_cls = AVAILABLE_TRAINERS[trainer_name]

    # We still need a way to let them decide if they are doing scratch or finetuning:
    strategy = st.radio(
        "Training Strategy", ["Train from Scratch", "Fine-Tune Existing Model"]
    )

    model_config = {
        "prep_cls": prep_cls,
        "prep_kwargs": prep_kwargs,
        "trainer_cls": trainer_cls,
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
        # In the future we can utilize class registry here for native Scikit-Learn models!
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
