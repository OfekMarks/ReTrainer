import streamlit as st
from typing import Dict, Any
from preprocessors import AVAILABLE_PREPROCESSORS
import class_registry
from ui_model_browser import render_model_browser
import trainers


def render_model_config() -> Dict[str, Any]:
    """Renders the UI for configuring ML Pipeline: Preprocessing + Model selection from MLflow."""
    st.header("2. ML Pipeline Components")

    st.subheader("Data Preprocessor")
    prep_name = st.selectbox(
        "Preprocessor Implementation", options=list(AVAILABLE_PREPROCESSORS.keys())
    )
    prep_cls = AVAILABLE_PREPROCESSORS[prep_name]
    prep_kwargs = class_registry.render_dynamic_params(
        prep_cls, key_prefix="preprocessor"
    )

    st.markdown("---")

    browser_result = render_model_browser()

    st.markdown("---")

    model_config = {
        "prep_cls": prep_cls,
        "prep_kwargs": prep_kwargs,
        "trainer_cls": getattr(trainers, browser_result["trainer_class"]),
        "model_uri": browser_result["model_uri"],
        "params": {},
    }

    return model_config
