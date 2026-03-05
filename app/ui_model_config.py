import streamlit as st
from typing import Dict, Any, List
from preprocessors import AVAILABLE_PREPROCESSORS
from splitters import AVAILABLE_SPLITTERS
import class_registry
from ui_model_browser import render_model_browser
import trainers


def _render_preprocessing_stages() -> List[Dict[str, Any]]:
    """Renders an ordered list of preprocessing stages with add/remove controls."""
    st.subheader("Data Preprocessor")

    if "prep_stages" not in st.session_state:
        st.session_state.prep_stages = []

    # Add stage button
    col_select, col_add = st.columns([3, 1])
    with col_select:
        new_stage_name = st.selectbox(
            "Add Preprocessing Stage",
            options=list(AVAILABLE_PREPROCESSORS.keys()),
            key="new_stage_selector",
        )
    with col_add:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add", use_container_width=True):
            st.session_state.prep_stages.append(new_stage_name)
            st.rerun()

    # Render current stages
    stages_config = []
    for i, stage_name in enumerate(st.session_state.prep_stages):
        stage_cls = AVAILABLE_PREPROCESSORS[stage_name]

        col_label, col_up, col_down, col_remove = st.columns([6, 1, 1, 1])
        with col_label:
            st.markdown(f"**{i + 1}. {stage_name}**")
        with col_up:
            if i > 0 and st.button("⬆", key=f"up_{i}"):
                st.session_state.prep_stages[i], st.session_state.prep_stages[i - 1] = (
                    st.session_state.prep_stages[i - 1], st.session_state.prep_stages[i]
                )
                st.rerun()
        with col_down:
            if i < len(st.session_state.prep_stages) - 1 and st.button("⬇", key=f"down_{i}"):
                st.session_state.prep_stages[i], st.session_state.prep_stages[i + 1] = (
                    st.session_state.prep_stages[i + 1], st.session_state.prep_stages[i]
                )
                st.rerun()
        with col_remove:
            if st.button("🗑️", key=f"remove_{i}"):
                st.session_state.prep_stages.pop(i)
                st.rerun()

        stage_kwargs = class_registry.render_dynamic_params(
            stage_cls, key_prefix=f"prep_stage_{i}"
        )
        stages_config.append({"cls": stage_cls, "kwargs": stage_kwargs})

        if i < len(st.session_state.prep_stages) - 1:
            st.markdown("---")

    if not st.session_state.prep_stages:
        st.info("No preprocessing stages added. Raw data will be passed directly to splitting.")

    return stages_config


def _render_splitter_config() -> Dict[str, Any]:
    """Renders the train/test splitter and target column configuration."""
    st.subheader("Train/Test Split")
    splitter_name = st.selectbox(
        "Split Strategy", options=list(AVAILABLE_SPLITTERS.keys())
    )
    splitter_cls = AVAILABLE_SPLITTERS[splitter_name]
    splitter_kwargs = class_registry.render_dynamic_params(
        splitter_cls, key_prefix="splitter"
    )

    st.subheader("Target Column")
    target_column = st.text_input(
        "Column to predict", value="survived", key="target_column"
    )

    return {
        "splitter_cls": splitter_cls,
        "splitter_kwargs": splitter_kwargs,
        "target_column": target_column,
    }


def render_model_config() -> Dict[str, Any]:
    """Renders the UI for configuring ML Pipeline: Preprocessing + Splitting + Model."""
    st.header("2. ML Pipeline Components")

    stages_config = _render_preprocessing_stages()

    st.markdown("---")

    splitter_config = _render_splitter_config()

    st.markdown("---")

    browser_result = render_model_browser()

    model_config = {
        "stages": stages_config,
        "splitter_cls": splitter_config["splitter_cls"],
        "splitter_kwargs": splitter_config["splitter_kwargs"],
        "target_column": splitter_config["target_column"],
        "trainer_cls": getattr(trainers, browser_result["trainer_class"]),
        "model_uri": browser_result["model_uri"],
    }

    return model_config
