import streamlit as st


from data_loaders import AVAILABLE_LOADERS
import class_registry


def render_data_config() -> dict:
    """Renders the UI for configuring Data Ingestion dynamically."""
    st.header("1. Data Configuration")
    st.write("Configure connection to Data Engine.")

    loader_name = st.selectbox(
        "Data Loader Implementation", options=list(AVAILABLE_LOADERS.keys())
    )
    loader_cls = AVAILABLE_LOADERS[loader_name]

    st.markdown("---")
    st.subheader(f"Configure {loader_name}")

    loader_kwargs = class_registry.render_dynamic_params(
        loader_cls, key_prefix="data_loader"
    )

    return {
        "loader_cls": loader_cls,
        "loader_kwargs": loader_kwargs,
    }
