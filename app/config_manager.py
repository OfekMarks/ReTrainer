import json
import streamlit as st


def get_state_keys() -> tuple:
    """Returns prefixes and exact keys that belong to the pipeline configuration."""
    return (
        "data_loader",
        "prep_stages",
        "splitter",
        "target_column",
        "metrics_to_log",
        "experiment_name",
        "run_name",
        "selected_registered_model",
        "selected_model_version",
    )


def export_config() -> str:
    """Serializes relevant session state to a JSON string."""
    config_state = {}
    valid_keys = get_state_keys()

    for k, v in st.session_state.items():
        if k.startswith(valid_keys):
            config_state[k] = v

    return json.dumps(config_state, indent=4)


def import_config(uploaded_file) -> None:
    """Loads a JSON file and injects it into the Streamlit session state."""
    try:
        config_state = json.load(uploaded_file)
        valid_keys = get_state_keys()

        for k, v in config_state.items():
            if k.startswith(valid_keys):
                st.session_state[k] = v

        st.success("Configuration loaded successfully.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to parse configuration: {e}")
