import json
import streamlit as st


STATE_PREFIX = "cfg_"


def export_config() -> str:
    """Serializes relevant session state to a JSON string."""
    config_state = {}

    for k, v in st.session_state.items():
        if k.startswith(STATE_PREFIX):
            config_state[k] = v

    return json.dumps(config_state, indent=4)


def import_config(uploaded_file) -> None:
    """Loads a JSON file and injects it into the Streamlit session state."""
    if uploaded_file is None:
        return

    try:
        config_state = json.load(uploaded_file)

        if not isinstance(config_state, dict):
            st.error("Invalid configuration format: Expected a JSON object.")
            return

        for k, v in config_state.items():
            if k.startswith(STATE_PREFIX):
                st.session_state[k] = v

        st.success("Configuration loaded successfully.")
        st.rerun()
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse configuration: {e}")
    except (TypeError, AttributeError) as e:
        st.error(f"Invalid configuration structure: {e}")
