import streamlit as st


def render_data_config() -> dict:
    """Renders the UI for configuring DagsHub Data Ingestion."""
    st.header("1. Data Configuration")
    st.write("Configure connection to DagsHub Data Engine.")

    repo_name = st.text_input(
        "DagsHub Repository (owner/repo)", value="ofekmarks/my-first-repo"
    )
    datasource_name = st.text_input("Datasource Name", value="datasource")

    # We can expand this list in the future
    format_type = st.selectbox("Format Type", options=["dataframe"])

    target_column = st.text_input("Target Column Name", value="survived")

    return {
        "repo": repo_name,
        "datasource_name": datasource_name,
        "format_type": format_type,
        "target_column": target_column,
    }
