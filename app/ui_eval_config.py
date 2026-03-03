import streamlit as st


def render_evaluation_config() -> dict:
    """Renders the UI for configuring Metrics and Tracker Configuration."""
    st.header("3. Evaluation & Tracking")

    st.subheader("Metrics")
    metrics_to_log = st.multiselect(
        "Select Metrics to Log",
        options=["ROC", "Recall", "Precision", "IoU"],
        default=["ROC", "Recall", "Precision"],
    )

    st.subheader("Experiment Tracker")
    experiment_name = st.text_input(
        "Experiment Name", value="cpu_retraining_experiment"
    )
    run_name = st.text_input("Run Name", value="retraining_my-first-repo_datasource")

    return {
        "metrics_to_log": metrics_to_log,
        "experiment_name": experiment_name,
        "run_name": run_name,
    }
