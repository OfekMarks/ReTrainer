import streamlit as st
import mlflow
import mlflow.exceptions
import os
import tempfile
from mlflow.artifacts import download_artifacts


def render_results(run_id: str):
    """
    Fetches and displays metrics, parameters, and plots from an MLflow run.
    """
    st.header("📊 Pipeline Results")

    try:
        run = mlflow.get_run(run_id)
        data = run.data

        # --- Metrics Section ---
        st.subheader("Metrics")
        if data.metrics:
            cols = st.columns(len(data.metrics))
            for col, (name, value) in zip(cols, data.metrics.items()):
                col.metric(label=name, value=f"{value:.4f}")
        else:
            st.info("No metrics logged for this run.")

        st.divider()

        # --- Plots Section ---
        st.subheader("Plots")
        # List artifacts to find plots
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, "plots")

        if artifacts:
            with tempfile.TemporaryDirectory() as tmp_dir:
                for artifact in artifacts:
                    if artifact.path.endswith((".png", ".jpg", ".jpeg")):
                        local_path = download_artifacts(
                            run_id=run_id, artifact_path=artifact.path, dst_path=tmp_dir
                        )
                        st.image(
                            local_path,
                            caption=os.path.basename(artifact.path),
                            use_container_width=True,
                        )
        else:
            st.info("No plots found in the artifacts.")

        st.divider()

        # --- Parameters Section ---
        with st.expander("View Parameters & Tags"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Parameters**")
                if data.params:
                    st.json(data.params)
                else:
                    st.write("None")
            with col2:
                st.write("**Tags**")
                if data.tags:
                    st.json(data.tags)
                else:
                    st.write("None")

    except mlflow.exceptions.MlflowException as e:
        st.error(f"MLflow error loading results: {e}")
    except (TypeError, AttributeError, ValueError) as e:
        st.error(f"Error parsing result data: {e}")
