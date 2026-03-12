import streamlit as st
import sys
import os

# Ensure the parent directory is in sys.path so we can import our platform architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_data_config import render_data_config
from ui_model_config import render_model_config
from ui_eval_config import render_evaluation_config
from execution_engine import execute_pipeline
from ui_model_register import render_registration_form

st.set_page_config(page_title="ReTrainer Pipeline Dashboard", layout="wide")


def main():
    st.title("🚀 ReTrainer Pipeline Dashboard")
    st.markdown(
        "Easily configure and instantly launch training pipelines into our SOLID architecture backend."
    )

    data_config = render_data_config()
    st.divider()
    model_config = render_model_config()
    st.divider()
    eval_config = render_evaluation_config()

    if st.button("Launch Training Pipeline", type="primary", use_container_width=True):
        st.divider()
        with st.spinner("Executing Pipeline Sequence..."):
            try:
                run_id = execute_pipeline(data_config, model_config, eval_config)
                st.session_state["finished_run_id"] = run_id
                st.success("✅ Training Pipeline Completed Successfully!")
                st.balloons()
            except Exception as e:
                st.error(f"Pipeline Failed: {e}")

    if "finished_run_id" in st.session_state:
        render_registration_form(st.session_state["finished_run_id"])

if __name__ == "__main__":
    main()
