import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
from settings import settings

# Import the existing fetching logic from the browser module to stay DRY
from ui_model_browser import _fetch_registered_models


def register_model_version(run_id: str, model_name: str) -> bool:
    """
    Registers the model produced in `run_id` under `model_name` in the MLflow Model Registry.
    It automatically fetches the `trainer_class` tag from the run and attaches it to the new
    Model Version.
    
    Returns True if registration and tagging was successful, False otherwise.
    """
    try:
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        
        # 1. Register the model using mlflow basic API (expecting the 'model' path)
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        
        # 2. Get trainer_class from the run
        run = client.get_run(run_id)
        trainer_class = run.data.tags.get("trainer_class")
        
        # 3. Explicitly set tag on the newly registered model version
        if trainer_class:
            client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key="trainer_class",
                value=trainer_class
            )
            st.success(f"✅ Successfully registered version {mv.version} of **'{model_name}'** and applied trainer class '{trainer_class}'.")
        else:
            st.warning(f"Successfully registered version {mv.version} of **'{model_name}'**, but no trainer class tag was found on the original run.")
            
        return True
    except Exception as e:
        st.error(f"Failed to register model: {e}")
        return False


def render_registration_form(run_id: str) -> None:
    """
    Renders an inline Streamlit UI block for registering a model from a finished run.
    Allows the user to select an existing model name to add a new version to,
    or type a completely new model name.
    """
    st.divider()
    st.subheader("💾 Register Model from this Run")
    st.markdown("Save this newly trained model to the MLflow Registry so you can serve it or retrain on top of it later.")
    
    # Check what existing models we have in the local MLflow registry
    registered_models = _fetch_registered_models()
    model_options = [rm.name for rm in registered_models] if registered_models else []
    
    col_input, col_button = st.columns([3, 1])
    
    with col_input:
        # If there are existing models, give them a dropdown but let them opt to write a new one
        if model_options:
            create_new = st.checkbox("Create New Model Name", value=False)
            
            if create_new:
                model_name = st.text_input("New Model Name", key="registry_new_model_name")
            else:
                model_name = st.selectbox("Select Existing Model", options=model_options, key="registry_existing_model_name")
                st.caption("A new Version will be appended to this existing Model.")
        else:
            # If no models exist in MLflow yet, force basic text input
            model_name = st.text_input("Model Name for Registry", key="registry_model_name")

    with col_button:
        # Align the button to the bottom of the input fields
        st.markdown("<br>" if not model_options or (model_options and not create_new) else "<br>" * 2, unsafe_allow_html=True)
        if st.button("Register Model", use_container_width=True):
            if not model_name:
                st.error("Please provide a model name.")
            else:
                with st.spinner("Registering model..."):
                    register_model_version(run_id=run_id, model_name=model_name)
