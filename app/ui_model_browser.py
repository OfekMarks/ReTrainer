import datetime
import pandas as pd
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from typing import Dict, Any, List, Optional
from settings import settings
from operator import attrgetter


@st.cache_data(ttl=60)
def _fetch_registered_models() -> Optional[List[RegisteredModel]]:
    """Fetches all registered models from the MLflow Model Registry (cached for 60s)."""
    try:
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        models = client.search_registered_models()

        if not models:
            st.warning("No registered models found in the MLflow Model Registry.")
            return 

        return models
    except Exception as e:
        st.error(f"⚠️ Could not connect to MLflow Registry: {e}")
        return None


@st.cache_data(ttl=60)
def _fetch_model_versions(model_name: str):
    """Fetches all versions for a given registered model (cached for 60s)."""
    try:
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            st.warning(f"No versions found for model '{model_name}'.")
            return None

        return versions
    except Exception as e:
        st.error(f"⚠️ Could not fetch versions for '{model_name}': {e}")
        return None


def _build_alias_map(registered_model: RegisteredModel) -> Dict[str, List[str]]:
    """Builds a {version_number: [alias_names]} map from the RegisteredModel's aliases dict."""
    alias_map: Dict[str, List[str]] = {}
    for alias_name, version_num in getattr(registered_model, "aliases", {}).items():
        alias_map.setdefault(str(version_num), []).append(alias_name)

    return alias_map


def _render_model_selector(registered_models: List[RegisteredModel]) -> RegisteredModel:
    """Renders a selectbox for choosing a registered model and returns the selected RegisteredModel."""
    model_names = list(map(attrgetter("name"), registered_models))
    selected_name = st.selectbox("Registered Model", options=model_names)

    selected_rm = next(rm for rm in registered_models if rm.name == selected_name)
    if getattr(selected_rm, "description", None):
        st.caption(selected_rm.description)

    return selected_rm


def _render_versions_table(versions_sorted: List[ModelVersion], alias_map: Dict[str, List[str]]) -> None:
    """Builds and displays a table of model version metadata."""
    version_df = pd.DataFrame([{
        "Version": v.version,
        "Aliases": ", ".join(alias_map.get(str(v.version), [])) or "—",
        "Status": v.status,
        "Created": datetime.datetime.fromtimestamp(
            v.creation_timestamp / 1000
        ).strftime("%Y-%m-%d %H:%M"),
        "Run ID": v.run_id if v.run_id else "—",
    } for v in versions_sorted])

    st.markdown("##### Available Versions")
    st.dataframe(
        version_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Version": st.column_config.TextColumn("Version", width="small"),
            "Aliases": st.column_config.TextColumn("Aliases", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Created": st.column_config.TextColumn("Created", width="medium"),
            "Run ID": st.column_config.TextColumn("Run ID", width="medium"),
        },
    )


def _render_version_picker(versions_sorted: List, alias_map: Dict[str, List[str]]) -> str:
    """Renders a selectbox for picking a specific model version and returns the selected version string."""
    version_options = [str(v.version) for v in versions_sorted]
    selected_version = st.selectbox(
        "Select Version to Retrain",
        options=version_options,
        format_func=lambda v: f"v{v}" + (
            f" ({', '.join(alias_map[v])})"
            if alias_map.get(v)
            else ""
        ),
    )

    return selected_version


def _resolve_selection(selected_name: str, selected_version: str, versions_sorted: List) -> Dict[str, Any]:
    """Builds the model URI and looks up the trainer class tag from the selected version."""
    model_uri = f"models:/{selected_name}/{selected_version}"

    selected_mv = next(v for v in versions_sorted if str(v.version) == selected_version)
    trainer_class = selected_mv.tags.get("trainer_class")

    st.success(f"🔗 Selected: `{model_uri}` — Trainer: **{trainer_class}**")

    return {"model_uri": model_uri, "trainer_class": trainer_class}


def render_model_browser() -> Optional[Dict[str, Any]]:
    """
    Renders an interactive model browser that lists all MLflow registered models
    and their versions, letting the user pick one for retraining.
    Returns a dict with 'model_uri' and 'trainer_class' (if tagged), or None.
    """
    st.subheader("🗂️ Select Model for Retraining")

    registered_models = _fetch_registered_models()
    if not registered_models:
        return None

    selected_rm = _render_model_selector(registered_models)
    alias_map = _build_alias_map(selected_rm)

    versions = _fetch_model_versions(selected_rm.name)
    if not versions:
        return None

    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)

    _render_versions_table(versions_sorted, alias_map)
    selected_version = _render_version_picker(versions_sorted, alias_map)

    return _resolve_selection(selected_rm.name, selected_version, versions_sorted)
