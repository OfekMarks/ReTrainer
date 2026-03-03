import streamlit as st
from typing import Any, Dict, Type
from pydantic import BaseModel


def render_dynamic_params(cls: Type[Any], key_prefix: str) -> Dict[str, Any]:
    """
    Uses Streamlit to dynamically render form inputs based on a class's
    Pydantic ConfigModel.
    Returns a dictionary of the captured user inputs.
    """
    config_model = getattr(cls, "ConfigModel", None)

    if not config_model or not issubclass(config_model, BaseModel):
        # No configuration required for this class
        return {}

    user_inputs = {}

    # In Pydantic v2, fields are stored in model_fields
    for field_name, field_info in config_model.model_fields.items():
        # Get description and default
        desc = field_info.description if field_info.description else field_name
        default_val = field_info.default or ""

        # Generate a unique key for streamlit state
        st_key = f"{key_prefix}_{cls.__name__}_{field_name}"

        # Infer type from annotation
        field_type = field_info.annotation

        # Determine the appropriate Streamlit widget
        if field_type is bool:
            user_inputs[field_name] = st.checkbox(
                desc, value=bool(default_val), key=st_key
            )
        elif field_type is int:
            user_inputs[field_name] = st.number_input(
                desc,
                value=int(default_val or 0),
                step=1,
                key=st_key,
            )
        elif field_type is float:
            user_inputs[field_name] = st.number_input(
                desc,
                value=float(default_val or 0.0),
                format="%.4f",
                key=st_key,
            )
        else:
            # Fallback to text input for strings and everything else
            user_inputs[field_name] = st.text_input(
                desc,
                value=str(default_val),
                key=st_key,
            )

    return user_inputs
