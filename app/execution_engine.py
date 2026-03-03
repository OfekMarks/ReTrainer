from sklearn.ensemble import RandomForestClassifier
from trackers.mlflow_tracker import MLflowTracker
from trainer import run_training_pipeline
import streamlit as st


def instantiate_from_config(cls, kwargs: dict):
    """
    Instantiates a class by dynamically unpacking and validating its kwargs
    through its Pydantic ConfigModel if one exists.
    """
    if hasattr(cls, "ConfigModel"):
        validated_config = cls.ConfigModel(**kwargs).model_dump()
        return cls(**validated_config)
    return cls(**kwargs)


def execute_pipeline(data_config: dict, model_config: dict, eval_config: dict):
    """
    Instantiates the SOLID architecture components based on UI data and
    dispatches the core training pipeline.
    """

    # 1. Instantiate Tracker
    st.write("Initializing Experiment Tracker...")
    with MLflowTracker(
        experiment_name=eval_config["experiment_name"],
        run_name=eval_config["run_name"],
    ) as tracker:

        # 2. Instantiate Data Loader
        st.write(
            f"Instantiating Data Loader: `{data_config['loader_cls'].__name__}`..."
        )
        loader = instantiate_from_config(
            data_config["loader_cls"], data_config["loader_kwargs"]
        )

        # 3. Handle Model Selection
        if model_config["strategy"] == "Fine-Tune Existing Model":
            st.write(
                f"Fetching existing model from MLflow: `{model_config['model_uri']}`..."
            )
            model = None
        else:
            st.write(
                f"Instantiating new {model_config['model_class_str']} from scratch..."
            )
            model = model_config["model_class_str"](**model_config["params"])

        # 4. Instantiate Preprocessor (which isn't dependent on tracker/model)
        prep = instantiate_from_config(
            model_config["prep_cls"], model_config["prep_kwargs"]
        )

        # 5. Instantiate Trainer
        trainer = model_config["trainer_cls"](
            model=model,
            model_uri=model_config["model_uri"],
            tracker=tracker,
            metrics=eval_config["metrics_to_log"],
        )

        # 5. Execute Pipeline
        st.write("Dispatched pipeline directly to ReTrainer Backend. Running...")

        run_training_pipeline(
            tracker=tracker,
            data_loader=loader,
            preprocessor=prep,
            model_trainer=trainer,
        )
