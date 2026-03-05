from trackers.mlflow_tracker import MLflowTracker
from trainer import run_training_pipeline
from preprocessors.preprocessing_pipeline import PreprocessingPipeline
from splitters.target_splitter import TargetSplitter
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

    if not model_config.get("model_uri"):
        st.error("No model selected. Please select a model from the registry.")
        return

    st.write("Initializing Experiment Tracker...")
    with MLflowTracker(
        experiment_name=eval_config["experiment_name"],
        run_name=eval_config["run_name"],
    ) as tracker:

        st.write(
            f"Instantiating Data Loader: `{data_config['loader_cls'].__name__}`..."
        )
        loader = instantiate_from_config(
            data_config["loader_cls"], data_config["loader_kwargs"]
        )

        # Build preprocessing pipeline from ordered stages
        st.write(f"Building Preprocessing Pipeline ({len(model_config['stages'])} stages)...")
        prep_stages = [
            instantiate_from_config(stage["cls"], stage["kwargs"])
            for stage in model_config["stages"]
        ]
        pipeline = PreprocessingPipeline(stages=prep_stages)

        # Build splitters
        splitter = instantiate_from_config(
            model_config["splitter_cls"], model_config["splitter_kwargs"]
        )
        target_splitter = TargetSplitter(target_column=model_config["target_column"])

        st.write(
            f"Loading existing model from MLflow: `{model_config['model_uri']}`..."
        )

        trainer = model_config["trainer_cls"](
            model_uri=model_config["model_uri"],
            tracker=tracker,
            metrics=eval_config["metrics_to_log"],
        )

        st.write("Dispatched pipeline directly to ReTrainer Backend. Running...")

        run_training_pipeline(
            tracker=tracker,
            data_loader=loader,
            preprocessing_pipeline=pipeline,
            splitter=splitter,
            target_splitter=target_splitter,
            model_trainer=trainer,
        )
