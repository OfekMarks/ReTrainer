from trainers.sklearn_classification_trainer import SklearnClassificationTrainer
from sklearn.ensemble import RandomForestClassifier
from data_loaders.dagshub_loader import DagsHubDataEngineLoader
from trackers.mlflow_tracker import MLflowTracker
from preprocessors.default_tabular_preprocessor import DefaultTabularPreprocessor
from trainer import run_training_pipeline
import streamlit as st


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
            f"Connecting to DagsHub Data Engine (Repo: `{data_config['repo']}`)..."
        )
        dagshub_loader = DagsHubDataEngineLoader(
            repo=data_config["repo"],
            datasource_name=data_config["datasource_name"],
            format_type=data_config["format_type"],
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
            model = RandomForestClassifier(**model_config["params"])

        # 4. Instantiate Trainer
        trainer = SklearnClassificationTrainer(
            model=model,
            model_uri=model_config["model_uri"],
            tracker=tracker,
            metrics=eval_config["metrics_to_log"],
        )

        # 5. Execute Pipeline
        st.write("Dispatched pipeline directly to ReTrainer Backend. Running...")

        run_training_pipeline(
            tracker=tracker,
            data_loader=dagshub_loader,
            preprocessor=DefaultTabularPreprocessor(
                target_column=data_config["target_column"]
            ),
            model_trainer=trainer,
        )
