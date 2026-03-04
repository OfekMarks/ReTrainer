import importlib

from typing import Any, Dict, List, Optional

import mlflow

from trackers.tracker_interface import ExperimentTracker
from settings import settings

from matplotlib.figure import Figure


class MLflowTracker(ExperimentTracker):
    """
    Concrete implementation of the ExperimentTracker interface using MLflow.
    Follows Single Responsibility Principle by handling only MLflow-specific interactions.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize the setup. Will default to pydantic settings if URI isn't provided.
        """
        super().__init__(experiment_name, run_name)
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri

    def setup_experiment(
        self, experiment_name: str, run_name: Optional[str] = None
    ) -> None:
        """
        Set up MLflow tracking URI, select experiment, and optionally start a named run.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name=experiment_name)

        # Ensure no dangling runs remain from previous unexpected crashes
        if mlflow.active_run():
            mlflow.end_run()

        if run_name:
            mlflow.start_run(run_name=run_name)

    def log_params(self, **params) -> None:
        """Log hyperparameters to MLflow."""
        mlflow.log_params(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log computed metrics directly to MLflow context natively."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(
        self, model: Any, name: str, flavor: str = "sklearn", **kwargs
    ) -> None:
        """
        Log a model to MLflow utilizing the specific module flavor.
        """
        try:
            module = importlib.import_module(f"mlflow.{flavor}")

            # Scikit-learn has a specific security warning for pickle. 'skops' is the modern standard.
            if flavor == "sklearn":
                kwargs["serialization_format"] = "skops"

            module.log_model(model, name, **kwargs)
        except ModuleNotFoundError:
            raise ValueError(f"Unsupported flavor: {flavor}")

    def load_model(self, model_uri: str, flavor: str = "sklearn", **kwargs) -> Any:
        """
        Load a native model (e.g., PyTorch nn.Module, Scikit-learn estimator) from MLflow memory
        so it can be fine-tuned or evaluated.
        """
        try:
            module = importlib.import_module(f"mlflow.{flavor}")
            return module.load_model(model_uri, **kwargs)
        except ModuleNotFoundError:
            raise ValueError(f"Unsupported flavor: {flavor}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the active MLflow run."""
        mlflow.set_tag(key, value)

    def end_run(self) -> None:
        """Safely end the MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """
        Log a local file or directory as an artifact to the active MLflow run.
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_figure(self, fig: Figure, artifact_path: str) -> None:
        """
        Log a matplotlib figure as an artifact to the active MLflow run.
        """
        mlflow.log_figure(fig, artifact_path)
