import os

import importlib

import mlflow

from typing import Any, Dict, List, Optional
from settings import settings
from metrics import get_metric_func


def setup_mlflow(experiment_name: str, run_name: Optional[str] = None):
    """

    Set up MLflow tracking URI and authentication, and optionally start a named run.

    Uses the DagsHub token for authentication.
    """

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name=experiment_name)

    # End any dangling active runs just in case

    if mlflow.active_run():

        mlflow.end_run()

    if run_name:

        mlflow.start_run(run_name=run_name)


def log_model(model: Any, name: str, flavor: str = "sklearn", **kwargs):
    """

    Log a model to MLflow. Supported flavors by default: sklearn, pytorch, xgboost.
    """

    try:

        module = importlib.import_module(f"mlflow.{flavor}")

        if flavor == "sklearn":

            # Scikit-learn has a specific security warning for pickle. 'skops' is the modern standard.

            kwargs["serialization_format"] = "skops"

        module.log_model(model, name, **kwargs)

    except ModuleNotFoundError:

        raise ValueError(f"Unsupported flavor: {flavor}")


def load_native_model(
    model_name: str, version: str = "latest", flavor: str = "sklearn"
):
    """

    Load a native model (e.g., PyTorch nn.Module, Scikit-learn estimator) from MLflow model registry

    so it can be fine-tuned or evaluated.
    """

    model_uri = f"models:/{model_name}/{version}"

    module = importlib.import_module(f"mlflow.{flavor}")

    return module.load_model(model_uri)


def log_evaluation_metrics(
    metric_names: List[str], y_true: Any, y_pred: Any, step: Optional[int] = None
):
    """

    Dynamically evaluates and logs a list of requested metrics by calling

    their registered functions. Will cleanly skip unregistered/incompatible metrics.
    """
    for metric_name in metric_names:

        try:

            logger_func = get_metric_func(metric_name)

            logger_func(y_true, y_pred, step=step)

        except ValueError as e:

            print(f"Notice: Metric '{metric_name}' is skipped. Reason: {e}")
