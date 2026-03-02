from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import importlib


class ExperimentTracker(ABC):
    """
    Abstract interface for experiment tracking systems (e.g. MLflow, Weights & Biases, ClearML).
    Follows the Dependency Inversion Principle (DIP) to allow swapping backends.
    """

    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self.run_name = run_name

    def __enter__(self):
        self.setup_experiment(self.experiment_name, self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()

    @abstractmethod
    def setup_experiment(
        self, experiment_name: str, run_name: Optional[str] = None
    ) -> None:
        """Initialize the tracking environment and optionally start a run."""
        pass

    @abstractmethod
    def log_params(self, **params) -> None:
        """Log hyperparameters to the active tracking run."""
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log key evaluation metrics to the active tracking run."""
        pass

    @abstractmethod
    def log_model(
        self, model: Any, name: str, flavor: str = "sklearn", **kwargs
    ) -> None:
        """Serialize and log a trained model artifact."""
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current tracking run."""
        pass
