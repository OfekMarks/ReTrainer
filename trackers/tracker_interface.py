from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import importlib


class ExperimentTracker(ABC):
    """
    Abstract interface for experiment tracking systems (e.g. MLflow, Weights & Biases, ClearML).
    Follows the Dependency Inversion Principle (DIP) to allow swapping backends.
    """

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
