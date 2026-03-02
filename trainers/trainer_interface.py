from abc import ABC, abstractmethod
from typing import Any, List, Optional
from trackers.tracker_interface import ExperimentTracker
from evaluator import Evaluator


class TrainerInterface(ABC):
    """
    Abstract interface for model training and evaluation.
    Couples the specific model framework (e.g., Scikit-Learn, PyTorch) with the evaluator,
    as each ML task and framework may require different prediction logic and metrics.
    """

    VALID_METRICS = []

    def __init__(
        self,
        metrics: List[str],
        tracker: ExperimentTracker,
        model: Any = None,
        model_uri: Optional[str] = None,
    ):
        self.metrics = metrics
        self.tracker = tracker
        self.model = model
        self.model_uri = model_uri
        self.evaluator = Evaluator(tracker=tracker)

        self._validate_metrics()
        self.load_model()

    def load_model(self, flavor: str = "sklearn") -> None:
        """Loads the model from the tracker if a model_uri is provided."""
        if self.model_uri:
            print(
                f"Loading existing model for fine-tuning from MLflow: {self.model_uri}"
            )
            self.model = self.tracker.load_model(
                model_uri=self.model_uri, flavor=flavor
            )
        elif self.model is None:
            raise ValueError("Either 'model' or 'model_uri' must be provided.")

    def _validate_metrics(self):
        for metric in self.metrics:
            if metric not in self.VALID_METRICS:
                raise ValueError(
                    f"Invalid metric: {metric}. Valid metrics are: {self.VALID_METRICS}"
                )

    @abstractmethod
    def train_and_evaluate(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        metrics_to_log: List[str],
    ) -> None:
        """
        Trains the model, generates format-specific predictions, evaluates them,
        logs parameters and artifacts to the tracker.
        """
        pass
