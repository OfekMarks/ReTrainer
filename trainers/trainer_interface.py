from abc import ABC, abstractmethod
from typing import Any, List
from trackers.tracker_interface import ExperimentTracker
from evaluator import Evaluator


class TrainerInterface(ABC):
    """
    Abstract interface for model training and evaluation.
    Couples the specific model framework (e.g., Scikit-Learn, PyTorch) with the evaluator,
    as each ML task and framework may require different prediction logic and metrics.
    """

    VALID_METRICS = []

    def __init__(self, metrics: List[str], tracker: ExperimentTracker, model: Any):
        self.metrics = metrics
        self.tracker = tracker
        self.model = model
        self.evaluator = Evaluator(tracker=tracker)

        self._validate_metrics()

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
