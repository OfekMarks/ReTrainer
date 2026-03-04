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
    FLAVOR = None

    def __init__(
        self,
        metrics: List[str],
        tracker: ExperimentTracker,
        model_uri: str,
    ):
        self.metrics = metrics
        self.tracker = tracker
        self.model_uri = model_uri
        self.evaluator = Evaluator(tracker=tracker)

        self._validate_metrics()
        self.model = self.tracker.load_model(
            model_uri=self.model_uri, flavor=self.FLAVOR
        )

    def log_trained_model(self, name: str = "model") -> None:
        """Logs the model and tags the run with the trainer class for future auto-detection."""
        self.tracker.log_model(self.model, name=name, flavor=self.FLAVOR)
        self.tracker.set_tag("trainer_class", type(self).__name__)


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
