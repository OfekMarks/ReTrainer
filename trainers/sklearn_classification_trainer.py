from typing import Any, Dict, List
from trackers.tracker_interface import ExperimentTracker
from evaluator import Evaluator
from trainers.trainer_interface import TrainerInterface


class SklearnClassificationTrainer(TrainerInterface):
    """
    Concrete trainer for Scikit-Learn classification tasks.
    Knows how to generate class probabilities and hard predictions for classification metrics.
    """

    VALID_METRICS = ["Precision", "Recall", "ROC"]

    def train_and_evaluate(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
    ) -> None:
        self.model.fit(X_train, y_train)

        y_pred_classes = self.model.predict(X_test)

        if len(self.model.classes_) == 2:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = self.model.predict_proba(X_test)

        metrics_proba = [m for m in self.metrics if "ROC" in m.upper()]
        metrics_classes = [m for m in self.metrics if m not in metrics_proba]

        if metrics_proba:
            self.evaluator.log_metrics(metrics_proba, y_test, y_pred_proba, step=1)
        if metrics_classes:
            self.evaluator.log_metrics(metrics_classes, y_test, y_pred_classes, step=1)

        self.tracker.log_model(self.model, name="model", flavor="sklearn")
