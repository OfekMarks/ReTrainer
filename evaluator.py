from typing import Any, List, Optional
from metrics import calculate_metric
from trackers.tracker_interface import ExperimentTracker


class Evaluator:
    """
    Orchestrates the evaluation phase of the machine learning pipeline.
    Adheres to the Single Responsibility Principle: It only routes data
    between the pure mathematical metrics registry and the generic experiment tracker.
    """

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def log_metrics(
        self,
        metric_names: List[str],
        y_true: Any,
        y_pred: Any,
        step: Optional[int] = None,
    ) -> None:
        """
        Dynamically calculates requested metrics using registered pure-math functions
        and passes the resulting dictionaries to the tracked experiment.
        """
        metrics = {}
        for metric_name in metric_names:
            try:
                metric = calculate_metric(metric_name, y_true, y_pred)

                # Some metrics (like ROC) also return plots
                if isinstance(metric, tuple):
                    metric, plot = metric
                    self.tracker.log_figure(plot, f"plots/{metric_name}.png")

                metrics.update({metric_name: metric})

            except ValueError as e:
                print(f"Notice: Metric '{metric_name}' is skipped. Reason: {e}")

        self.tracker.log_metrics(metrics, step=step)
