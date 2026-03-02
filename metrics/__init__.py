from typing import Callable, Dict, List, Any

# Central registry mapping metric names to their corresponding logging functions
METRICS_REGISTRY: Dict[str, Callable] = {}


def register_metric(name: str):
    """
    A decorator designed to register a metric logging function under a specific name.

    Args:
        name: The name of the metric (e.g., 'ROC', 'Recall')
    """

    def decorator(func: Callable):
        METRICS_REGISTRY[name] = func
        return func

    return decorator


def get_available_metrics() -> List[str]:
    """Returns a list of all registered metric names."""
    return list(METRICS_REGISTRY.keys())


def calculate_metric(metric_name: str, y_true: Any, y_pred: Any) -> Any:
    """Retrieves the logging function for a given metric."""
    if metric_name not in METRICS_REGISTRY:
        raise ValueError(
            f"Metric '{metric_name}' is not registered. Available metrics: {get_available_metrics()}"
        )

    metric_calculation_func = METRICS_REGISTRY[metric_name]
    metric = metric_calculation_func(y_true, y_pred)

    return metric


from . import roc, recall, precision, iou
