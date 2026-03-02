from typing import Any
from sklearn.metrics import precision_score

from . import register_metric


@register_metric("Precision")
def compute_precision(y_true: Any, y_pred: Any, average: str = "macro") -> float:
    """
    Computes the Precision score.
    Note: y_pred should be class labels (hard predictions).
    """
    score = precision_score(y_true, y_pred, average=average)

    return score
