from typing import Any
from sklearn.metrics import recall_score

from . import register_metric


@register_metric("Recall")
def compute_recall(y_true: Any, y_pred: Any, average: str = "macro") -> float:
    """
    Computes the Recall score.
    Note: y_pred should be class labels (hard predictions).
    """
    score = recall_score(y_true, y_pred, average=average)

    return score
