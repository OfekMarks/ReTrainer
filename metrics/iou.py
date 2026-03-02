from typing import Any
from sklearn.metrics import jaccard_score

from . import register_metric


@register_metric("IoU")
def compute_iou(y_true: Any, y_pred: Any, average: str = "macro") -> float:
    """
    Computes the Intersection over Union (Jaccard Index) score.
    Note: For flat classification mapping (or masks), Jaccard Index is equivalent to IoU.
    """
    score = jaccard_score(y_true, y_pred, average=average)

    return score
