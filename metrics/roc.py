from typing import Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, RocCurveDisplay

from . import register_metric


@register_metric("ROC")
def compute_roc(y_true: Any, y_pred: Any) -> Tuple[float, Figure]:
    """
    Computes ROC AUC score and generates an ROC Curve plot if possible.
    Note: y_pred should be probabilities for ROC.
    """
    auc_val = roc_auc_score(y_true, y_pred)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_pred, ax=ax)

    return auc_val, fig
