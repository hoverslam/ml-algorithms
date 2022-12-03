import numpy as np


def accuracy_score(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    """Calculate the percentage of correct predictions.

    Args:
        y_true (np.ndarray): Correct labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Fraction of correctly classified samples.
    """
    return np.mean(y_pred == y_true)