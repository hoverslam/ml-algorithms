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

def mean_squared_error(y_true:np.ndarray, y_pred:np.ndarray, root:bool=False) -> float:
    """Calculate the average squared difference between the estimated values and the actual value.

    Args:
        y_true (np.ndarray): Correct labels.
        y_pred (np.ndarray): Predicted labels.
        root (bool, optional): If set to True the RMSE is calculated. Defaults to False.

    Returns:
        float: Mean squared error (MSE). If root=True the root mean squared error (RMSE) is returned.
    """
    if root:
        return np.sqrt(np.mean((y_pred - y_true)**2))
    else: 
        return np.mean((y_pred - y_true)**2)

def mean_absolute_error(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    """Calculate arithmetic average of the absolute errors.

    Args:
        y_true (np.ndarray): Correct labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Mean absolute error (MAE).
    """
    return np.mean(np.abs(y_pred - y_true))

def r2_score(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    """Calcualte the coefficient of determination (R squared).

    Args:
        y_true (np.ndarray): Correct labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: R squared (R^2).
    """
    rss = np.sum((y_true - y_pred)**2)
    tss = np.sum((y_true - np.mean(y_true))**2)
    
    return 1 - (rss / tss)