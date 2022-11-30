import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles


def generate_data(n: int = 1000, lin_sep: bool = True) -> list[np.ndarray]:
    """Generate 2-dimensional data set.

    Args:
        n (int, optional): Number of observations. Defaults to 1000.
        lin_sep (bool, optional): Generate linearly seperable data. Defaults to True.

    Returns:
        list[np.ndarray]: List of features (X) and corresponding labels (y).
    """
    if lin_sep:
        X, y = make_blobs(n, centers=2)
    else:  
        X, y = make_circles(n, noise=0.1, factor=0.2)
        
    y = np.where(y == 0, -1, y)
    
    return (X, y)

def split_data(X: np.ndarray, y:np.ndarray, train_size: int = 0.8) -> list[np.ndarray]:
    """Split data into training and test set.

    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        train_size (int, optional): Percentage of the dataset to include in the train split. Defaults to 0.8.

    Returns:
        list[np.ndarray]: List containing train-test split.
    """
    ind = int(len(y) * train_size)
    
    return(X[:ind, ], X[ind:,], y[:ind], y[ind:]) 

def plot_data(X: np.ndarray, y: np.ndarray) -> None:
    """Plot data points with their corresponding class.

    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
    """
    _, ax = plt.subplots()
    ax.scatter(X.T[0], X.T[1], s=5, c=y)
    plt.show()