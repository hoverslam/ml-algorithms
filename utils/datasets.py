import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_circles, make_moons


def generate_data(n, type:str="linear") -> list[np.ndarray]:
    """Generate 2-dimensional data set.

    Args:
        n (_type_): Number of observations.
        type (str, optional): Type of data (linear, clusters, circles, moons). Defaults to "linear".

    Returns:
        list[np.ndarray]: List of features (X) and corresponding label (y).
    """
    
    match type:
        case "linear": 
            X, y = make_blobs(n, centers=2)
        case "clusters":
            X, y = make_classification(n, n_features=2, n_redundant=0, class_sep=3)
        case "circles": 
            X, y = make_circles(n, noise=0.1, factor=0.2)
        case "moons": 
            X, y = make_moons(n, noise=0.1)
        case other:
            print("No such data type available!")

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