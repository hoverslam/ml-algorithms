import numpy as np

from utils.metrics import r2_score


class LinearRegression:
    """Implements a linear regression using OLS estimator.
    """

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Fit a regression line to the training data.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The target values.
        """
        n_samples, _ = X.shape
        y = y.reshape((n_samples, 1))
        
        # Add bias to feature matrix
        input = np.hstack((np.ones((1, n_samples)).T, X))

        # OLS estimator: https://en.wikipedia.org/wiki/Ordinary_least_squares
        gram_matrix = np.dot(input.T, input)
        self.weights = np.linalg.inv(gram_matrix).dot(input.T).dot(y)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """Make predictions on a given feature matrix.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted values.
        """
        # Add bias to feature matrix
        input = np.hstack((np.ones((1, X.shape[0])).T, X))
        
        return np.dot(input, self.weights).reshape(-1)
    
    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Caclulate the R2 score for a given feature matrix and the true target values.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The target values.

        Returns:
            float: R squared (R^2).
        """
        return r2_score(y, self.predict(X))


class RidgeRegression:
    """Implements Ridge regression using OLS estimator.
    """
    
    def __init__(self, alpha:float=1.0) -> None:
        """Initialize Ridge regression.

        Args:
            alpha (float, optional): L2 term, controlling regularization strength. 
                Defaults to 1.0.
        """
        self.alpha = alpha
        
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Fit a regression line to the training data.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The target values.
        """
        n_samples, n_features = X.shape
        y = y.reshape((n_samples, 1))
        
        # Add bias to feature matrix
        input = np.hstack((np.ones((1, n_samples)).T, X))

        # OLS estimator: https://en.wikipedia.org/wiki/Ridge_regression
        gram_matrix = np.dot(input.T, input) + self.alpha * np.eye(n_features+1)
        self.weights = np.linalg.inv(gram_matrix).dot(input.T).dot(y)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """Make predictions on a given feature matrix.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted values.
        """
        # Add bias to feature matrix
        input = np.hstack((np.ones((1, X.shape[0])).T, X))
        
        return np.dot(input, self.weights).reshape(-1)
    
    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Caclulate the R2 score for a given feature matrix and the true target values.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The target values.

        Returns:
            float: R squared (R^2).
        """
        return r2_score(y, self.predict(X))