import numpy as np


class SquareLoss:
    """Implements the Square Loss and it's gradient.
    """
    
    def loss(self, y:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        """Calculate the squared error between each element.

        Args:
            y (np.ndarray): The target value.
            y_pred (np.ndarray): The input value.

        Returns:
            np.ndarray: The squared error.
        """
        return 0.5 * np.power(y - y_pred, 2)
    
    def gradient(self, y:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        """Calculate the gradient of the squared error between each element.

        Args:
            y (np.ndarray): The target value.
            y_pred (np.ndarray): The input value.

        Returns:
            np.ndarray: Gradient of the squared error.
        """ 
        return -1 * (y - y_pred)
    

class BinaryCrossEntropyLoss:
    """Implements the Binary Cross Entropy and it's gradient.
    """

    def loss(self, y:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        """Calculate the cross entropy between the target and the input probabilities.

        Args:
            y (np.ndarray): The target probabilities.
            y_pred (np.ndarray): The input probabilities.

        Returns:
            np.ndarray: The cross entropy of binary classifications.
        """
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
         
        return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def gradient(self, y:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        """Calculate the gradient of the binary cross entropy.

        Args:
            y (np.ndarray): The target probabilities.
            y_pred (np.ndarray): The input probabilities.

        Returns:
            np.ndarray: Gradient of the binary cross entropy.
        """ 
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10) 
        
        return -(y / y_pred) + (1 - y) / (1 - y_pred)