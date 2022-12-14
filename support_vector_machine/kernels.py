import numpy as np

class Linear:
    
    def __init__(self) -> None:
        """Initialize linear kernel.
        """
        pass

    def __call__(self, x:np.ndarray, y:np.ndarray=None) -> np.ndarray:
        """Calculate linear kernel matrix of two matrices.

        Args:
            x (np.ndarray): A feature matrix of shape (n_samples, n_features)
            y (np.ndarray, optional): An optional feature matrix of shape (n_samples, n_features). 
                If None, uses y = x. Defaults to None. 

        Returns:
            np.ndarray: The linear kernel matrix of shape (n_samples_x, n_samples_y).
        """
        if y is None:
            y = x
            
        return x @ y.T

class Polynomial:
    
    def __init__(self, degree:int=3, offset:int=0) -> None:
        """Initialize polynomial kernel.

        Args:
            degree (int, optional): Degree of the polynomial kernel. Defaults to 3.
            offset (int, optional): Constant offset added to inner product. Defaults to 0.
        """
        self.degree = degree
        self.offset = offset
        
    def __call__(self, x:np.ndarray, y:np.ndarray=None) -> np.ndarray:
        """Calculate polynomial kernel matrix of two matrices.

        Args:
            x (np.ndarray): A feature matrix of shape (n_samples, n_features)
            y (np.ndarray, optional): An optional feature matrix of shape (n_samples, n_features). 
                If None, uses y = x. Defaults to None.

        Returns:
            np.ndarray: The polynomial kernel matrix of shape (n_samples_x, n_samples_y).
        """
        if y is None:
            y = x
            
        return (self.offset + (x @ y.T))**self.degree

class RBF:
    # TODO: Implement RBF kernel, https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    # The easy way with two loops is in O(n^2), but there are different types of approximation.
    pass