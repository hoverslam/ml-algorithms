import numpy as np
import matplotlib.pyplot as plt


class SLP:
    """Implements a Single Layer Perceptron for linear binary classification.
    """
    
    def __init__(self, lr:float=1e-3, tol:float=1e-5, max_iter:int=100) -> None:
        """Initialize Percetpron.

        Args:
            lr (float, optional): The learning rate defines the amount by which 
                the error effects the weight update. Defaults to 1e-3.
            tol (float, optional): Stopping criterion for a good enough solution. Defaults to 1e-5.
            max_iter (int, optional): Maximum number of epochs over the training data. Defaults to 100.
        """
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        
        # Heaviside step function
        self.activation_function = lambda x: np.where(x <= 0, 0, 1)
        
    def fit(self, X: np.ndarray, y:np.ndarray, show:bool=False) -> None:
        """Fit the Perceptron according to the given training data.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The class values (0 or 1).
            show (bool, optional): Plot the movement of the decision boundary during training. Defaults to False.
        """
        # Initialize values
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        
        # Add bias to feature matrix and weight vector
        input = np.hstack((np.ones((1, n_samples)).T, X))
        self.weights = np.insert(self.weights, 0, 0.1)       

        # Training loop
        lim_min = X.min(axis=0) - 1      
        lim_max = X.max(axis=0) + 1        
        for epoch in range(self.max_iter):
            for i in range(n_samples):
                output = np.dot(input[i], self.weights)
                self.weights += self.lr * (y[i] - self.activation_function(output)) * input[i]
            
            # Plot training               
            if n_features == 2 and show:            
                x_values, y_values = self.decision_boundary()
                plt.clf()
                plt.title(f"Iteration: {epoch+1}")
                plt.xlim(lim_min[0], lim_max[0])
                plt.ylim(lim_min[1], lim_max[1])
                plt.scatter(X.T[0], X.T[1], s=5, c=y)
                plt.plot(x_values, y_values, c="black")
                plt.pause(1e-5) 
            
            # Stop training when error is smaller than tolerance
            error = np.mean(np.abs(y - self.predict(X)))
            if error < self.tol:                
                print(f"Solution found after {epoch+1} iteration(s).")
                break
            
        if error > self.tol:
            print("Maximum number of iterations reached. No solution found!")
 
        plt.show()            
            
    def predict(self, X:np.ndarray) -> np.ndarray:
        """Perform classification on samples in X.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: Class labels for samples in X.
        """
        # Add bias to feature matrix
        input = np.hstack((np.ones((1, X.shape[0])).T, X))

        return self.activation_function(np.dot(input, self.weights))
    
    def decision_boundary(self) -> list[np.ndarray]:
        """Calculate the decision boundary for current weights.

        Returns:
            list[np.ndarray]: Two points defining the decision boundary.
        """
        x = np.array([-100, 100])
        slope = (-self.weights[0] / self.weights[2]) / (self.weights[0] / self.weights[1])
        intercept = -self.weights[0] / self.weights[2]
        y = slope * x + intercept
        
        return (x, y)