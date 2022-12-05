import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from utils.losses import BinaryCrossEntropyLoss
from utils.metrics import accuracy_score


class SLP:
    """Implements a (Single Layer) Perceptron for linear binary classification.
    """
    
    def __init__(self, lr:float=1e-2, tol:float=1e-5, max_iter:int=100) -> None:
        """Initialize Percetpron.

        Args:
            lr (float, optional): The learning rate defines the amount by which 
                the error effects the weight update. Defaults to 1e-2.
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
        best_solution = [1e10, self.weights]       
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
                print(f"Optimal solution found after {epoch+1} iteration(s).")
                break
            
            # Update "pocket" if solution is better than the current one
            if best_solution[0] > error:
                best_solution = [error, self.weights]
            
        if error > self.tol:
            print("Maximum number of iterations reached. Best solution is used.")
            self.weights = best_solution[1]
 
        # Plot final (i.e. best found) decision boundary
        if n_features == 2 and show:
            x_values, y_values = self.decision_boundary()
            plt.clf()
            plt.title(f"Iteration: {epoch+1}")
            plt.xlim(lim_min[0], lim_max[0])
            plt.ylim(lim_min[1], lim_max[1])
            plt.scatter(X.T[0], X.T[1], s=5, c=y)
            plt.plot(x_values, y_values, c="black")
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
    
    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Caclulate the mean accuracy on the given data and labels.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The true labels for X.

        Returns:
            float: The mean accuracy of the predictions.
        """
        return accuracy_score(y, self.predict(X))
    
    
class MLP:
    """Implements a Multi Layer Perceptron for binary classification.
    """
       
    def __init__(self, input_dim:int, hidden_dims:tuple[int], activation:str="sigmoid", 
                 lr:float=1e-3, epochs:int=1000) -> None:
        """Initialize MLP.

        Args:
            input_dim (int): Number of features.
            hidden_dims (tuple[int]): Number of neurons per layer.
            activation (str, optional): Activation function for the hidden layer 
                (currently only one is available). Defaults to "sigmoid".
            lr (float, optional): The learning rate effects the amount by which 
                the weights are updated. Defaults to 1e-3.
            epochs (int, optional): Number of training iterations. Defaults to 1000.
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 1    # TODO: Implement multi-class prediction
        self.lr = lr
        self.epochs = epochs
        self.batch_size = 1    # TODO Implement mini-batch gradient descent
        
        activation_functions = {
            # [0] = function, [1] = it's derivative
            "sigmoid": [lambda x: 1 / (1.0 + np.exp(-x)), lambda x: x * (1.0 - x)]
        }
        self.activation = activation_functions[activation]
        self.loss = BinaryCrossEntropyLoss()
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Fit the model according to the given training data.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The class values (0 or 1).
        """
        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
                
        # Initialize parameter
        self.init_params()

        # Training loop
        loss_hist = {}        
        pbar = trange(self.epochs)
        for epoch in pbar:
            loss = 0
            for idx in range(n_samples):
                # Forward propagation
                outputs = self.forward(X[idx])
                loss += self.loss.loss(y[idx], outputs[-1])
                
                # Gradient descent (update weights)
                input = X[idx].reshape(self.batch_size, n_features)
                gradient = self.loss.gradient(y[idx], outputs[-1])
                self.gradient_descent(input, outputs, gradient)
                
            loss_hist[epoch+1] = loss / n_samples
            pbar.set_description(f"Loss: {np.sum(loss / n_samples):.4f}")
    
    def forward(self, X:np.ndarray) -> list[np.ndarray]:
        """One forward pass from the input to the output layer.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            list[np.ndarray]: The raw output of each layer.
        """
        output_per_layer = []

        # Pass through input layer
        output = np.dot(X, self.weights[0]) + self.bias[0]
        output = self.activation[0](output)
        output_per_layer.append(output)
        
        # Pass through hidden layers
        for i in range(len(self.hidden_dims) - 1):
            output = np.dot(output, self.weights[i+1]) + self.bias[i+1]
            output = self.activation[0](output)
            output_per_layer.append(output)
            
        # Pass through output layer
        output = np.dot(output, self.weights[-1]) + self.bias[-1]
        output = self.activation[0](output)
        output_per_layer.append(output)
    
        return output_per_layer
                   
    def gradient_descent(self, input:np.ndarray, outputs:list[np.ndarray], 
                         gradient_of_loss: np.ndarray) -> None:
        """Backpropagate and update weights according to the gradients.

        Args:
            input (np.ndarray): The input samples.
            outputs (list[np.ndarray]): The raw output of each layer.
            gradient_of_loss (np.ndarray): Gradient of the loss from output layer.
        """
        # TODO: Make loop
        # Backpropagate
        d_l1 = gradient_of_loss * self.activation[1](outputs[1])
        d_l0 = np.dot(d_l1, self.weights[1].T) * self.activation[1](outputs[0])

        # Update weights and bias
        self.weights[0] -= self.lr * np.dot(input.T, d_l0)
        self.bias[0] -= self.lr * np.sum(d_l0, axis=0)
        self.weights[1] -= self.lr * np.dot(outputs[0].T, d_l1)
        self.bias[1] -= self.lr * np.sum(d_l1, axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on samples in X.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray:  Class labels for samples in X.
        """
        output = self.forward(X)[-1]    # Only the output layer is used for predictions
        
        return (1 * (output > 0.5)).reshape(-1)
    
    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Caclulate the mean accuracy on the given data and labels.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The true labels for X.

        Returns:
            float: The mean accuracy of the predictions.
        """
        return accuracy_score(y, self.predict(X))
       
    def init_params(self):
        """Initialize weights and bias.
        """
        self.weights = []
        self.bias = []
        
        # Initialize input layer
        dim = (self.input_dim, self.hidden_dims[0])
        self.weights.append(np.random.random(dim))
        self.bias.append(np.full((1, self.hidden_dims[0]), 1e-3))
        
        # Initialize hidden layer
        for i, n in enumerate(self.hidden_dims[1:]):
            dim = (self.weights[i].shape[1], n)
            self.weights.append(np.random.random(dim))
            self.bias.append(np.full((1, n), 1e-3))
            
        # Initialize output layer
        dim = (self.weights[-1].shape[1], self.output_dim)
        self.weights.append(np.random.random(dim))
        self.bias.append(np.full((1, self.output_dim), 1e-3))