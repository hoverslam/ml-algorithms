import numpy as np
from tqdm import trange

from neural_network.nn import Layer
from neural_network.autograd import Node


class MLP:
    """Implements a Multi Layer Perceptron (MLP) for simple multivariate regression.
    """
    
    def __init__(self, n_in:int, n_hidden:tuple, n_out:int=1, lr:float=1e-3) -> None:
        """Initialize MLP.

        Args:
            n_in (int): Number of input features.
            n_hidden (tuple): Number of hidden layers.
            n_out (int, optional): Number of outputs. Only single output right now!
                Defaults to 1.
            lr (float, optional): Learning rate for parameter updates. Defaults to 1e-3.
        """
        self.layers = self.create_network(n_in, n_hidden, n_out)
        self.lr = lr
    
    def __call__(self, x:list[float]) -> Node:
        """Forward pass of a single observation.

        Args:
            x (list[float]): A vector of input features.

        Returns:
            Node: Output value.
        """
        for layer in self.layers:
            x = layer(x)
            
        return x[0]
    
    def parameters(self) -> list[Node]:
        """Return all parameters in this MLP.

        Returns:
            list[Node]: A list of all weights and biases.
        """
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self) -> None:
        """Sets all gradients to zero.
        """
        for p in self.parameters():
            p.grad = 0.0
    
    def create_network(self, n_in:int, n_hidden:tuple, n_out:int) -> list[Layer]:
        """Create all layers need for the given input.

        Args:
            n_in (int): Number of input features.
            n_hidden (tuple): Number of hidden layers.
            n_out (int): Number of output values.

        Returns:
            list[Layer]: A list of all layers.
        """
        # Input layer
        layers = [Layer(n_in, n_hidden[0])]
        
        # Hidden layers
        for i, n in enumerate(n_hidden[1:]):
            layers.append(Layer(n_hidden[i], n))
            
        # Output layer
        layers.append(Layer(n_hidden[-1], n_out, lin=True))
        
        return layers
    
    def fit(self, X:np.ndarray, y:np.ndarray, epochs:int=1000) -> dict:
        """Fit the model according to the given training data.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The target values.
            epochs (int, optional): Number of training epochs. Defaults to 1000.

        Returns:
            dict: Dictionary containing loss per epoch.
        """      
        # Convert to python list
        X, y = X.tolist(), y.tolist()
        
        # Training loop
        history = {"epoch":[], "loss":[]}
        pbar = trange(epochs)
        for i in pbar:
            # Forward pass
            logits = [self(x) for x in X]
            loss = sum((target - pred)**2 for target, pred in zip(y, logits)) / len(logits)
            
            # Backward pass
            self.zero_grad()
            loss.backward()
            
            # Update weights
            for p in self.parameters():
                p.value -= self.lr * p.grad
            
            # Save training loss    
            history["epoch"].append(i+1)
            history["loss"].append(loss.value)
            
            # Update progress bar
            pbar.set_description(f"Loss={loss.value:.4f}")
            
        return history
    
    def predict(self, X:np.ndarray) -> list[float]:
        """Make predictions on samples in X.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            list[float]: List of predictions.
        """
        predictions = []
        for x in X.tolist():
            predictions.append(self(x).value)
            
        return predictions