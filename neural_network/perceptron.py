import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from utils.losses import BinaryCrossEntropyLoss
from utils.metrics import accuracy_score
 
    
class MLP:
    """Implements a Multi Layer Perceptron for binary classification.
    """
       
    def __init__(self, input_dim:int, hidden_dims:tuple[int], activation:str="sigmoid", 
                 lr:float=1e-3, epochs:int=100) -> None:
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
    
    def fit(self, X:np.ndarray, y:np.ndarray, plot_history:bool=False) -> None:
        """Fit the model according to the given training data.

        Args:
            X (np.ndarray): The input samples.
            y (np.ndarray): The class values (0 or 1).
            plot_history (bool, optional): Plot the loss over all epochs.
        """
        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
                
        # Initialize parameter
        self.init_params()

        # Training loop
        hist = {"epoch":[], "loss":[]}        
        pbar = trange(self.epochs)
        for epoch in pbar:
            # Shuffle data
            s = np.arange(n_samples)
            np.random.shuffle(s)
            X_s, y_s = X[s], y[s]
        
            loss = 0
            for idx in range(X_s.shape[0]):
                # Forward propagation
                outputs = self.forward(X_s[idx])
                loss += self.loss.loss(y_s[idx], outputs[-1])
                
                # Gradient descent (update weights)
                input = X_s[idx].reshape(self.batch_size, n_features)
                gradient = self.loss.gradient(y_s[idx], outputs[-1])
                self.gradient_descent(input, outputs, gradient)
            
            loss_epoch = np.sum(loss / X_s.shape[0])    
            hist["epoch"].append(epoch + 1)
            hist["loss"].append(loss_epoch)
            pbar.set_description(f"Loss: {loss_epoch:.4f}")
            
        if plot_history:
            self.plot_loss(hist)
    
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
        # Backpropagate starting with the output layer    
        grads = []    
        grads.append(gradient_of_loss * self.activation[1](outputs[-1]))
        
        # Backpropgate through remaining layers (in reversed order)
        for i in reversed(range(len(outputs[:-1]))):
            layer = np.dot(grads[-1], self.weights[i+1].T) * self.activation[1](outputs[i])
            grads.append(layer)

        # Update weights and bias of input layer
        grads = grads[::-1]    # reverse list of gradients
        self.weights[0] -= self.lr * np.dot(input.T, grads[0])
        self.bias[0] -= self.lr * np.sum(grads[0], axis=0)
        
        # Update weights and bias of remaining layers
        for i in range(len(grads)-1):
            self.weights[i+1] -= self.lr * np.dot(outputs[i].T, grads[i+1])
            self.bias[i+1] -= self.lr * np.sum(grads[i+1], axis=0)
 
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
        
    def plot_loss(self, hist:dict) -> None:
        """Plot loss curve of training.

        Args:
            hist (dict): Loss of each epoch.
        """
        epoch = hist["epoch"]
        loss = hist["loss"]
        
        plt.title("Loss curve of training")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(epoch, loss, marker=".")
        plt.show() 