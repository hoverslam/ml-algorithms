import random

from neural_network.autograd import Node


class Neuron:
    """Implements a single neuron.
    """
    
    def __init__(self, n_in:int, lin=False) -> None:
        """Initialize neuron.

        Args:
            n_in (int): Number of input features.
            lin (bool, optional): If True this neuron gives a linear output. Otherwise
                a tanh activation function is used. Defaults to False.
        """
        self.weights = [Node(random.uniform(-1, 1)) for _ in range(n_in)]
        self.bias = Node(0.01)
        self.lin = lin
        
    def __call__(self, x:list[Node]) -> Node:
        """A forward pass of this neuron.

        Args:
            x (list[Node]): A list of input features.

        Returns:
            Node: Output of this neuron.
        """
        out =  sum([xi * wi for xi, wi in zip(x, self.weights)]) + self.bias
    
        return out.tanh() if not self.lin else out
        
    def parameters(self) -> list[Node]:
        """Returns the parameters of this neuron.

        Returns:
            list[Node]: A list of all weights and the bias.
        """
        return self.weights + [self.bias]


class Layer:
    """Implements a layer containing multiple neurons.
    """
    
    def __init__(self, n_in:int, n_out:int, **kwargs) -> None:
        """_summary_

        Args:
            n_in (int): Number of input features.
            n_out (int): Number of output values.
        """
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]
    
    def __call__(self, x:list[Node]) -> list[Node]:
        """A forward pass of this layer.

        Args:
            x (list[Node]): A list of input features.

        Returns:
            list[Node]: Output values of this layer.
        """
        return [n(x) for n in self.neurons]
    
    def parameters(self) -> list[Node]:
        """Returns all parameters in this layer.

        Returns:
            list[Node]: A list of weights and biases in this layer.
        """
        return [p for n in self.neurons for p in n.parameters()]