# Backpropagation from Andrej Karpathys amazing YouTube channel
# https://www.youtube.com/watch?v=VMj-3S1tku0

from __future__ import annotations
import math


# TODO: "Numpify" to speed everything up


class Node:
    """Implements a node in a computational graph with its value and global derivative.
    """
    
    def __init__(self, value:float, childs:tuple[Node]=()) -> None:
        """Initialize the node.

        Args:
            value (float): The value of this node.
            childs (tuple[Node], optional): A set of children nodes. Defaults to ().
        """
        self.value = value
        self.grad = 0
        self.childs = set(childs)
        self._backward = lambda: None      
        
    def __repr__(self):
        return f"Node(value={self.value:.4f}, grad={self.grad:.4f})"
    
    def backward(self) -> None:
        """Backpropagate through graph to calculate all partial derivatives with 
        respect to this node.
        """
        # Sort graph in topological order: https://en.wikipedia.org/wiki/Topological_sorting
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.childs:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Traverse the graph and apply the chain rule to each nodes gradient
        self.grad = 1.0
        for n in reversed(topo):
            n._backward()

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value + other.value, (self, other))
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value * other.value, (self, other))
        
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, n):
        out = Node(self.value**n, (self,))
        
        def _backward():
            self.grad += n * self.value**(n-1) * out.grad
        out._backward = _backward
        
        return out

    def tanh(self) -> Node:
        """Calculate output of the hyperbolic tangent (tanh) activation function and
        its backpropagated gradient.

        Returns:
            Node: Output of the hyperbolic tangent function.
        """
        x = self.value
        out = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Node(out, (self,))
        
        def _backward():
            self.grad += (1 - out.value**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self) -> Node:
        """Calculate output of the rectified linear unit (ReLU) activation function and
        its backpropagated gradient.

        Returns:
            Node: Output of the rectified linear unit function. 
        """
        out = Node(max(0, self.value), (self,))        
    
        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self) -> Node:
        """Calculate output of the sigmoid activation function and 
        its backpropagated gradient.

        Returns:
            Node: Output of the sigmoid function. 
        """
        out = Node(1 / (1 + math.exp(-self.value)), (self,))
        
        def _backward():
            self.grad += out.value * (1 - out.value) * out.grad
        out._backward = _backward
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
      
    def __rmul__(self, other):
        return self * other
        
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1

    def __neg__(self):
        return -1 * self