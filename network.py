
import numpy as np
import random

class Network(object):
    def __init__(self,sizes):
        """Constructor."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        """Return the output of the network if x is input."""
        for b, w in zip(self.biases, self.weights):
            b = np.squeeze(b) # convert b in the right format
            x = sigmoid(np.dot(w, x)+b)
        return x
    
    def backPropagate(self, x, y):
        

class NetworkBis():
    def __init__(self,sizes):
        """Constructor."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))