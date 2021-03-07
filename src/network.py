
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

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def update_mini_batch(self, x, y, eta):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backprop(x, y)
            gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(self, x, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activationsList = [np.array(x)] # list to store all the activations, layer by layer
        zList = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            b = np.squeeze(b)
            z = np.dot(w, activation)+b
            zList.append(z)
            activation = sigmoid(z)
            activationsList.append(np.array(activation))
        # backward pass
        delta = self.cost_derivative(activationsList[-1], y) * sigmoid_prime(zList[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = calculateGradientW(activationsList[-2],delta,len(zList[-1]))
        print("gradient_b[-1] = ",gradient_b[-1])
        print("gradient_w[-1] = ",gradient_w[-1])
        
        for l in range(2, self.num_layers):
            print("for")
            z = zList[-l]
            print("len(z) : ",len(z))
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            gradient_b[-l] = delta
            #gradient_w[-l] = np.dot(delta, activationsList[-l-1].transpose())
            gradient_w[-l] = calculateGradientW(activationsList[-l-1],delta,len(z))
            print("gradient_b[",-l,"] = ",gradient_b[-l])
            print("gradient_w[",-l,"] = ",gradient_w[-l])

        return (gradient_b, gradient_w)
    
def calculateGradientW(a,delta,repeat):
    print("---------|calculateGradient|------------")
    print("a : ",a)
    print("delta : ",delta)
    print("len(a) : ",len(a))
    aBuffer = a.transpose()
    aBuffer = np.tile(a,repeat).reshape(repeat,len(a)).transpose()
    """if three:
        aBuffer = np.tile(a,repeat).reshape(repeat,len(a)).transpose()
        #aBuffer = np.array([a,a,a]).transpose()
    else :
        aBuffer = np.array([a,a]).transpose()"""
    print("aBuffer : ",aBuffer)
    g_w = np.multiply(aBuffer,delta).transpose()
    print("---------|end-calculateGradient|------------")
    return g_w

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))