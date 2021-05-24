
import numpy as np
import random

class NeuralNetwork(object):

    def __init__(self,sizes):
        """Initialize the neural network."""
        self.nb_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    def forward(self, input):
        """Computes the output of the network, when fowarding into it the given input x."""
        for bias, weights in zip(self.biases, self.weights):
            bias = np.squeeze(bias) # convert b in the right format
            weightsTimesInput = np.dot(weights, input)
            input = sigmoid(weightsTimesInput+bias)
        return input

    def cost_derivative(self, output_activations, y):
        print("output_activations")
        print(output_activations)
        print("y")
        print(y)
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

    def backprop(self, input, output):
        """Computes the gradients for biases and weights to correct to neural network"""

        # initialize gradients
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # pass the input through the neural network

        # creating the structures for storing the z and activations for each layer
        zList = [] # z vectors for each layer
        activation = input # the first activation vector consists in the input vector
        activationsList = [np.array(input)] # activations vectors for each layer
        # iterating on each layer
        for biases, weights in zip(self.biases, self.weights): # get the biases and weights for the current layer
            biases = np.squeeze(biases)
            z = np.dot(weights, activation)+biases # compute z (using the previous activation vector)
            zList.append(z) # store the value z of the layer
            activation = sigmoid(z) # compute the activation from z and the sigmoid function
            activationsList.append(np.array(activation)) # store the activation of the layer

        # backpropagate using the computed activations
        delta = self.cost_derivative(activationsList[-1], output) * sigmoid_prime(zList[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = calculateGradientW(activationsList[-2],delta,len(zList[-1]))
        print("gradient_b[-1] = ",gradient_b[-1])
        print("gradient_w[-1] = ",gradient_w[-1])
        
        for l in range(2, self.nb_layers):
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