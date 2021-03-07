
import random
import numpy as np
from game import *
from network import *
from arena import *

def main1():

    print("hello")

    print(board)
    place(0)
    place(1)
    place(0)
    print(board)

    nw = Network([3,2,3])

    #inputLayer = boardToInputLayer(board)
    inputLayer = [1,0,0]
    print("inputLayer = ",inputLayer)

    outputLayer = nw.feedforward(inputLayer)
    print("outputLayer = ",outputLayer)

    print("try backpropagation")

    gradient_b, gradient_w = nw.backpropagate(inputLayer,[0,1,0])

    print("FIN BACKPROGATION")

    print("gradient_b : ",gradient_b)
    print("gradient_w : ",gradient_w)

    a = np.array([1,2])
    b = np.tile(a,3).reshape(3,2)
    print("b : ",b)

    """a = np.array([4,5])
    newA = np.array([a,a,a]).transpose()
    print("operation sur a")
    print(newA)
    delta = np.array([1,2,3])
    print("gradient_w")
    gradient_w = np.multiply(newA,delta).transpose()
    print(gradient_w)
    x1 = np.arange(6.0).reshape((3,2)).transpose()
    x2 = np.arange(3.0)
    print(x1)
    print(x2)
    print(np.multiply(x1,x2).transpose())"""

    print("end")

def main2():
    redNet = Network([42,16,7])
    yellowNet = Network([42,16,7])

    arena = Arena(redNet,yellowNet)

    arena.playMove()

    print("endMain2")

"""def sandPot():
    print("weights")
    print(nw.weights)
    print("biases")
    print(nw.biases)

    weights = np.array([[1,2,3],[4,5,6]])
    bi = np.array([1,2])
    a = np.random.rand(3)
    print("a = ",a)
    print("weights = ",weights)
    newA = np.dot(weights,a) + bi
    print("newA = ",newA)


    sizes = [3,2,3]
    biases  = [np.random.randn(1, y) for y in sizes[1:]]
    weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
    print("biases = ",biases)
    print("weights = ",weights)
    print("a = ",a)

    for b,w in zip(biases,weights):
        b = np.squeeze(b)
        print("for")
        print("b = ",b)
        print("w = ",w)
        a = np.dot(w,a) + b
        print("a = ",a)"""

main2()