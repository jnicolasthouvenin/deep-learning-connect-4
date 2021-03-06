
import random
import numpy as np
from game import *
from network import *

def boardToInputLayer(board):
    layer = np.empty(0,dtype=int)
    for raw in range(7):
        for line in range(6):
            state = board[raw][line] # state of the square (raw,line) : 0 for empty, 1 for yellow, 2 for red
            layer = np.append(layer,state)
    return layer

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

"""
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
    print("a = ",a)
"""
print("end")