
import numpy as np
import random

from game import *

class Arena(object):
    def __init__(self, redNet, yellowNet):
        self.redNet = redNet
        self.yellowNet = yellowNet
    
    def playMove():
        inputLayer = boardToInputLayer(board)