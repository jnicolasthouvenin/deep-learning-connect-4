
import numpy as np
import random

from game import *

class Arena(object):
    def __init__(self, redNet, yellowNet):
        print("ArenaConstructor")
        self.redNet = redNet
        self.yellowNet = yellowNet
    
    def playMove(self):
        print("playMove")
        inputLayer = boardToInputLayer(board)
        print("inputLayer : ",inputLayer)
        
        return 1