
import numpy as np
import random

class Arena(object):
    def __init__(self, redNet, yellowNet):
        print("ArenaConstructor")
        self.redNet = redNet
        self.yellowNet = yellowNet