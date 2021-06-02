
import numpy as np

from game import *

class Encoder:

    def __init__(self):
        pass

    def encode_prediction(self,output_layer):
        """Returns the predicted class associated with the given output layer"""
        predicted_class = 0
        if output_layer < 0.5:
            predicted_class = 0
        else:
            predicted_class = 1
        return predicted_class

    def encode_board(self,game):
        """Return the encoding of the board that can be given to the neural network"""
        # create the 42 neurons for the current player
        turn = game.get_turn()
        turn_input = 0
        if turn == 2:
            turn = 0
            turn_input = np.zeros(42)
        else:
            turn_input = np.ones(42)
        
        board = np.ndarray.flatten(np.array(game.get_board()))

        # create the board for each player
        one_input = (board == 1).astype(int)
        two_input = (board == 2).astype(int)
        
        # concatenate all three inputs into one input of 126 elements
        final_input = np.hstack((one_input,two_input))
        final_input = np.hstack((final_input,turn_input))
        
        return final_input

ENCODER = Encoder()