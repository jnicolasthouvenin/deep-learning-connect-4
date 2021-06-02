
from game import *
from encoder import *
from arena import *
from dataManager import *
from network import *

class Program:

    def __init__(self,the_game):
        self.the_game = the_game
        self.best_network = readNeuralNetwork("networks/best_network")
        self.new_network = NeuralNetwork([126,32,32,32,32,1],0.6)

    def select_move(self):
        return ARENA.select_move_net(self.best_network, self.the_game)

    def train_network(self):
        x,y = DATA_MANAGER.import_x_y_coupled_dataset("win_","loss_",200000)
        x_train,y_train,x_test,y_test = DATA_MANAGER.create_train_test_sets(x,y,10000)

        self.new_network.supervised_learning(x_train,y_train,x_test,y_test,10000,1,100,100)

    def set_network_structure(self,sizes,learning_rate):
        self.new_network = NeuralNetwork(sizes,learning_rate)
    