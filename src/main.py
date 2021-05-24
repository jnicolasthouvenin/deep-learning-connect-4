import sys

import numpy as np
import pandas as pd
from game import *

if len(sys.argv) > 1:
    if sys.argv[1] == "1":
        print("[INFO] Loading file network_tp...\n")
        from network_tp import *
    else:
        print("[INFO] Loading file network_tp2...\n")
        from network_tp2 import *
else:
    print("[INFO] Loading file network...\n")
    from network import *

def encodePred(label_pred):
    indMax = np.argmax(label_pred)
    if indMax == 0:
        return np.array([1.,0.,0.])
    elif indMax == 1:
        return np.array([0.,1.,0.])
    else:
        return np.array([0.,0.,1.])

def encodeLabel(label):
    if label == 1.0:
        label = np.array([1.,0.,0.])
    elif label == 0.0:
        label = np.array([0.,1.,0.])
    else:
        label = np.array([0.,0.,1.])
    return label

def score(label_pred,turn):
    indMax = np.argmax(label_pred)
    value = 0.
    if indMax == 0:
        if turn == 1:
            value = label_pred[0]
        else:
            value = label_pred[0]
    elif indMax == 1:
        value = 0.
    else:
        if turn == 1:
            value = -label_pred[2]
        else:
            value = label_pred[2]
    return value

def encodeBoard(board):
    board = np.array(board)
    board = np.array([board[0:7,5],board[0:7,4],board[0:7,3],board[0:7,2],board[0:7,1],board[0:7,0]],dtype = float).flatten()
    for i in range(len(board)):
        if board[i] == 2.:
            board[i] = -1.
    return board

def playMove(neural_network,turn):
    values = np.zeros(7)
    for col in range(7):
        if game.board_at(col,5) == 0: # if the network can play in col
            # copy state of the game
            gameCopy = game.copy_state()
            # play the move
            gameCopy.place(col)
            # evaluate the resulting board
            input = encodeBoard(gameCopy.get_board())
            print("input")
            print(input)
            label_pred = neural_network.forward(input)
            print("for col",col)
            print(label_pred)
            values[col] = score(label_pred,turn)
            #print(input)
            #print(values[col])
        else:
            values[col] = -1
    
    #print(values)
    move = np.argmax(values)
    game.place(move)
    pass

def main1():
    dataset = pd.read_csv("../data/c4_game_database.csv")

    print(dataset.head())

    dataset = dataset.to_numpy()

    #input = dataset[0,0:41]
    #label = dataset[0,42]

    #print(input)
    #print(label)

    nw_c4 = NeuralNetwork([42,16,16,16,3])

    #playMove(nw_c4)

    """print(game._board)
    game.place(0)
    game.place(0)
    game.place(1)
    game.place(2)
    game.place(3)
    game.place(4)
    game.place(5)
    game.place(6)
    print(game.get_board())
    board = np.array(game.get_board())
    print(board)
    print(board[0:6,0])
    input = np.array([board[0:7,5],board[0:7,4],board[0:7,3],board[0:7,2],board[0:7,1],board[0:7,0]],dtype = float).flatten()
    print(input)
    print(len(input))
    print(dataset[0,])"""

    trainLen = 300000
    testLen = 76000
    #trainLen = 1000
    #testLen = 100

    for i in range(trainLen):
        input = dataset[i,0:42]
        #print(len(input))
        #print(input)
        #print(dataset[i,42])
        label = encodeLabel(dataset[i,42])
        #print(label)
        nw_c4.train(input,label)

    goodPred = 0
    preds = 0

    for i in range(trainLen,trainLen+testLen):
        input = dataset[i,0:42]
        label_pred = encodePred(nw_c4.forward(input))
        label = encodeLabel(dataset[i,42])
        if (label_pred == label).all():
            goodPred += 1
        preds += 1

    view = Connect4Viewer(game=game)
    view.initialize()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if game.get_win() is None:
                    game.place(pygame.mouse.get_pos()[0] // SQUARE_SIZE)
                    print(game.get_board()[0][0])
                    if game.get_win() is None:
                        turn = game.get_turn()
                        playMove(nw_c4,turn)
                else:
                    game.reset_game()

    pygame.quit()

    print("classification rate =",goodPred/preds)

    """nw = NeuralNetwork([3,2,3])

    print("\n[INFO] Learning to predict [1,0,0]")
    print("[INFO] Training...\n")

    for i in range(20):
        y_pred = nw.train([1,0,0],[1,0,0])
        print("y_pred =",y_pred," - iteration :",i)


    print("\n[INFO] Terminated without errors\n")"""

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

    #arena = Arena(redNet,yellowNet)

    #arena.playMove()

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

main1()