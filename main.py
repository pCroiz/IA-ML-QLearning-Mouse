# Import
from qmaze import *
from agent import *
from game import *

#from keras import Sequential
#Dense, Activation,SGD , Adam, RMSprop,PReLU

# List of the action (encoded by a number)
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down'
}

# Create the maze
maze = np.array(np.load('maze_generator/maze.npy'))

# Creation of a Qmaze
qmaze = Qmaze(maze)

qmaze.draw()

# Creation of a Rat
rat = Rat(qmaze,actions_dict,initPosition=(0,0),eps=0.95)

# Creation of the neural network
neuralNetwork = NeuralNetwork(maze.size, maze.size, maze.size, len(actions_dict))

# Creation of a neural Rat
neuralRat = NeuralRat(qmaze,actions_dict,neuralNetwork)

# Load the neural model already trained
neuralRat.load_model('neuralNetwork/model.pth')

# Create the game (the board game)
game = Game(neuralRat,qmaze)

# Define the number of epochs
num_epochs = 20000

# Train the model
game.train(num_epochs,3000,True,displayEpsilon=True)

#neuralRat.save_model('neuralNetwork/model.pth')

game.playAnimation()

print("OK !")