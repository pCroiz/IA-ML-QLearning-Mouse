from qmaze import *
import numpy as np 
class Rat(object):

    def __init__(self,maze:Qmaze,possibleAction:enumerate,initPosition:tuple=(0,0)) -> None:

        # Get the maze
        self._maze = maze

        # Get the init Position
        self._initPosition = initPosition
        
        ### Initialisation of Q ####
        self._possibleAction = possibleAction

        # Get the number of actions
        nbrAction = len(possibleAction)

        # Get the size of the maze
        nrows,ncols = self._maze.shape

        # Initialize the Q matrix with random value between 0 and 1
        self._Q = np.random.rand(nrows, ncols, nbrAction)