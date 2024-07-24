from qmaze import *
import numpy as np
import random

class Rat(object):

    def __init__(self,maze:Qmaze,possibleAction:enumerate,eps:float=0.9,initPosition:tuple=(0,0)) -> None:

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

        # Set the value of epsilon
        self._eps = eps


    def act(self,state:tuple) -> int:
        """
        Reguarding the current state, choose the action a using the policy of Q

        Args:
            state (tuple): The current state

        Returns:
            int: The action choosen
        """
        
        # Get the indices
        i,j = state

        # Get the Q value for the current state
        Qvalue = self._Q[i,j,:]

        # Choose between exploitation (eps) or exploration (1-eps)
        choice = random.random()

        # In this case, we do the exploitation choice
        if choice < self._eps :
            
            # We choose the action with the maximum value of Q
            action = np.argmax(Qvalue)

        # In this one, we do the exploration choice
        else:
            
            # We choose a random action
            action = random.choice([i for i in range(len(self._possibleAction))])

        # return the choosen action
        return action

    def train(self,reward):
        pass