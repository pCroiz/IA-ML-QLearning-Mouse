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
        nrows,ncols = self._maze.shape()

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

    def updateQ(self,previousState:tuple,choosenAction:int,rewardReceived:float,state:tuple,) -> None :
        """
        Update the value of Q(s,a)

        Args:
            previousState (tuple): The state before the agent act
            choosenAction (int): The action the agent choosed reguarding the previous state
            rewardReceived (float): The reward the agent recevied by the environnement for the action he did
            state (tuple): The new current state
        """
        
        # Get the indices of the previous state and the new state
        i_prev, j_prev = previousState
        i, j = state

        # Get the current Q value for the previous state and chosen action
        Q_prev = self._Q[i_prev, j_prev, choosenAction]

        # Get the maximum Q value for the new state
        Q_max = max(self._Q[i, j])

        # Calculate the new Q value for the previous state and chosen action
        new_Q = (1 - self._alpha) * Q_prev + self._alpha * (rewardReceived + self._gamma * Q_max)

        # Update the Q matrix
        self._Q[i_prev, j_prev, choosenAction] = new_Q