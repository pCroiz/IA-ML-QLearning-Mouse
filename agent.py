from qmaze import *
import numpy as np
import random
import scipy.special as sp

import torch
import torch.nn as nn
import torch.optim as optim

class Rat(object):

    def __init__(self,maze:Qmaze,possibleAction:enumerate,initPosition:tuple=(0,0),eps:float=0.8,alpha:float=0.8,gamma:float=0.95) -> None:

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
        
        # Set the value of gamma
        self._gamma = gamma
        
        # Set the value of alpha (learning rate)
        self._alpha = alpha


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
        if choice < 1 - self._eps :
            
            # We choose the action with the maximum value of Q
            action = np.argmax(Qvalue)

        # In this one, we do the exploration choice
        else:
            
            # We choose a random action
            action = random.choice([i for i in range(len(self._possibleAction))])

        # return the choosen action
        return action

    def train(self,previousState:tuple,choosenAction:int,rewardReceived:float,state:tuple,) -> None :
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
        
    def setEpsilon(self,eps:float):
        self._eps = eps
    
    def getEpsilon(self):
        return self._eps
   
   
     
class NeuralNetwork(nn.Module):
    
    def __init__(self,Ni:int,Nh1:int,Nh2:int,No:int=4) -> None:
        """
        Initialisation of a neural network using for the deep q-learning algorithm

        Args:
            Ni (int): Numper of inputs neurons
            Nh1 (int): Number of neurons for the first layer
            Nh2 (int): Number of neurons for the second layer
            No (int, optional): Number of outputs neurons. Need to correspond to the number of possible actions. Defaults to 4.
        """
        
        # Init of the super class
        super().__init__()
        
        # Init of the layer
        self.fc1 = nn.Linear(Ni, Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)
        self.act = nn.ReLU()
        
        
    def forward(self, x, classification = False, additional_out=False):
        """
        Forward pass of the neural network. (The way through the neural network)

        Args:
            x ( ? )): Input
            classification (bool, optional): Whether to apply classification. Defaults to False.
            additional_out (bool, optional): Whether to return additional outputs. Defaults to False.

        Returns:
            torch.Tensor: Output tensor
        """
        
        # Pass input through the first layer and apply ReLU activation
        x = self.act(self.fc1(x))

        # Pass the output through the second layer and apply ReLU activation
        x = self.act(self.fc2(x))

        # Pass the output through the third layer to get the final output
        out = self.fc3(x)

        return out
    
def Qloss(batch, net, gamma=0.99, device="cpu"):
    """
    Compute the loss for a Deep Q-Learning (DQL) algorithm.

    Args:
        batch (tuple): A tuple containing the states, actions, next states, rewards, and ?.
        net (nn.Module): The neural network (Q-network) that estimates the Q-values.
        gamma (float, optional): The discount factor, which determines the importance of future rewards. Defaults to 0.99.
        device (str, optional): The device on which the computations are performed (e.g., "cuda" for GPU or "cpu" for CPU). Defaults to "cpu".

    Returns:
        torch.Tensor: The computed loss.
    """
    
    # Unpacking the batch
    states, actions, next_states, rewards, _ = batch
    
    # Get the number of states
    lbatch = len(states)
    
    # Pass the states through the neural network
    state_action_values = net(states.view(lbatch,-1))
    
    # Select the Q-value corresponding to the actions taken
    state_action_values = state_action_values.gather(1, actions.unsqueeze(-1))
    
    # Remove the extra dimension
    state_action_values = state_action_values.squeeze(-1)
    
    # Pass the next states through the neural network
    next_state_values = net(next_states.view(lbatch, -1))
    
    # Select the maximum Q-Value for each next state
    next_state_values = next_state_values.max(1)[0]
    
    # Detach the Next State Values from the Computation Graph
    next_state_values = next_state_values.detach()
    
    # Compute the Expected Q-Values Using the Bellman Equation
    expected_state_action_values = next_state_values * gamma + rewards
    
    # Compute the MSE
    return nn.MSELoss()(state_action_values, expected_state_action_values)

class NeuralRat(object):
    
    def __init__(self,maze:Qmaze,possibleAction:enumerate,neuralNetwork:NeuralNetwork,initPosition:tuple=(0,0),eps:float=0.8,alpha:float=0.8,gamma:float=0.95,device:str='cpu') -> None:
        
        # Get the maze
        self._maze = maze
        
        # Get the neural Network
        self.neuralNetwork = neuralNetwork
        
        # Set the optimizer
        self._optimizer = optim.Adam(self.neuralNetwork.parameters(), lr=1e-4)

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
        
        # Set the value of gamma
        self._gamma = gamma
        
        # Set the value of alpha (learning rate)
        self._alpha = alpha
        
        # Set the device we used (CPU or GPU)
        self._device = device

    def act(self,state:tuple,softMax:bool=True) -> int:
        """
        Reguarding the current environment state, compute the neww action using the neural Network

        Args:
            state (tuple): The environment state
            softMax (bool, optional): If we softmax the value . Defaults to True.

        Returns:
            int: Return the choosen action
        """
        
        # Converts the state into a torch Tensor
        state = torch.Tensor(state).to(self._device).view(1,-1)
        
        # computes the Q-values for the given state using the neural network
        Qvalue = self.neuralNetwork(state).cpu().detach().numpy().squeeze()
       
        # If softmax
        if softMax:
            p = sp.softmax(Qvalue/self._eps).squeeze()
            p /= np.sum(p)
            action = np.random.choice(len(self._possibleAction), p = p)
            
        # Choose between exploitation (eps) or exploration (1-eps)
        choice = random.random()

        # In this case, we do the exploitation choice
        if choice < 1 - self._eps :
            
            # We choose the action with the maximum value of Q
            action = np.argmax(Qvalue)

        # In this one, we do the exploration choice
        else:
            
            # We choose a random action
            action = random.choice([i for i in range(len(self._possibleAction))])

        # return the choosen action
        return action

    def train(self, previousState: tuple, chosenAction: int, rewardReceived: float, state: tuple):
        """
        Train the model. Here, we train the NeuralNetwork claas

        Args:
            previousState (tuple): The previous environment state (time t-1)
            chosenAction (int): The action the model choose (time t-1)
            rewardReceived (float): The reward he received for the action choosen (time t-1)
            state (tuple): The new environment state (time t)

        Returns:
            float: The current loss
        """
        
        
        # Convert states to tensors
        previousState = torch.Tensor(previousState).to(self._device).view(1, -1)
        state = torch.Tensor(state).to(self._device).view(1, -1)

        # Convert action to tensor
        chosenAction = torch.Tensor([chosenAction]).long().to(self._device)

        # Convert reward to tensor
        rewardReceived = torch.Tensor([rewardReceived]).to(self._device)

        # Create a batch with a single transition
        batch = (previousState, chosenAction, state, rewardReceived, None)

        # Compute the loss
        loss = Qloss(batch, self.neuralNetwork, gamma=self._gamma, device=self._device)

        # Zero the gradients
        self._optimizer.zero_grad()

        # Perform backpropagation
        loss.backward()

        # Update the network parameters
        self._optimizer.step()
        
        return loss.item()
    
    def setEpsilon(self,eps:float):
        self._eps = eps
    
    def getEpsilon(self):
        return self._eps