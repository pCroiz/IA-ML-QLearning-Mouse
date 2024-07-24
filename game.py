from qmaze import *
from agent import *
import matplotlib.pyplot as plt

class Game(object):

    def __init__(self,model:Rat,qmaze:Qmaze) -> None:
        self._model = model
        self._qmaze = qmaze
        
        
    def play(self,ratPosition:tuple=(0,0),**kwargs) -> str:
        """
        Play one game of the maze

        Args:
            ratPosition (tuple, optional): The initial rat position. Defaults to (0,0).
            
        Keywords Arguments:
            textDisplay (bool) : To display some information of the play

        Returns:
            str: The status of the game
        """
        
        # Display the text or not
        if 'textDisplay' in kwargs and kwargs['textDisplay']:
            textDisplay = True
        else:
            textDisplay = False
        
        stop = False
        numberIteration = 0
        
        # Reset the maze
        self._qmaze.reset(ratPosition)

        while not(stop):
    
            # Get the agent position
            state = self._qmaze.getAgentPosition()
    
            # The agent do a choice
            action = self._model.act(state)
    
            # The qmaze give a reward and a status according to the action choosen
            _, reward, status = self._qmaze.act(action)
    
            # Get the new state
            newSate = self._qmaze.getAgentPosition()
    
            # Update the model
            self._model.updateQ(state,action,reward,newSate)
    
            if status == 'lose':
                if textDisplay : print("The game has been losed in : " + str(numberIteration) + " iteration")
                stop = True
            elif status == 'win':
                if textDisplay : print("The game has been winned in : " + str(numberIteration) + " iteration")
                stop = True
            else:
                numberIteration += 1
                
        return status,numberIteration
    
    
    
    
    def train(self, numberOfWin:int) -> None :
        """
        Train the model to solve the maze

        Args:
            numberOfWin (int): The number of win before we consider the model effective
        """
        
        # Initialize the variable for the iteration
        numberOfIteration = 0
        
        # Initialize a list that keeps the number of iteration through the victory
        iterationForVictory = [100]
        
        # Initialize a list that keeps the number of iteration through the looses
        iterationForLoose = [100]
        
        # Start the loop
        while numberOfIteration < numberOfWin:
            
            # Play a game
            status,numberIteration = self.play(textDisplay=False)
            
            # If it's a win, we add the number of iteraton to the corresponding list
            if status == 'win' :
                
                # If we win we append the new number of iteration to the win list
                iterationForVictory.append(numberIteration)
                
                # And we dupplicate the last value for the loose list
                iterationForLoose.append(iterationForLoose[-1])
                
                # Increment the number of iteration
                numberOfIteration += 1
            else:
                
                # If we loose we append the new number of iteration to the loose list
                iterationForLoose.append(numberIteration)
                
                # And we dupplicate the last value for the win list
                iterationForVictory.append(iterationForVictory[-1])
                
        # Finally we plot the evolution of the number of iteration through the win
        plt.plot([i for i in range(len(iterationForVictory))], iterationForVictory,'r', label='Wins')
        plt.plot([i for i in range(len(iterationForLoose))], iterationForLoose,'b', label='Losses')
        plt.xlabel('Number of Games')
        plt.ylabel('Number of Iterations')
        plt.title('Evolution of Number of Iterations Through the Games')
        plt.legend()
        plt.show()
