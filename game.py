from qmaze import *
from agent import *

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
    
            self._model.updateQ(state,action,reward,newSate)
    
            if status == 'lose':
                if textDisplay : print("The game has been losed in : " + str(numberIteration) + " iteration")
                stop = True
            elif status == 'win':
                if textDisplay : print("The game has been winned in : " + str(numberIteration) + " iteration")
                stop = True
            else:
                numberIteration += 1
                
        return status