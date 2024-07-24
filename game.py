from qmaze import *
from agent import *

class Game(object):

    def __init__(self,model:Rat,qmaze:Qmaze,rat_cell:tuple) -> None:
        self._model = model
        self._qmaze = qmaze
        self.rat_cell = rat_cell
        
        
    def play(self,ratPosition:tuple=(0,0)) -> None:
        """
        Play one game of the maze

        Args:
            ratPosition (tuple, optional): The initial rat position. Defaults to (0,0).
        """
        
        stop = False
        numberIteration = 0
        
        # Reset the maze
        self._qmaze.reset(ratPosition)

        while not(stop):
    
            # Get the agent position
            state = self._qmaze.getAgentPosition()
    
            # The agent do a choice
            action = self._rat.act(state)
    
            # The qmaze give a reward and a status according to the action choosen
            _, reward, status = self._qmaze.act(action)
    
            # Get the new state
            newSate = self._qmaze.getAgentPosition()
    
            self._rat.updateQ(state,action,reward,newSate)
    
            if status == 'lose':
                print("The game has been losed in : " + str(numberIteration) + " iteration")
                stop = True
            elif status == 'win':
                print("The game has been winned in : " + str(numberIteration) + " iteration")
                stop = True
            else:
                numberIteration