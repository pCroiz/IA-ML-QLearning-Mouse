from qmaze import *
from agent import *
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

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
    
            # If the model is a classic Rat
            if isinstance(self._model, Rat) :
                
                # Get the agent position
                state = self._qmaze.getAgentPosition()
    
                # The agent do a choice
                action = self._model.act(state)
    
                # The qmaze give a reward and a status according to the action choosen
                _, reward, status = self._qmaze.act(action)
    
                # Get the new state
                newSate = self._qmaze.getAgentPosition()
    
                # Update the model
                self._model.train(state,action,reward,newSate)
            
            # If the model is the neuralRat
            else:

                # Get the current state of the maze
                envState = self._qmaze.observe()

                # The agent do a choice
                action = self._model.act(envState)

                # Get the new state of the maze
                nextEnvstate, reward, status = self._qmaze.act(action)

                # Train the model
                self._lossList.append(self._model.train(envState,action,reward,nextEnvstate))
    
            if status == 'lose':
                if textDisplay : print("The game has been losed in : " + str(numberIteration) + " iteration")
                stop = True
            elif status == 'win':
                if textDisplay : print("The game has been winned in : " + str(numberIteration) + " iteration")
                stop = True
            else:
                numberIteration += 1
                
        return status,numberIteration
    
    def playAnimation(self,ratPosition:tuple=(0,0)):
        """
        Play the game while displaying animation of it

        Args:
            ratPosition (tuple, optional): Initial position of the rat. Defaults to (0,0).
        """
        
        # Reset the maze
        self._qmaze.reset(ratPosition)
        
        # Set up the figure and axes for animation
        fig, ax = plt.subplots()
        img = ax.imshow(self._qmaze.draw_env(), interpolation='none', cmap='gray')

        def update(frame):
            
            # If the model is a classic Rat
            if isinstance(self._model, Rat) :
                
                # Get the agent position
                state = self._qmaze.getAgentPosition()
    
                # The agent do a choice
                action = self._model.act(state)
    
                # The qmaze give a reward and a status according to the action choosen
                _, reward, status = self._qmaze.act(action)
    
                # Get the new state
                newSate = self._qmaze.getAgentPosition()
    
                # Update the model
                self._model.train(state,action,reward,newSate)
            
            # If the model is the neuralRat
            else:

                # Get the current state of the maze
                envState = self._qmaze.observe()

                # The agent do a choice
                action = self._model.act(envState)

                # Get the new state of the maze
                nextEnvstate, reward, status = self._qmaze.act(action)

                # Train the model
                self._model.train(envState,action,reward,nextEnvstate)

            if status == 'lose' or status == 'win':
                return True

            # Update the image data
            img.set_data(self._qmaze.draw_env())

            return [img]

        # Set up the animation
        ani = animation.FuncAnimation(fig, update, interval=100)

        # Show the animation
        plt.show()

    
       
    def train(self, numOfEpochs:int,cutoff:int=3000,useVariableEpsilon:bool=True,**kwargs) -> None :
        """
        Train the model

        Args:
            numOfEpochs (int): number of iteration we want to train the model
            cutoff (int, optional): Variable to modify the distribution of epsilon. Defaults to 3000.
            useVariableEpsilon (bool, optional): Modify epsilon while training the model. Defaults to True.
        """
        
        # Variable to track the evolution of wins and looses
        evolution = []
        
        # Initialize the list of the decrescent epsilon
        if useVariableEpsilon:
            epsilon = calculate_epsilon(numOfEpochs,cutoff,('displayEpsilon' in kwargs and kwargs['displayEpsilon']))

        # Start the loop
        for i in tqdm(range(numOfEpochs)):
            
            # Set epsilon
            if useVariableEpsilon : self._model.setEpsilon(epsilon[i])
            
            # Play a game
            status,_ = self.play(textDisplay=False)
            
            # If it's a win, we add the number of iteraton to the corresponding list
            if status == 'win' :
                
                # If we win we append 1
                evolution.append(1)
                
            else:
                # Else, we append 0
                evolution.append(0)
                

        # Display the number of wins and losses
        tqdm.write(f"\033[92mWins: {np.sum(evolution)}\033[0m")  # Green text for wins
        tqdm.write(f"\033[91mLosses: {i-np.sum(evolution)}\033[0m")  # Red text for losses  
        
        # Display the evolution of the wins and the looses
        epochs = range(numOfEpochs)
        plt.plot(epochs, [sum(evolution[:i+1]) for i in epochs], 'g', label='Wins')
        plt.plot(epochs, [i - sum(evolution[:i+1]) for i in epochs], 'r', label='Losses')
        plt.xlabel('Number of Games')
        plt.ylabel('Number of Wins/Losses')
        plt.title('Evolution of Wins and Losses Through the Games')
        plt.legend()
        plt.show()

def calculate_epsilon(numOfEpochs:int,cutoff:int=3000, display:bool=False):
    """
    Compute a list of decrescent epsilon

    Args:
        numOfEpochs (int): number of epsilon we want
        cutoff (int, optional): Variable to modify the distribution of epsilon. Defaults to 3000.
        display (bool, optional): To display the . Defaults to False.

    Returns:
        list: epsilon list
    """
    # Calculate the decaying epsilon values
    epsilon = np.exp(-np.arange(numOfEpochs) / cutoff)

    # Find the cutoff index
    cutoff_index = min(100 * int(numOfEpochs / cutoff), numOfEpochs - 1)

     # Apply the cutoff to the epsilon values
    epsilon[epsilon > epsilon[cutoff_index]] = epsilon[cutoff_index]
    
    # Display the evolution of epsilon
    if display :
        plt.plot(epsilon, color = 'orangered', ls = '--')
        plt.xlabel('Epochs')
        plt.ylabel('Epsilon')
        plt.show()
        

    return epsilon