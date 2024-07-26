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
                self._model.train(envState,action,reward,nextEnvstate)
    
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
        Train the model to solve the maze

        Args:
            numOfEpochs (int): The number of game to train the model
        """
        
        # Initialize the variable for the iteration
        numberOfIteration = 0
        
        # Initialize a list that keeps the number of iteration through the victory
        iterationForVictory = [100]
        
        # Initialize a list that keeps the number of iteration through the looses
        iterationForLoose = [100]
        
        # Initialize the list of the decrescent epsilon
        if useVariableEpsilon:
            epsilon = calculate_epsilon(numOfEpochs,cutoff,('displayEpsilon' in kwargs and kwargs['displayEpsilon']))

        # Start the loop
        for i in tqdm(range(numOfEpochs)):
            
            # Set epsilon
            if useVariableEpsilon : self._model.setEpsilon(epsilon[i])
            
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
                
                print(numberOfIteration)
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

def calculate_epsilon(numOfEpochs:int,cutoff:int=3000, display:bool=False):
    # Calculate the decaying epsilon values
    epsilon = np.exp(-np.arange(numOfEpochs) / cutoff)

    # Find the cutoff index
    cutoff_index = min(100 * int(numOfEpochs / cutoff), numOfEpochs - 1)

     # Apply the cutoff to the epsilon values
    epsilon[epsilon > epsilon[cutoff_index]] = epsilon[cutoff_index]
    
    if display :
        plt.plot(epsilon, color = 'orangered', ls = '--')
        plt.xlabel('Epochs')
        plt.ylabel('Epsilon')
        plt.show()
        

    return epsilon