import numpy as np

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

rat_mark = 0.5

class Qmaze(object):
    """
    
    Attributes:
        - _maze (np.array) : The initial maze (the environment)
        - maze (np.array) : The maze at time t
        - target (tuple) : The target. It's position is in the right bottom of the grid
        - free_cells (list) : List of the free cells
        - rat (tuple) : The initial position of the rat
        - state (tuple) : ?
        - min_reward (float) : ?
        - total_reward (float) : The value of the reward accumulated by the rat
        - visited (set) : The position visited by the rat

    """

    def __init__(self,maze:np.array,rat:tuple=(0,0)) -> None :

        # Get the maze (the environment)
        self._maze = maze

        # Get the shape of the maze
        nrows,ncols = self._maze.shape
        
        # Build the target
        self.target = (nrows - 1, ncols -1)

        # Get the list of the free cells
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        self.free_cells.remove(self.target)

        # Verify that the target and the rat are in free cells
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        
        # Finally, init the maze
        self.reset(rat)

    def reset(self,rat:tuple) -> None:
        # Reset the rat
        self.rat = rat

        # Reset the maze
        self.maze = np.copy(self._maze)

        # Put to the maze the position of the rat
        row,col = rat
        self.maze[row,col] = rat_mark

        #
        self.state = (row,col,'start')

        #
        self.min_reward = -0.5 * self.maze.size

        # 
        self.total_reward = 0

        self.visited = set()


    def update_state(self,action):
        # Get the dimension of the maze
        nrows,ncols = self.maze.shape

        # Get the state of the rat
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        # If rat in a valid position
        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        # Get the list of the valid actions
        valid_actions = self.valid_actions()

        # If there is no valid action
        if not valid_actions:
            nmode = 'blocked'

        # Elif the action is a valid action
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        # invalid action, no change in rat position
        else:             
            mode = 'invalid'

        # Update the new state
        self.state = (nrow, ncol, nmode)


    def get_reward(self):
        """
        Give the reward reguarding the situation of the rat
        """

        # Get the state of the rat
        rat_row, rat_col, mode = self.state

        # Get the size of the maze
        nrows, ncols = self.maze.shape

        # If the rat is at the cheese the reward is 1
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 1.0
        
        # If he is blocked
        if mode == 'blocked':
            return self.min_reward - 1
        
        # If he go to a cell he already visited
        if (rat_row, rat_col) in self.visited:
            return -0.25
        
        # If the action is invalid (like go on a wall)
        if mode == 'invalid':
            return -0.75
        
        # If he progress through the maze
        if mode == 'valid':
            return -0.04

    def act(self, action):
        """
        Process one step in the maze
        """
        # Update the action
        self.update_state(action)

        # Get the reward
        reward = self.get_reward()

        # Update the total reward value
        self.total_reward += reward

        # Get the status of the game
        status = self.game_status()

        # Observe the maze (usefull to draw the maze)
        envstate = self.observe()

        return envstate, reward, status
    

    def observe(self):
        """
        observe the maze ?? 
        """
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate
    
    def draw_env(self):
        """
        Create a canvas usefull to draw the maze with matplotlib
        """
        # Create the canvas
        canvas = np.copy(self.maze)

        # Get the shape
        nrows, ncols = self.maze.shape

        # Clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0

        # Draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark

        return canvas
    

    def game_status(self):
        """
        Get the game status reguarding the situation
        """

        # Look if the game is loosed
        if self.total_reward < self.min_reward:
            return 'lose'
        
        # Look if the game is win
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 'win'

        return 'not_over'