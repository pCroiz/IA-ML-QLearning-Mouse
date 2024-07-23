import numpy as np

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

    def __init__(self,maze,rat=(0,0)) -> None :

        # Get the maze (the environment)
        self._maze = np.array(maze)

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

    def reset(self,rat) -> None:
        # Reset the rat
        self.rat = rat

        # Reset the maze
        self.maze = np.copy(self._maze)

        # Put to the maze the position of the rat
        row,col = rat
        self.maze[row,col] = 0.5

        #
        self.state = (row,col,'start')

        #
        self.min_reward = -0.5 * self.maze.size

        # 
        self.total_reward = 0

        self.visited = set()