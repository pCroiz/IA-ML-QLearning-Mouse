from qmaze import *

class Rat(object):

    def __init__(self,maze:Qmaze,initPosition:tuple=(0,0)) -> None:
        # Get the maze
        self._maze = maze

        # Get the init Position
        self._initPosition = initPosition
        