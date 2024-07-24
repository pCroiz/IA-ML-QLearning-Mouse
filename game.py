from qmaze import *

class Game(object):

    def __init__(self,model,qmaze:Qmaze,rat_cell:tuple) -> None:
        self._model = model
        self._qmaze = qmaze
        self.rat_cell = rat_cell