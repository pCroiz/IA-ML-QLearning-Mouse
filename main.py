# List of the action (encoded by a number)
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down'
}

# List of recompense
move = 0.04
get_cheese = 1.0
go_wall = 0.75
go_outside = 0.8
go_already_visited = 0.25
loose_treshold = -0.5