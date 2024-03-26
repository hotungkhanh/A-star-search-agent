# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from .core import PlayerColor, Coord, PlaceAction
from .utils import render_board

def adjacent(coord):
    mylist = []
    mylist.append(coord.down())
    mylist.append(coord.up())
    mylist.append(coord.left())
    mylist.append(coord.right())
    return mylist


# returns the minimum cost in a vector( if
# there are multiple goal states)
def uniform_cost_search(board: dict[Coord, PlayerColor], target: Coord):

    targets = adjacent(target)
     
    # minimum cost upto
    # goal state from starting
    cost = {}
    parent = {}
 
    # map to store visited node
    visited = {}

    # create a priority queue
    queue = []

    for r in range(11):
        for c in range(11):
            cost[Coord(r, c)] = 10**8
 
    # insert the starting indices
    for coord in board:
        if board[coord] == PlayerColor.RED:
            queue.append([0, coord])
            cost[coord] = 0
        else:
            visited[coord] = 1
 
    # while the queue is not empty
    while (len(queue) > 0):
 
        # get the top element of the
        queue = sorted(queue)
        currentsquare = queue[-1]
 
        # pop the element
        del queue[-1]
 
        # get the original value
        currentsquare[0] *= -1
 
        # check if the element is part of
        # the goal list
        if (currentsquare[1] in targets):
 
            # pop the element
            del queue[-1]
 
            queue = sorted(queue)
            parent[target] = currentsquare[1]
            break
 
        # check for the non visited nodes
        # which are adjacent to present node
        if currentsquare[1] not in visited:
            for adjsquare in adjacent(currentsquare[1]):

                if adjsquare not in board:
                    if cost[adjsquare] > cost[currentsquare[1]] + 1:
                        queue.append( [(currentsquare[0] + 1)* -1, adjsquare])
                        cost[adjsquare] = cost[currentsquare[1]] + 1
                        parent[adjsquare] = currentsquare[1]

        visited[currentsquare[1]] = 1

    path = []
    node = target
    while node is not None:
        path.insert(0, node)
        node = parent.get(node)
    if len(path) == 1:
        return None
    return path

def search(
    board: dict[Coord, PlayerColor], 
    target: Coord
):
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.  
        `target`: the target BLUE coordinate to remove from the board.
    
    Returns:
        A list of "place actions" as PlaceAction instances, or `None` if no
        solution is possible.
    """

    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a colour-coded version!
    print(render_board(board, target, ansi=True))

    return uniform_cost_search(board, target)