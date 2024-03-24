# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from .core import PlayerColor, Coord, PlaceAction
from .utils import render_board


# returns the minimum cost in a vector( if
# there are multiple goal states)
def uniform_cost_search(board: dict[Coord, PlayerColor], target: Coord):
     
    # minimum cost upto
    # goal state from starting
    answer = 10**8
 
    # create a priority queue
    queue = []
 
    # insert the starting indices
    for coord in board:
        if board[coord] == PlayerColor.RED:
            queue.append([0, coord])
 
    # map to store visited node
    visited = {}
 
    # count
    count = 0
 
    # while the queue is not empty
    while (len(queue) > 0):
 
        # get the top element of the
        queue = sorted(queue)
        p = queue[-1]
 
        # pop the element
        del queue[-1]
 
        # get the original value
        p[0] *= -1
 
        # check if the element is part of
        # the goal list
        if (p[1] == target):
 
            # get the position
            index = target.index(p[1])
 
            # if a new goal is reached
            if answer == 10**8:
                count += 1
 
            # if the cost is less
            if (answer > p[0]):
                answer = p[0]
 
            # pop the element
            del queue[-1]
 
            queue = sorted(queue)
            if (count == len(goal)):
                return answer
 
        # check for the non visited nodes
        # which are adjacent to present node
        if (p[1] not in visited):
            for i in range(len(graph[p[1]])):
 
                # value is multiplied by -1 so that
                # least priority is at the top
                queue.append( [(p[0] + cost[(p[1], graph[p[1]][i])])* -1, graph[p[1]][i]])
 
        # mark as visited
        visited[p[1]] = 1
 
    return answer

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