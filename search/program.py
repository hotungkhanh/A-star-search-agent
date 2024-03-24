# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from .core import PlayerColor, Coord, PlaceAction
from .utils import render_board
from collections import defaultdict as dd


def search(
    board: dict[Coord, PlayerColor], 
    target: Coord
) -> list[PlaceAction] | None:
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
    print(render_board(board, target, ansi=False))

    # Do some impressive AI stuff here to find the solution...
    # ...
    # ... (your solution goes here!)
    # ...

    # compute all possible starting nodes 

    # answer = uniform_cost_search(board, target)
    path = (astar(board, target))
    for coord in path:
        print(coord)
    # Here we're returning "hardcoded" actions as an example of the expected
    # output format. Of course, you should instead return the result of your
    # search algorithm. Remember: if no solution is possible for a given input,
    # return `None` instead of a list.
    return [
        PlaceAction(Coord(2, 5), Coord(2, 6), Coord(3, 6), Coord(3, 7)),
        PlaceAction(Coord(1, 8), Coord(2, 8), Coord(3, 8), Coord(4, 8)),
        PlaceAction(Coord(5, 8), Coord(6, 8), Coord(7, 8), Coord(8, 8)),
    ]

class Node():
    """A node class for A* pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    # def __eq__(self, other):
    #     return self.position == other.position
    
    def __str__(self) -> str:
        return f"Coord: {self.position}   f={self.f}"


def smaller_rc(p1: Coord, p2: Coord):
    # needs fixing
    rdiff = abs(p1.r - p2.r)
    if rdiff > (11//2):
        rdiff = 11 - rdiff
    
    cdiff = abs(p1.c - p2.c)
    if cdiff > (11//2):
        cdiff = 11 - cdiff
    # print(Coord(rdiff, cdiff))
    return Coord(rdiff, cdiff)
    


def astar(board, target):
    # get targets
    # targets = adjacent(target)

    # get starting nodes
    start_nodes = []
    for coord in board:
        if board[coord] == PlayerColor.RED:
            start_node = Node(None, coord)
            start_node.g = start_node.h = start_node.f = 0
            start_nodes.append(start_node)

    open = []       # list of nodes
    closed = []

    open.extend(start_nodes)

    # loop until reaching goal state
    while len(open) > 0:
        
        # get curr node
        curr_node = open[0]
        curr_idx = 0
        for idx, item in enumerate(open):
            if item.f < curr_node.f:
                curr_node = item
                curr_idx = idx
        
        # pop curr node off open list, add it to closed
        open.pop(curr_idx)
        closed.append(curr_node)

        # check if goal is found
        if curr_node.position == target:
            path = []
            current = curr_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]
        
        # generate children as node
        children = []

        for adj in adjacent(curr_node.position):
            # ensure target isn't ignored in board
            if adj == target:
                new_node = Node(parent=curr_node, position=adj)
                children.append(new_node)
            if adj not in board:
                new_node = Node(parent=curr_node, position=adj)
                children.append(new_node)

        # loop through children
        for child in children:

            # if child on closed list
            if child in closed:
                print("child in closed or blocked")
                continue        # skip to next child

            # otherwise create child
            child.g = curr_node.g + 1
            # TODO: find closest target and calculate manhatten dist
            # TODO: calc manhatten dist with wrap around
            target_pos = smaller_rc(child.position, target)
            child.h = ((child.position.r - target.r)**2) + ((child.position.c - target.c)**2)          
            # manhatten dist
            child.f = child.g + child.h

            for open_node in open:
                if child == open_node and child.g > open_node.g:
                    continue
            
            open.append(child)


def adjacent(coord: Coord):
    adjacent_nodes = []
    adjacent_nodes.append(coord.down())
    adjacent_nodes.append(coord.up())
    adjacent_nodes.append(coord.left())
    adjacent_nodes.append(coord.right())
    return adjacent_nodes

def uniform_cost_search(board: dict[Coord, PlayerColor], 
                        target: Coord):
    '''
    board: a dict with key = Coord, value = colour Enum
    target: a Coord 
    start: an array of Coord
    '''
    targets = adjacent(target)
    answer = 10**8
    pq = []
    visited = dd(lambda: None)  # dictionary of expanded/visited nodes
    cost = dd(lambda: None)
    parent = {}

    for coord in board:
        if board[coord] == PlayerColor.RED:
            pq.append([0, coord])
            cost[coord] = 0
        else:
            visited[coord] = 1


    while (len(pq) > 0):
        pq = sorted(pq)         # largest to smallest
        curr_node = pq.pop(0)
        print(f" curr_node: {curr_node}")

        # check if curr node is the target
        if curr_node[1] in targets:
            if answer > curr_node[0]:
                answer = curr_node[0]
            break


        if visited[curr_node[1]] is None:
            children = adjacent(curr_node[1])
            for child in children:
                if child not in board:
                    if cost[child] is None:
                        pq.append([curr_node[0] + 1, child])
                        cost[child] = cost[curr_node[1]] + 1
                        parent[child] = curr_node[1]
                    elif cost[child] > cost[curr_node[1]] + 1:
                        pq.append([curr_node[0] + 1, child])
                        cost[child] = cost[curr_node[1]] + 1
                        parent[child] = curr_node[1]
        
        visited[curr_node[1]] = 1


    return answer