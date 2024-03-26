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

    # path = (astar(board, target))
    # for coord in path:
    #     print(coord)



class State():
    '''
    A class representing one state of the board for A* pathfinding
    '''

    def __init__(self, parent=None, 
                 board: dict[Coord, PlayerColor]=None, 
                 piece: PlaceAction=None):
        self.parent = parent        # parent node
        self.board = board          # dict with key = Coord, val = colour
        self.piece = piece          # a placeAction i.e. the piece added to parent 

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other: 'State'):
        return self.board == other.board
    
    def __str__(self) -> str:
        return f"f={self.f}"

    def generate_children(self) -> list['State']:
        '''
        Use search algorithm (DFS?) to find all possible pieces to place
        '''

        # Thoughts...
        # Use DFS to find all possible place actions touching the most recently placed piece 
        # We should be able to assume that all other possible combos have already been generated
        # in parent state and thus don't need to be generated again

        # So we should only need to run DFS with each of the 4 squares of the new piece as starting nodes

        children = []
        return children

def manhattan_dist(p1: Coord, p2: Coord):
    '''
    Finds shortest manhatten distance between two Coords
    Takes into account the torus nature of board
    '''

    rdiff = min(abs(p1.r - p2.r), 10 - abs(p1.r - p2.r))
    cdiff = min(abs(p1.c - p2.c), 10 - abs(p1.c - p2.c))

    return rdiff**2 + cdiff**2
    

def line_removal(state: State):
    for i in range(11):
        all_coords = state.board.keys()
        


def astar(board, target):
    # get targets to fill
    row_to_fill = []
    col_to_fill = []
    for i in range(11):
        row_square = Coord(target.r, i)
        col_square = Coord(i, target.c)
        if row_square not in board:
            row_to_fill.append(row_square)
        if col_square not in board:
            col_to_fill.append(col_square)

    # get starting nodes
    start_state = State(None, board)
    render_board(start_state.board, target)

    # lists of states
    open = []
    closed = []

    open.append(start_state)

    # loop until reaching goal state
    while len(open) > 0:
        
        # get curr state  (i.e. state with highest priority)
        curr_state = open[0]
        curr_idx = 0
        for idx, state in enumerate(open):
            if state.f < curr_state.f:
                curr_state = state
                curr_idx = idx
        
        # pop curr node off open list, add it to closed
        open.pop(curr_idx)
        closed.append(curr_state)

        # check if target is removed
        # TODO: refine expression to check if target is removed
        if target == None:
            path = []
            current = curr_state
            while current is not None:
                path.append(current.board)
                current = current.parent
            return path[::-1]
        
        continue 

        # ------------------------------------------------------------
        # Under Construction
        # ------------------------------------------------------------
        # generate children as node
        children = curr_state.generate_children()

        # loop through children
        for child in children:

            # if child on closed list
            if child in closed:
                print("child in closed")
                continue        # skip to next child

            # otherwise create child
            child.g = curr_state.g + 1
            # TODO: h(x) = manhattan_dist to row/col + number of square left to fill in row/col     
            child.h = manhattan_dist(child.position, target)
            # child.h = 0
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