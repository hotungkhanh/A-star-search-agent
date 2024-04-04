# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from .core import PlayerColor, Coord, PlaceAction, BOARD_N
from .utils import render_board
from queue import PriorityQueue as pq
import time

PIECE_N = 4

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

    return astar(board, target)

class State():
    """
    A class representing one state of the game, along with its path cost, 
    heuristic function value, and overall evaluation function value for A*
    pathfinding.
    """

    def __init__(self, parent=None, 
                 board: dict[Coord, PlayerColor]=None, 
                 piece: PlaceAction=None):
        self.parent = parent        # parent State
        self.board = board          # a dictionary representing current board
        self.piece = piece          # the PlaceAction added to the parent State
                                        # to create this current State
        
        self.g_n = 0                # cost so far to reach n (path cost)
        self.h_n = 0                # estimated cost to goal from n
        self.f_n = 0                # estimated total cost of path through n to goal

    def __eq__(self, other: 'State'):
        return self.hashable_value() == other.hashable_value()

    def hashable_value(self):
        return frozenset(self.board.items())
    
    def __gt__(self, other: 'State'):
        return (self.f_n > other.f_n)
    
    def generate_children(self) -> list['State']:
        children = []

        for coord, color in self.board.items():
            if color == PlayerColor.RED:
                piece_combinations = self.generate_piece_combinations(coord)
                for piece_coords in piece_combinations:
                    new_board = dict(self.board)
                    for new_coord in piece_coords:
                        new_board[new_coord] = PlayerColor.RED
                    new_piece = PlaceAction(*piece_coords)
                    new_state = State(self, new_board, new_piece)
                    new_state = line_removal(new_state)
                    children.append(new_state)

        return children

    def generate_piece_combinations(self, touched_coord) -> list:
        """
        Generate all possible piece combinations touching a given coordinate.
        """

        piece_combinations = set()
        stack = [(touched_coord, [])]

        while stack:
            current_coord, current_piece = stack.pop()
            if len(current_piece) == PIECE_N:
                piece_combinations.add(tuple(sorted(current_piece)))
            else:
                for adjacent_coord in adjacent(current_coord):
                    if ((adjacent_coord not in self.board) and 
                        (adjacent_coord not in current_piece)):
                        stack.append((adjacent_coord, current_piece + 
                                      [adjacent_coord]))

        return piece_combinations
    
def adjacent(
        coord: Coord
):
    """
    Computes all 4 possible adjacent coordinates

    Parameters:
        `coord`: a `Coord` instance that represents the coordinate that we want
        to find adjacent coordinates for

    Returns:
        An array of adjacent coordinates on the board
    """

    return [coord.down(), coord.up(), coord.left(), coord.right()]


def heuristic(
        state: State, 
        target
) -> int:
    """
    Computes the heuristic function h(n) used for A* pathfinding

    Parameters:
        `state`: a `State` instance that represents the given board state
        `target`: the target BLUE coordinate to remove from the board

    Returns:
        The integer value of h(n)
    """

    row_counter = 0
    col_counter = 0

    nearest_row = BOARD_N
    nearest_col = BOARD_N

    if target not in state.board.keys():
        return 0

    for coord in state.board:

        # count how many tokens in the row & column of the target are filled
        if coord.r == target.r:
            row_counter += 1
        if coord.c == target.c:
            col_counter += 1

        # find the nearest distance from the RED coordinates
        # to the row and column of the target
        if state.board[coord] == PlayerColor.RED:
        
            rdiff = min(abs(coord.r - target.r), BOARD_N - abs(coord.r - target.r))
            cdiff = min(abs(coord.c - target.c), BOARD_N - abs(coord.c - target.c))
            
            if rdiff < nearest_row:
                nearest_row = rdiff

            if cdiff < nearest_col:
                nearest_col = cdiff

    h_n = min(nearest_row + (BOARD_N - row_counter),
              nearest_col + (BOARD_N - col_counter))
            
    return h_n


def line_removal(
        state: State
) -> State:
    """
    Removes any rows or columns that are completely filled with blocks
    i.e. performs the line removal mechanic

    Parameters:
        `state`: a State instance that we want to check line removal for
    
    Returns:
    A new State with the appropriate rows or columns removed
    """

    new_state = State(state.parent, {}, state.piece)
    del_row = []
    del_col = []

    # locate rows and columns to remove
    for i in range(BOARD_N):

        # simultaneously check row i and col i to see if they are filled 
        row_counter = 0
        col_counter = 0
        for coord in state.board:
            if coord.r == i:
                row_counter += 1
            if coord.c == i:
                col_counter += 1
        if (row_counter >= BOARD_N):
            del_row.append(i)
        if (col_counter >= BOARD_N):
            del_col.append(i)
    
    # create a new State with the previously filled rows or columns removed
    for key in state.board.keys():
        if key.r not in del_row and key.c not in del_col:
            new_state.board[key] = state.board[key]

    return new_state
        

def astar(
        board: dict[Coord, PlayerColor], 
        target: Coord
):
    """
    A modified version of A* pathfinding used to solve the game
    Goal: remove the target Coord from the board

    Parameters:
        `board`: a dictionary representing the initial board state, with `Coord`
        instances as keys, and `PlayerColor` instances as values
        `target`: the target BLUE coordinate to remove from the board

    Returns:
        A list of PlaceActions to reach the goal, or `None` if the goal is not 
        achievable
    """

    start_time = time.time()

    # create the starting State
    start_state = State(None, board)

    # check whether the starting board is the goal state
    start_state = line_removal(start_state)
    if target not in start_state.board.keys():
        return

    frontier = pq()         # a priority queue of unexplored States
    explored = set()        # a set of explored States

    frontier.put(start_state)

    # loop until reaching goal state
    while frontier.qsize() > 0:

        # get the next State for expansion (i.e. state with highest priority)
        curr_state = frontier.get()
        
        if curr_state.hashable_value() not in explored:

            # check if goal state has been reached
            if target not in curr_state.board.keys():
                    path = []
                    current = curr_state
                    while current is not None:
                        if current.piece:
                            path.append(current.piece)
                        current = current.parent
                    print("Runtime:", time.time() - start_time)
                    return path[::-1]

            # generate children States of the current State
            children = curr_state.generate_children()

            # loop through children and calculate f(n) = g(n) + h(n)
            for child in children:

                child.g_n = curr_state.g_n + PIECE_N
                child.h_n = heuristic(child, target)
                child.f_n = child.g_n + child.h_n

                if child.hashable_value() not in explored:
                    frontier.put(child)

        # mark current state as explored
        explored.add(curr_state.hashable_value())
        
    return