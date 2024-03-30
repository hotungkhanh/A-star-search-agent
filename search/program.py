# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from .core import PlayerColor, Coord, PlaceAction
from .utils import render_board
from queue import PriorityQueue as pq
import time


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

    # Do some impressive AI stuff here to find the solution...
    return astar(board, target)

class State():
    '''
    A class representing one state of the game, along with its path cost, 
    heuristic function value, and overall evaluation function value for A*
    pathfinding.
    '''

    def __init__(self, parent=None, 
                 board: dict[Coord, PlayerColor]=None, 
                 piece: PlaceAction=None):
        self.parent = parent        # parent node
        self.board = board          # dict with key = Coord, val = colour
        self.piece = piece          # a placeAction i.e. the piece added to parent
        
        self.hashvalue = self.__hash__()
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other: 'State'):
        return self.hashvalue == other.hashvalue
    
    def __str__(self) -> str:
        return f"f={self.hashvalue}"

    def __hash__(self) -> int:
        all_coords = []
        for tup in self.board.items():
            all_coords.append((tup))

        return hash(tuple(sorted(all_coords)))
    
    def __gt__(self, other: 'State'):
        return (self.f > other.f)

    def test_gen_children(self, target) -> list['State']:
        '''
        Calling function for recursive children generation
        [under construction]
        '''
        children = []

        for curr_coord, colour in self.board.items():
            adj_pieces = []
            if colour == PlayerColor.BLUE:
                continue

            adj_coords = adjacent(curr_coord)

            # recursively find adj pieces 
            for adj in adj_coords:
                if adj in self.board.keys():
                    continue
                curr_path = [adj]

                adj_piece = self.recursive_adj_cells(adj, curr_path, 1, 4)

                adj_pieces.extend(adj_piece)

            for new_piece_coords in adj_pieces:
                new_board = dict(self.board)
                for new_coord in new_piece_coords:
                    new_board[new_coord] = PlayerColor.RED
                new_piece = PlaceAction(*new_piece_coords)
                new_state = State(self, new_board, new_piece)
                new_state = line_removal(new_state)

                children.append(new_state)

        return children

    def recursive_adj_cells(self, curr_coord, curr_path, depth, max_depth):
        '''
        Recursively generates children states
        [under construction]
        '''
        print("------------NEW RECUR CALL ---------------")
        print("curr coord:", curr_coord)      
        print("curr path:", curr_path)  

        if depth >= max_depth:
            print("----- BASE CASE ------")
            print(curr_path)
            return curr_path
        adj_coords = adjacent(curr_coord)

        all_deeper_paths = []      # a list of list of coords

        for adj in adj_coords:
            if (adj not in self.board.keys()) and (adj not in curr_path):
                # print("here")
                new_path = curr_path + [adj]
                deeper_paths = self.recursive_adj_cells(adj, new_path, depth + 1, max_depth)
                print('\ndeeper:', deeper_paths)
                all_deeper_paths.append(deeper_paths)

        print()
        print(all_deeper_paths)
        
        # if depth == max_depth:
        #     return[[curr_coord] + path for path in adj_cells]

        return all_deeper_paths

    def generate_children(self, target) -> list['State']:
        children = []

         # Iterate over all red cells on the board
        for coord, color in self.board.items():
            if color == PlayerColor.RED:

                # STEP 1
                onecell = []
                adjacent_coords = [coord.down(), coord.up(), coord.left(), coord.right()]
                for adjacent_coord in adjacent_coords:
                    if adjacent_coord in self.board.keys():
                        continue
                    onecell.append([adjacent_coord])

                # STEP 2
                twocell = []
                for one in onecell:
                    if one:
                        for last in one:
                            adjacent_coords = [last.down(), last.up(), last.left(), last.right()]
                            for adjacent_coord in adjacent_coords:
                                if adjacent_coord in self.board.keys():
                                    continue
                                twocell.append(one + [adjacent_coord])

                # STEP 3
                threecell = []
                for two in twocell:
                    if two:
                        for last in two:
                            adjacent_coords = [last.down(), last.up(), last.left(), last.right()]
                            for adjacent_coord in adjacent_coords:
                                if adjacent_coord in self.board.keys() or adjacent_coord in two:
                                    continue
                                threecell.append(two + [adjacent_coord])


                # STEP 4
                fourcell = []
                for three in threecell:
                    if three:
                        for last in three:
                            adjacent_coords = [last.down(), last.up(), last.left(), last.right()]
                            for adjacent_coord in adjacent_coords:
                                if adjacent_coord in self.board.keys() or adjacent_coord in three:
                                    continue
                                fourcell.append(three + [adjacent_coord])

                myset = set()
                for new_piece_coords in fourcell:
                    myset.add(tuple(sorted(new_piece_coords)))

                for new_piece_coords in myset:
                    # create new board
                    new_board = dict(self.board)
                    for new_coord in sorted(new_piece_coords):
                        new_board[new_coord] = PlayerColor.RED
                    new_piece = PlaceAction(*new_piece_coords)
                    new_state = State(self, new_board, new_piece)
                    new_state = line_removal(new_state)

                    children.append(new_state)

        return children
    
def adjacent(
        coord: Coord
):
    '''
    Computes all 4 possible adjacent coordinates

    Parameters:
        `coord`: a `Coord` instance that represents a coordinate that we want
        to find adjacent coordinates for

    Returns:
        An array of adjacent coordinates on the board
    '''

    adjacent_coords = []
    adjacent_coords.append(coord.down())
    adjacent_coords.append(coord.up())
    adjacent_coords.append(coord.left())
    adjacent_coords.append(coord.right())
    return adjacent_coords


def heur(
        state: State, 
        target
) -> int:
    '''
    Computes the heuristic function h(x) used for A* pathfinding

    Parameters:
        `state`: a `State` instance that represents the given board state
        `target`: the target BLUE coordinate to remove from the board

    Returns:
        The integer value of h(x)
    '''

    row_counter = 0
    col_counter = 0

    nearest_row = 11
    nearest_col = 11

    if target not in state.board.keys():
        return 0

    for coord in state.board:
        if coord.r == target.r:
            row_counter += 1
        if coord.c == target.c:
            col_counter += 1

        if state.board[coord] == PlayerColor.RED:
            rdiff = min(abs(coord.r - target.r), 11 - abs(coord.r - target.r))
            cdiff = min(abs(coord.c - target.c), 11 - abs(coord.c - target.c))
            
            if rdiff < nearest_row:
                nearest_row = rdiff

            if cdiff < nearest_col:
                nearest_col = cdiff

    heur = min(nearest_row + (11 - row_counter), nearest_col + (11 - col_counter))
            
    return heur


def line_removal(
        state: State
) -> State:
    '''
    Takes a State as input

    Removes any rows or columns that are completely filled with blocks
    i.e. performs the line removal mechanic

    Returns a new State with the appropriate rows or columns removed
    '''

    new_state = State(state.parent, {}, state.piece)
    del_row = []
    del_col = []

    # locate rows and cols to remove
    for i in range(11):
        # simultaneously check row i and col i to see if they are filled 
        row_counter = 0
        col_counter = 0
        for coord in state.board:
            if coord.r == i:
                row_counter += 1
            if coord.c == i:
                col_counter += 1
        if (row_counter >= 11):
            del_row.append(i)
        if (col_counter >= 11):
            del_col.append(i)
    
    # remove specified rows and cols if any
    for key in state.board.keys():
        if (key.r not in del_row) and (key.c not in del_col):
            new_state.board[key] = state.board[key]

    return new_state
        

def astar(
        board: dict[Coord, PlayerColor], 
        target: Coord
):
    '''
    Uses a modified version of A* pathfinding to solve the game
    Goal: remove the target Coord from the board

    Parameters:
        `board`: a dictionary representing the initial board state, with `Coord`
        instances as keys, and `PlayerColor` instances as values
        `target`: the target BLUE coordinate to remove from the board

    Returns:
        A list of PlaceActions to reach the goal, or `None` if the goal is not 
        achievable
    '''

    start_time = time.time()
    # get starting nodes
    start_state = State(None, board)

    # lists of states
    frontier = pq()         # stores a list of state
    explored = set()         # only stores states

    frontier.put(start_state)


    # loop until reaching goal state
    while frontier.qsize() > 0:

        # get curr state  (i.e. state with highest priority)
        curr_state = frontier.get()
        
        if curr_state not in explored:

            if target not in curr_state.board.keys():
                    path = []
                    current = curr_state
                    while current is not None:
                        if current.piece:
                            path.append(current.piece)
                        current = current.parent
                    print("total runtime in secs:", time.time() - start_time)
                    return path[::-1]

            # generate children as states
            children = curr_state.generate_children(target)

            # loop through children
            for child in children:

                child.g = curr_state.g + 4
                child.h = heur(child, target)
                child.f = child.g + child.h

                if child not in explored:
                    frontier.put(child)

        explored.add(curr_state)
        
    return