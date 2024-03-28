# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from .core import PlayerColor, Coord, PlaceAction
from .utils import render_board
from collections import defaultdict as dd
from queue import PriorityQueue as pq
import time


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
    print(render_board(board, target, ansi=True))

    # Do some impressive AI stuff here to find the solution...
    return astar(board, target)

class State():
    '''
    A class representing one state of the board for A* pathfinding
    '''

    def __init__(self, parent=None, 
                 board: dict[Coord, PlayerColor]=None, 
                 piece: PlaceAction=None):
        self.parent = parent        # parent node
        all_coords = []
        for tup in board.items():
            all_coords.append(tup)
        self.board = tuple(all_coords)      # a tuple of tuples 
        self.piece = piece          # a placeAction i.e. the piece added to parent 

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other: 'State'):
        return self.__hash__() == other.__hash__()
    
    def __str__(self) -> str:
        return f"f={self.f}"

    def __hash__(self) -> int:
        return hash(self.board)
    
    def __gt__(self, other: 'State'):
        return self.f > other.f

    def dict_board(self) -> dict[Coord, PlayerColor]:
        '''
        Converts the board (tuple of tuples) into dictionary
        [For testing Purposes i.e. rendering board]
        '''
        board_dict = {}
        for tup in self.board:
            board_dict[tup[0]] = tup[1]
        return board_dict

    def test_gen_children(self, target) -> list['State']:
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
                new_piece = PlaceAction(new_piece_coords)
                new_state = State(self, new_board, new_piece)
                new_state = line_removal(new_state)

                children.append(new_state)

        return children

    def recursive_adj_cells(self, curr_coord, curr_path, depth, max_depth):
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
        for square in self.board:
            # print(f"coord, color: {coord}, {color}")
            if square[1] == PlayerColor.BLUE:
                continue

            # STEP 1
            onecell = []
            adjacent_coords = adjacent(square[0])
            for adjacent_coord in adjacent_coords:
                if any(adjacent_coord in i for i in self.board):
                    continue
                onecell.append([adjacent_coord])

            # print(f"onecell: {onecell}")

            # STEP 2
            twocell = []
            for one in onecell:
                # print(f" for {one} in onecell:")
                if one:
                    for last in one:
                        adjacent_coords = adjacent(last)
                        for adjacent_coord in adjacent_coords:
                            if any(adjacent_coord in i for i in self.board):
                                continue
                            twocell.append(one + [adjacent_coord])

            # print(f"twocell: {twocell}")

            # STEP 3
            threecell = []
            for two in twocell:
                if two:
                    for last in two:
                        adjacent_coords = adjacent(last)
                        for adjacent_coord in adjacent_coords:
                            if any(adjacent_coord in i for i in self.board) or (adjacent_coord in two):
                                continue
                            threecell.append(two + [adjacent_coord])

            # print(f"three: {threecell}")

            # STEP 4
            fourcell = []
            for three in threecell:
                if three:
                    for last in three:
                        adjacent_coords = adjacent(last)
                        for adjacent_coord in adjacent_coords:
                            if any(adjacent_coord in i for i in self.board) or (adjacent_coord in three):
                                continue
                            fourcell.append(three + [adjacent_coord])

            # print(f"four: {fourcell}")

            for new_piece_coords in fourcell:
                # create new board
                new_board = self.dict_board()
                for new_coord in new_piece_coords:
                    new_board[new_coord] = PlayerColor.RED
                new_piece = PlaceAction(*new_piece_coords)
                new_state = State(self, new_board, new_piece)
                new_state = line_removal(new_state)

                children.append(new_state)

        return children
    
def adjacent(coord: Coord):
    '''
    Takes a Coord as an argument
    Returns an array of all 4 possible adjacent Coords
    '''
    adjacent_coords = []
    adjacent_coords.append(coord.down())
    adjacent_coords.append(coord.up())
    adjacent_coords.append(coord.left())
    adjacent_coords.append(coord.right())
    return adjacent_coords


def heur(state: State, target) -> int:
    row_counter = 0
    col_counter = 0

    nearest_row = 11
    nearest_col = 11

    if not any(target in i for i in state.board):
        return 0

    for coord, colour in state.board:
        if coord.r == target.r:
            row_counter += 1
        if coord.c == target.c:
            col_counter += 1

        if colour == PlayerColor.RED:
            rdiff = min(abs(coord.r - target.r), 11 - abs(coord.r - target.r))
            cdiff = min(abs(coord.c - target.c), 11 - abs(coord.c - target.c))
            
            if rdiff < nearest_row:
                nearest_row = rdiff

            if cdiff < nearest_col:
                nearest_col = cdiff

    # print(f"nearest_row: {nearest_row}")
    # print(f"nearest_col: {nearest_col}")

    heur = min(nearest_row + (11 - row_counter), nearest_col + (11 - col_counter))
    # print(f"heur = {heur}")
            
    return heur


def line_removal(state: State) -> State:
    '''
    Checks if any rows are columns are completely filled with blocks
    If there is, remove them from board
    Return as new state

    [Completed & Tested]
    '''
    del_row = []
    del_col = []

    # locate rows and cols to remove
    for i in range(11):
        # simultaneously check row i and col i to see if they are filled 
        row_counter = 0
        col_counter = 0
        for square in state.board:
            if square[0].r == i:
                row_counter += 1
            if square[0].c == i:
                col_counter += 1
        if (row_counter >= 11):
            del_row.append(i)
        if (col_counter >= 11):
            del_col.append(i)
    
    # remove specified rows and cols if any
    new_board = {}
    for square in state.board:
        if (square[0].r not in del_row) and (square[0].c not in del_col):
            new_board[square[0]] = square[1]
    new_state = State(state.parent, new_board, state.piece)

    return new_state
        

def astar(board, target):
    start_time = time.time()
    # get starting nodes
    start_state = State(None, board)
    print(render_board(start_state.dict_board(), target, ansi=True))

    # lists of states
    open = pq()         # stores a list of state
    closed = []         # only stores states

    open.put(start_state)


    # loop until reaching goal state
    while open.qsize() > 0:
        
        # get curr state  (i.e. state with highest priority)
        curr_state = open.get()
        # print(render_board(curr_state.dict_board(), target, ansi=True))

        closed.append(curr_state)

        # check if target is removed
        # TODO: refine expression to check if target is removed
        if not (any(target in i for i in curr_state.board)):
            print(f"FINAL ANSWER: child.f = {curr_state.f}, child.g = {curr_state.g}, child.h = {curr_state.h}")
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

            print(render_board(child.dict_board(), target, ansi=True))
            # if child on closed list
            if child in closed:
                print("child in closed\n\n\n")
                continue        # skip to next child

            # otherwise create child
            child.g = curr_state.g + 4
            child.h = heur(child, target)
            # TODO: h(x) = manhattan_dist to row/col + number of square left to fill in row/col     
            child.f = child.g + child.h
            print(f"child.f = {child.f}, child.g = {child.g}, child.h = {child.h}")

            # check if the child state is already in the queue
            for open_node in open.queue:
                # open_node is a state
                if child.board == open_node.board and child.g >= open_node.g:
                    continue

            open.put(child)

            # better = True
            # for open_node in open.queue:
            #     if child == open_node[1] and child.g > open_node[1].g:
            #         better = False
            #         continue

            # # if child state is already in the queue and has lower g(x), don't add child to queue
            # if better:
            #     open.put((child.f, child))

