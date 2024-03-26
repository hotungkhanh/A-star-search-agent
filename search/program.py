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
    print(render_board(board, target, ansi=True))

    # state = State(None, board)
    # children = state.generate_children()

    # for child in children:
    #     print(render_board(child.board, target, ansi=True))

    # Do some impressive AI stuff here to find the solution...

    # path = (astar(board, target))
    # for coord in path:
    #     print(coord)

    return astar(board, target)

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

    def generate_children(self, target) -> list['State']:
        children = []

        # Iterate over all red cells on the board
        for coord, color in self.board.items():
            # print(f"coord, color: {coord}, {color}")
            if color == PlayerColor.RED:

                # STEP 1
                onecell = []
                adjacent_coords = [coord.down(), coord.up(), coord.left(), coord.right()]
                for adjacent_coord in adjacent_coords:
                    if adjacent_coord in self.board.keys():
                        continue
                    onecell.append([adjacent_coord])

                # print(f"onecell: {onecell}")

                # STEP 2
                twocell = []
                for one in onecell:
                    # print(f" for {one} in onecell:")
                    if one:
                        last = one[-1]
                        adjacent_coords = [last.down(), last.up(), last.left(), last.right()]
                        for adjacent_coord in adjacent_coords:
                            if adjacent_coord in self.board.keys():
                                continue
                            twocell.append(one + [adjacent_coord])

                # print(f"twocell: {twocell}")

                # STEP 3
                threecell = []
                for two in twocell:
                    if two:
                        last = two[-1]
                        adjacent_coords = [last.down(), last.up(), last.left(), last.right()]
                        for adjacent_coord in adjacent_coords:
                            if adjacent_coord in self.board.keys() or adjacent_coord in two:
                                continue
                            threecell.append(two + [adjacent_coord])

                # print(f"three: {threecell}")

                # STEP 4
                fourcell = []
                for three in threecell:
                    if three:
                        last = three[-1]
                        adjacent_coords = [last.down(), last.up(), last.left(), last.right()]
                        for adjacent_coord in adjacent_coords:
                            if adjacent_coord in self.board.keys() or adjacent_coord in three:
                                continue
                            fourcell.append(three + [adjacent_coord])

                # print(f"four: {fourcell}")

                for new_piece_coords in fourcell:
                    new_board = dict(self.board)
                    for new_coord in new_piece_coords:
                        new_board[new_coord] = PlayerColor.RED
                    new_piece = PlaceAction(*new_piece_coords)
                    new_state = State(self, new_board, new_piece)
                    new_state = line_removal(new_state, target)

                    children.append(new_state)

                    # # 3 steps
                    # for i in range(3):                  
                    #     new_adjacent_coords = [curr_coord.down(), curr_coord.up(), curr_coord.left(), curr_coord.right()]
                        
                    #     for new_adjacent_coord in new_adjacent_coords:
                    #         new_coord = new_adjacent_coord
                    #         print(f"new_coord: {new_coord}")

                    #         # Check if the new coordinate overlaps with existing cells
                    #         if new_coord in self.board.keys() or new_coord in new_piece_coords:
                    #             valid_placement = False
                    #             continue

                    #         new_piece_coords.append(new_coord)
                    #         print(f"new_piece_coords: {new_piece_coords}")
                    #         curr_cord = new_coord
                
                    #     # If the placement is valid, create a new State object representing the board after the placement
                    #     if valid_placement and len(new_piece_coords) == 4:
                    #         new_board = dict(self.board)
                    #         for new_coord in new_piece_coords:
                    #             new_board[new_coord] = PlayerColor.RED
                    #         new_piece = PlaceAction(*new_piece_coords)
                    #         new_state = State(parent=self, board=new_board, piece=new_piece)
                    #         children.append(new_state)

        return children[::-1]

def manhattan_dist(p1: Coord, p2: Coord):
    '''
    Finds shortest manhatten distance between two Coords
    Takes into account the torus nature of board
    '''

    rdiff = min(abs(p1.r - p2.r), 10 - abs(p1.r - p2.r))
    cdiff = min(abs(p1.c - p2.c), 10 - abs(p1.c - p2.c))

    return rdiff**2 + cdiff**2
    

def line_removal(state: State, target) -> State:
    '''
    Checks if any rows are columns are completely filled with blocks
    If there is, remove them from board
    Return as new state

    [Completed & Tested]
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
        if target not in curr_state.board.keys():
            path = []
            current = curr_state
            while current is not None:
                if current.piece:
                    path.append(current.piece)
                current = current.parent
            return path[::-1]
        
        # continue 

        # ------------------------------------------------------------
        # Under Construction
        # ------------------------------------------------------------
        # generate children as node
        children = curr_state.generate_children(target)

        # loop through children
        for child in children:

            print(render_board(child.board, target, ansi=True))

            # if child on closed list
            if child in closed:
                print("child in closed")
                continue        # skip to next child

            # otherwise create child
            child.g = curr_state.g + 1
            # TODO: h(x) = manhattan_dist to row/col + number of square left to fill in row/col     
            # child.h = manhattan_dist(child.position, target)
            # child.h = 0
            child.f = child.g # + child.h

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