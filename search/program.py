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

    print(board)
    print()

    # Do some impressive AI stuff here to find the solution...
    # ...
    # ... (your solution goes here!)
    # ...

    # compute all possible starting nodes 

    answer = uniform_cost_search(board, 
                                [Coord(r=8, c=8), Coord(r=9, c=7), Coord(r=9, c=9)], 
                                [Coord(r=1, c=5), Coord(r=2, c=3), Coord(r=2, c=5)])
    print(answer)
    # Here we're returning "hardcoded" actions as an example of the expected
    # output format. Of course, you should instead return the result of your
    # search algorithm. Remember: if no solution is possible for a given input,
    # return `None` instead of a list.
    return [
        PlaceAction(Coord(2, 5), Coord(2, 6), Coord(3, 6), Coord(3, 7)),
        PlaceAction(Coord(1, 8), Coord(2, 8), Coord(3, 8), Coord(4, 8)),
        PlaceAction(Coord(5, 8), Coord(6, 8), Coord(7, 8), Coord(8, 8)),
    ]

def adjacent(coord: Coord):
    adjacent_nodes = []
    adjacent_nodes.append(coord.down())
    adjacent_nodes.append(coord.up())
    adjacent_nodes.append(coord.left())
    adjacent_nodes.append(coord.right())
    return adjacent_nodes

def uniform_cost_search(board: dict[Coord, PlayerColor], 
                        target: list[Coord], 
                        start_nodes: list[Coord]):
    '''
    board: a dict with key = Coord, value = colour Enum
    target: a Coord 
    start: an array of Coord
    '''

    answer = 10**8     # stores what?
    

    for start in start_nodes:
        print()
        print("--New Start Node--")
        pq = []
        pq.append([1, start])       #[path cost, coordinate]

        visited = dd(lambda: None)  # dictionary of expanded/visited nodes
        cost = dd(lambda: None)
        cost[start] = 0

        parent = {}

        while (len(pq) > 0):
            pq = sorted(pq)         # largest to smallest
            curr_node = pq.pop(0)
            print(f" curr_node: {curr_node}")

            # check if curr node is the target
            if curr_node[1] in target:
                if answer > curr_node[0]:
                    answer = curr_node[0]
                print(answer)
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