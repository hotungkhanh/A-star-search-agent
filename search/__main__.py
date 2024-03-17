# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

from sys import stdin
from .core import PlayerColor, Coord, PlaceAction
from .program import search

# WARNING: Please *do not* modify any of the code in this file, as this could
#          break things in the submission environment. Failed test cases due to
#          modification of this file will not receive any marks. 
#
#          To implement your solution you should modify the `search` function
#          in `program.py` instead, as discussed in the specification.

SOLUTION_PREFIX = "$SOLUTION"


def parse_input(input: str) -> tuple[dict[Coord, PlayerColor], Coord]:
    """
    Parse input into the required data structures.
    """
    target = None
    state = {}

    try:
        for r, line in enumerate(input.strip().split("\n")):
            for c, p in enumerate(line.split(",")):
                p = p.strip()
                if line[0] != "#" and line.strip() != "" and p != "":
                    state[Coord(r, c)] = {
                        "r": PlayerColor.RED,
                        "b": PlayerColor.BLUE,
                    }[p.lower()]
                if p == "B":
                    target = Coord(r, c)

        assert target is not None, "Target coordinate 'B' not found"

        return state, target

    except Exception as e:
        print(f"Error parsing input: {e}")
        exit(1)


def print_result(sequence: list[PlaceAction] | None):
    """
    Print the given action sequence, one action per line, or "NOT_FOUND" if no
    sequence was found.
    """
    if sequence is not None:
        for action in sequence:
            print(f"{SOLUTION_PREFIX} {action}")
    else:
        print(f"{SOLUTION_PREFIX} NOT_FOUND")


def main():
    """
    Main entry point for program.
    """
    input = parse_input(stdin.read())
    sequence: list[PlaceAction] | None = search(*input)
    print_result(sequence)


if __name__ == "__main__":
    main()
