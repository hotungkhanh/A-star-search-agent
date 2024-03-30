from core import PlayerColor, Coord, PlaceAction
from collections import defaultdict as dd
from queue import PriorityQueue as pq

q = pq()
q.put(["hi"])
q.put(["bye"])

# print(q.get())
# print(q.get())
# print(q.empty())

print("Coord Hash")
c1 = Coord(5, 2)
c2 = Coord(5, 2)
c3 = Coord(2, 5)
print(hash(c1))
print(hash(c2))
print(hash(c1) == hash(c2))
# True: same coord hashes to same value

print(hash(c1) == hash(c3))
# False: diff coord hashes to diff value

print("\nPlayer Colour Hash")
print(hash(PlayerColor.RED))
print(hash(PlayerColor.BLUE))
# player colours hash to diff values

print("\n(Coord, Colour) Hash")
d1 = (c1, PlayerColor.RED)
d2 = (c3, PlayerColor.BLUE)
d3 = (Coord(6, 4), PlayerColor.RED)

print(hash(d1))
print(hash(d2))

print("\nTuple of (Coord, Colour) Hash")
e1 = (d1, d2, d3)
e2 = (d2, d1, d3)
print(e1)
print(hash(e1) == hash(e2))
# Tuple of Tuples is hashable

t1 = (3, 2)
print()


tt1 = ((3, 2), (1, 2))

for one, two in tt1:
    print(one)