import datetime
from z3 import *

PIECE_SHAPES = (
    # 0
    ((0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2), (0, 3), (1, 3), (2, 3)),
    # 1
    ((1, 0), (1, 1), (1, 2), (0, 3), (1, 3)),
    # 2
    ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2), (2, 2), (1, 3), (2, 3)),
    # 3
    ((0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2), (0, 3), (1, 3), (2, 3)),
    # 4
    ((1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2), (1, 3), (2, 3)),
    # 5
    ((0, 0), (1, 0), (2, 0), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)),
    # 6
    ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (0, 3), (1, 3)),
    # 7
    ((0, 0), (0, 1), (1, 1), (1, 2), (0, 3), (1, 3), (2, 3)),
    # 8
    ((0, 0), (1, 0), (0, 1), (1, 1), (1, 2), (2, 2), (1, 3), (2, 3)),
    # 9
    ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)),
)

def rotated(i, ts):
    M = {0: [1, 0, 0, 1],
         1: [0, 1, -1, 0],
         2: [-1, 0, 0, -1],
         3: [0, -1, 1, 0]}[i]
    return [(M[0] * x + M[1] * y, M[2] * x + M[3] * y)
            for (x, y) in ts]

class Piece(object):
    def __init__(self, index):

        self.x = Int('x_%s' % index)
        self.y = Int('y_%s' % index)
        self.rot = Int('rot_%s' % index)
        self.z = Int('z_%s' % index)

        self.score = index // 2
        self.name = index

        self.tiles = PIECE_SHAPES[index // 2]
        self.adjacent = list(set([(tx + dx, ty + dy)
                for (dx, dy) in ((0, 1), (0, -1), (1, 0), (-1, 0))
                for tx, ty in self.tiles])
            .difference(self.tiles))

    def any(self, other, self_attr, other_attr=None):
        if other_attr is None:
            other_attr = self_attr
        cond = False
        for i in range(2):
            for j in range(2):
                cond = If(
                    And(self.rot == i, other.rot == j),
                    Or([And(ax + self.x == bx + other.x,
                            ay + self.y == by + other.y)
                       for (ax, ay) in rotated(i, getattr(self, self_attr))
                       for (bx, by) in rotated(j, getattr(other, other_attr))]),
                    cond)
        return cond

    def num_over(self, other):
        cond = 0
        for i in range(2):
            for j in range(2):
                cond = If(
                    And(self.rot == i, other.rot == j),
                    Sum([If(And(ax + self.x == bx + other.x,
                                ay + self.y == by + other.y), 1, 0)
                       for (ax, ay) in rotated(i, self.tiles)
                       for (bx, by) in rotated(j, other.tiles)]),
                    cond)
        return If(self.z == other.z + 1, cond, 0)

    def is_overlapping(self, other):
        return And(self.z == other.z, self.any(other, 'tiles'))

    def is_over(self, other):
        return And(self.z == other.z + 1, self.any(other, 'tiles'))

    def is_adjacent(self, other):
        option_a = len(self.tiles) + len(other.adjacent)
        option_b = len(self.adjacent) + len(other.tiles)
        return And(self.z == other.z,
            self.any(other, 'tiles', 'adjacent') if option_a < option_b else
            self.any(other, 'adjacent', 'tiles'))

pieces = [Piece(i) for i in range(4)]

solver = Optimize()
start = datetime.datetime.now()

# Add a general no-overlapping constraint
solver.add(Not(Or([
    pieces[i].is_overlapping(pieces[j])
    for i in range(len(pieces))
    for j in range(0, i)])))

for p in pieces:
    solver.add(p.rot < 4)
    others = [q for q in pieces if q != p]

    # Each piece must either be lonely on its layer, or have a neighbor
    solver.add(Or(
        And([p.z != q.z for q in others]),
        Or([p.is_adjacent(q) for q in others])))

    over = [p.num_over(q) for q in others]
    # Each piece must be over more than one other piece
    # and completely supported
    solver.add(Or(p.z == 0, And(PbGe([(o > 0, 1) for o in over], 2),
                                Sum(over) == len(p.tiles))))

score = Sum([p.z * p.score for p in pieces])
print("About to maximize model")
solver.maximize(score)
print("Built model in %s" % (datetime.datetime.now() - start))

start = datetime.datetime.now()
if solver.check() == CheckSatResult(1):
    model = solver.model()
    print("Solved in %s with score %s" % (datetime.datetime.now() - start, model.eval(score)))
else:
    raise RuntimeError("Model is unsat")

tiles = {}
for p in pieces:
    for (x, y) in rotated(model.eval(p.rot).as_long(), p.tiles):
        tiles[(x + model.eval(p.x).as_long(),
               y + model.eval(p.y).as_long(),
               model.eval(p.z).as_long())] = p.name // 2

COLOR = [
    '\033[7m',      # 0: bright white
    '\033[47m',     # 1: slightly greyer
    '\033[43m',     # 2: orange
    '\033[103m'     # 3: bright orange
    '\033[42m'      # 4: green
    '\033[44m'      # 5: blue
    '\033[104m'     # 6: blue-grey
    '\033[45m'      # 7: purple
    '\033[105m'     # 8: pink
    '\033[101m'     # 9: red
]
xmin = min([x for (x, _, _) in tiles.keys()])
xmax = max([x for (x, _, _) in tiles.keys()])
ymin = min([y for (_, y, _) in tiles.keys()])
ymax = max([y for (_, y, _) in tiles.keys()])
zmin = min([z for (_, _, z) in tiles.keys()])
zmax = max([z for (_, _, z) in tiles.keys()])

for z in range(zmin, zmax + 1):
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if x in [xmin, xmax] or y in [ymin, ymax]:
                s = '. '
            else:
                s = '  '

            if (x, y, z) in tiles:
                t = tiles[(x, y, z)]
                print('%s%s%s' % (COLOR[t], s, '\33[0m'), end='')
            else:
                print(s, end='')
        print('')
    print('')
