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

class Piece(object):
    def __init__(self, index, name):
        self.pattern = PIECE_SHAPES[index]

        self.x = BitVec('x_%s' % name, 7)
        self.y = BitVec('y_%s' % name, 7)
        self.rot = BitVec('rot_%s' % name, 2)
        self.z = BitVec('z_%s' % name, 4)
        self.score = index
        self.name = name

        self.tiles = []
        for (tx, ty) in self.pattern:
            tx_ = If(self.rot == 0, BitVecVal(tx, 8),
                  If(self.rot == 1, BitVecVal(ty, 8),
                  If(self.rot == 2, BitVecVal(-tx, 8),
                                    BitVecVal(-ty, 8))))
            ty_ = If(self.rot == 0, BitVecVal(ty, 8),
                  If(self.rot == 1, BitVecVal(-tx, 8),
                  If(self.rot == 2, BitVecVal(-ty, 8),
                                    BitVecVal(-tx, 8))))
            self.tiles.append((ZeroExt(1, self.x) + tx_,
                               ZeroExt(1, self.y) + ty_))

        adj = set([(tx + dx, ty + dy)
            for (dx, dy) in ((0, 1), (0, -1), (1, 0), (-1, 0))
            for tx, ty in self.pattern])
        self.adj = []
        for (tx, ty) in adj:
            tx_ = If(self.rot == 0, BitVecVal(tx, 8),
                  If(self.rot == 1, BitVecVal(ty, 8),
                  If(self.rot == 2, BitVecVal(-tx, 8),
                                    BitVecVal(-ty, 8))))
            ty_ = If(self.rot == 0, BitVecVal(ty, 8),
                  If(self.rot == 1, BitVecVal(-tx, 8),
                  If(self.rot == 2, BitVecVal(-ty, 8),
                                    BitVecVal(-tx, 8))))
            self.adj.append((ZeroExt(1, self.x) + tx_,
                             ZeroExt(1, self.y) + ty_))

    def adjacent(this, other):
        return And(this.z == other.z,
               Or([And(ax == bx, ay == by)
                   for (ax, ay) in this.tiles
                   for (bx, by) in other.adj]))

    def overlapping(this, other):
        return And(this.z == other.z,
               Or([And(ax == bx, ay == by)
                   for (ax, ay) in this.tiles
                   for (bx, by) in other.tiles]))

    def over(this, other):
        ''' Returns true if any of these tiles are over the other piece
        '''
        return And(this.z == other.z + 1,
               Or([And(ax == bx, ay == by)
                   for (ax, ay) in this.tiles
                   for (bx, by) in other.tiles]))

    def over_two(this, others):
        ''' Returns true if this piece is over at least two other pieces
        '''
        return PbGe([(this.over(o), 1) for o in others], 2)

    def supported(this, others):
        ''' Returns true if all of this piece's tiles are supported
        '''
        conditions = []
        for (ax, ay) in this.tiles:
            supported = []
            for o in others:
                supported.append(
                    And(this.z == o.z + 1,
                        Or([And(ax == bx, ay == by)
                            for (bx, by) in o.tiles])))
            conditions.append(Or(supported))
        return And(conditions)

    def lonely(this, others):
        ''' Returns true if this is the only piece with the given Z value
        '''
        return And([Not(this.z == o.z) for o in others])

bag = [i // 2 for i in range(5)]
pieces = [Piece(b, i) for (i, b) in enumerate(bag)]

s = Optimize()
start = datetime.datetime.now()
for (i, p) in enumerate(pieces):
    print("%i / %i" % (i + 1, len(pieces)))
    # Add extra constraints to the first piece, to narrow the search space
    if i == 0:
        s.add(p.rot == 0)
        s.add(p.x == 0)
        s.add(p.y == 0)

    others = [o for o in pieces if o != p]
    s.add(Or(p.z == 0, And(p.over_two(others), (p.supported(others)))))
    s.add(And([Not(p.overlapping(o)) for o in others]))
    s.add(Or(p.lonely(others), Or([p.adjacent(o) for o in others])))

score = Sum([p.z * p.score for p in pieces])
s.maximize(score)
print("Built model in %s" % (datetime.datetime.now() - start))

start = datetime.datetime.now()
s.check()
model = s.model()
print("Solved in %s with score %s" % (datetime.datetime.now() - start, model.eval(score)))

tiles = {}
for p in pieces:
    for (x, y) in p.tiles:
        tiles[(model.eval(x).as_signed_long(),
               model.eval(y).as_signed_long(),
               model.eval(p.z).as_signed_long())] = p.score

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
