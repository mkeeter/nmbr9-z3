#[macro_use] extern crate lazy_static;
#[macro_use] extern crate itertools;

extern crate array_init;
use array_init::array_init;

extern crate z3;
use z3::*;

use std::collections::HashSet;
use std::fmt;

// PIECE_TYPES is the number of unique piece shapes
const PIECE_TYPES: u8 = 10;

// This is the number of individual pieces per shape
const NUM_COPIES: u8 = 2;

// PIECE_COUNT is the number of pieces; two per type
const PIECE_COUNT: u8 = NUM_COPIES * PIECE_TYPES;

lazy_static! {
pub static ref PIECE_SHAPES: [Vec<(i32, i32)>; PIECE_TYPES as usize] = [
    // 0
    vec![(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2), (0, 3), (1, 3), (2, 3)],
    // 1
    vec![(1, 0), (1, 1), (1, 2), (0, 3), (1, 3)],
    // 2
    vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2), (2, 2), (1, 3), (2, 3)],
    // 3
    vec![(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2), (0, 3), (1, 3), (2, 3)],
    // 4
    vec![(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, 2), (1, 3), (2, 3)],
    // 5
    vec![(0, 0), (1, 0), (2, 0), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)],
    // 6
    vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (0, 3), (1, 3)],
    // 7
    vec![(0, 0), (0, 1), (1, 1), (1, 2), (0, 3), (1, 3), (2, 3)],
    // 8
    vec![(0, 0), (1, 0), (0, 1), (1, 1), (1, 2), (2, 2), (1, 3), (2, 3)],
    // 9
    vec![(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3)],
];


pub static ref TILE_COUNT: [u8; PIECE_TYPES as usize] = {
    let mut out = [0; PIECE_TYPES as usize];
    for (i, p) in PIECE_SHAPES.iter().enumerate() {
        out[i] = p.len() as u8;
    }
    return out;
};

pub static ref ADJACENT_TILES: [Vec<(i32, i32)>; PIECE_TYPES as usize] = {
    let mut out: [Vec<_>; PIECE_TYPES as usize] = array_init(|_| vec![]);
    for (i, p) in PIECE_SHAPES.iter().enumerate() {
        let pts = p.iter().cloned().collect::<HashSet<_>>();
        let adj = iproduct!(p.iter(), [(1, 0), (-1, 0), (0, 1), (0, -1)].iter())
            .map(|((px, py), (dx, dy))| (px + dx, py + dy))
            .collect::<HashSet<_>>();
        out[i] = adj.difference(&pts).cloned().collect();
    }
    out
};

pub static ref COLLISION_TILES: [Vec<(i32, i32)>; PIECE_TYPES as usize] = {
    let mut out: [Vec<_>; PIECE_TYPES as usize] = array_init(|_| vec![]);
    for (i, p) in PIECE_SHAPES.iter().enumerate() {
        let pts = p.iter().cloned().collect::<HashSet<_>>();
        out[i] = p.iter()
            .filter(|(x, y)|
                [(1, 0), (-1, 0), (0, 1), (0, -1)].iter()
                .any(|(dx, dy)| !pts.contains(&(x + dx, y + dy))))
            .cloned()
            .collect();
    }
    out
};

}   // end of lazy_static

////////////////////////////////////////////////////////////////////////////////

/*
 *  A Stack stores which level each of 20 pieces is placed on, using a bitmask
 *
 *  Unplaced pieces are marked by being at level 0.
 *
 *  Each piece has a 4-bit z level, since we can't build a > 10-layer stack
 */
type StackInt = i128;
const Z_BITS: StackInt = 4;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct Stack(StackInt);

impl Stack {
    fn new() -> Stack {
        Stack(0)
    }

    fn from_int(t: u16) -> Stack {
        assert!(t < (NUM_COPIES as u16 + 1).pow(PIECE_TYPES as u32));
        let mut out = Self::new();
        let mut t = t;
        for i in 0.. {
            if t == 0 {
                break;
            }
            for _ in 0..(t % (NUM_COPIES as u16 + 1)) {
                out.place(i, 0);
            }
            t /= NUM_COPIES as u16 + 1;
        }
        out
    }

    // Places an instance of a particular piece type,
    // maintaining sorted order among that type so that
    // place doesn't lead to order-dependent values.
    fn place(&mut self, t: u8, level: u8) {
        assert!(t < PIECE_TYPES);

        let mut levels = [0xFF; NUM_COPIES as usize];
        let mut n = 0;
        while n != NUM_COPIES {
            if let Some(z) = self.get_z(t * NUM_COPIES + n) {
                levels[n as usize] = z;
                n += 1;
            } else {
                break;
            }
        }
        assert!(n != NUM_COPIES);
        levels[n as usize] = level;
        levels.sort();

        for i in 0..=n {
            self.set_z(t * NUM_COPIES + i, levels[i as usize]);
        }
    }

    // Returns the z position a particular piece
    fn get_z(&self, i: u8) -> Option<u8> {
        assert!(i < PIECE_COUNT);
        ((self.0 >> (i as StackInt * Z_BITS)) as u8 & 0xF).checked_sub(1)
    }

    // Returns the z position a particular piece
    // This is a low-level utility; you should probably call place() instead,
    // which deduplicates pieces to ensure consistent ordering.
    fn set_z(&mut self, i: u8, z: u8) {
        assert!(i < PIECE_COUNT);
        assert!(z < ((1 << Z_BITS) - 1));

        let offset = i as StackInt * Z_BITS;
        self.0 &= !(0xF << offset);
        self.0 |=  (z as StackInt + 1) << offset;
    }

    // Returns the number of pieces with shape t that have been placed
    // (this is always in the range 0 to NUM_COPIES, inclusive)
    fn placed(&self, t: u8) -> u8 {
        assert!(t < PIECE_TYPES);
        (0..NUM_COPIES)
            .filter_map(|i| self.get_z(t * NUM_COPIES + i))
            .count() as u8
    }

    // Returns the total height of this stack, assuming it is well-constructed
    fn height(&self) -> u8 {
        (0..PIECE_COUNT).filter_map(|i| self.get_z(i))
                        .max()
                        .map(|i| i + 1)
                        .unwrap_or(0)
    }

    // Returns the area of a particular layer
    fn area(&self, z: u8) -> u32 {
        (0..PIECE_COUNT)
            .filter(|i| self.get_z(*i).unwrap_or(0xFF) == z)
            .map(|i| TILE_COUNT[(i / NUM_COPIES) as usize] as u32)
            .sum()
    }

    // Tries to place this stack onto the bottom stack
    //
    // Returns None if we run out of pieces, but performs no
    // other validity checking.
    fn onto(&self, bottom: &Stack) -> Option<Stack> {
        // First sanity-check: return None if we run out of pieces
        for t in 0..PIECE_TYPES {
            if self.placed(t) + bottom.placed(t) > NUM_COPIES {
                return None;
            }
        }

        let mut out = Stack::new();
        let h = bottom.height();
        for t in 0..PIECE_TYPES {
            for i in 0..NUM_COPIES {
                if let Some(z) = self.get_z(t * NUM_COPIES + i) {
                    out.place(t, z + h);
                }
                if let Some(z) = bottom.get_z(t * NUM_COPIES + i) {
                    out.place(t, z);
                }
            }
        }
        return Some(out);
    }

    fn validate(&self) -> bool {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let int_sort = ctx.int_sort();
        let int_set_sort = ctx.set_sort(&int_sort);

        // Build free variables for every active piece
        let placements = (0..PIECE_COUNT)
            .filter(|&i| self.get_z(i).is_some())
            .map(|i| Placement::new(i, &ctx))
            .collect::<Vec<_>>();

        // Build Z-sorted lists of tile-sets per layer
        let h = self.height();
        let mut layers = Vec::new();
        for _ in 0..h {
            layers.push(Vec::new());
        }
        for p in placements.iter() {
            layers[self.get_z(p.i).unwrap() as usize].push(
                TileSets::new(p.i, p, &ctx, &int_set_sort));
        }

        // Add constraints to the solver
        let mut solver = Solver::new(&ctx);
        let empty_set = ctx.named_const("empty_set", &int_set_sort);

        // Bounds on coordinates
        let lower = ctx.from_i64(-50 as i64);
        let upper = ctx.from_i64( 50 as i64);
        let zero = ctx.from_i64(0 as i64);
        let four = ctx.from_i64(4 as i64);
        for p in placements.iter() {
            solver.assert(&p.x.gt(&lower));
            solver.assert(&p.x.lt(&upper));

            solver.assert(&p.y.gt(&lower));
            solver.assert(&p.y.lt(&upper));

            solver.assert(&p.rot.ge(&zero));
            solver.assert(&p.rot.lt(&four));
        }

        for (z, layer) in layers.iter().enumerate() {
            for (i, piece) in layer.iter().enumerate() {
                // Each piece must not collide with any other pieces
                let collisions: Vec<&_> = layer.iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, p)| &p.collision)
                    .collect();
                let collision_set = empty_set.set_union(&collisions);
                let my_collisions = piece.collision.set_intersect(&[&collision_set]);
                solver.assert(&my_collisions.set_subset(&empty_set));

                // Each piece must be adjacent to at least one piece
                let adjacents: Vec<&_> = layer.iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, p)| &p.adjacent)
                    .collect();
                let adjacent_set = empty_set.set_union(&adjacents);
                let my_adjacents = piece.collision.set_intersect(&[&adjacent_set]);
                solver.assert(&my_adjacents.set_subset(&empty_set).not());
            }
        }

        for (below, above) in layers.iter().zip(layers.iter().nth(1)) {
            // Each piece must be fully supported
            let below_set = empty_set.set_union(
                &below.iter()
                    .map(|p| &p.tiles)
                    .collect::<Vec<&_>>());
            let above_set = empty_set.set_union(
                &above.iter()
                    .map(|p| &p.tiles)
                    .collect::<Vec<&_>>());

            solver.assert(
                &below_set.set_subset(
                    &below_set.set_intersect(&[&above_set])));

            for piece in above.iter() {
                let is_above: Vec<Ast> = below.iter()
                    .map(|b| piece.tiles.set_intersect(&[&b.tiles])
                                        .set_subset(&empty_set)
                                        .not())
                    .collect();
                let weights = vec![1; below.len()];
            }
            // Each piece must be over at least two pieces
        }

        println!("Solving...");
        let success = solver.check();
        println!("Success: {}", success);

        success
    }
}

////////////////////////////////////////////////////////////////////////////////

struct Placement<'a> {
    i: u8,
    x: Ast<'a>,     /* int */
    y: Ast<'a>,     /* int */
    rot: Ast<'a>,   /* int, 0-3 */
}

impl<'a> Placement<'a> {
    fn new(i: u8, ctx: &'a Context) -> Placement<'a> {
        let x = ctx.named_int_const(&format!("x_{}", i));
        let y = ctx.named_int_const(&format!("y_{}", i));
        let rot = ctx.named_int_const(&format!("rot_{}", i));

        Placement {
            i: i,
            x: x,
            y: y,
            rot: rot,
        }
    }
}

struct TileSets<'a> {
    tiles: Ast<'a>,     /* set */
    collision: Ast<'a>, /* set */
    adjacent: Ast<'a>,  /* set */
}

impl<'a> TileSets<'a> {
    fn new(i: u8, p: &'a Placement,
           ctx: &'a Context, int_set_sort: &'a Sort) -> TileSets<'a> {
        let t = (i / NUM_COPIES) as usize;

        // Build our collision sets
        let tiles = Self::new_set(
            &format!("tiles_{}", i),
            &PIECE_SHAPES[t],
            &p.x, &p.y, &p.rot, ctx, int_set_sort);

        let collision = Self::new_set(
            &format!("collision_{}", i),
            &COLLISION_TILES[t],
            &p.x, &p.y, &p.rot, ctx, int_set_sort);

        let adjacent = Self::new_set(
            &format!("adjacent_{}", i),
            &ADJACENT_TILES[t],
            &p.x, &p.y, &p.rot, ctx, int_set_sort);

        TileSets {
            tiles: tiles,
            collision: collision,
            adjacent: adjacent,
        }
    }

    fn new_set(name: &str, pts: &[(i32, i32)],
               dx: &'a Ast, dy: &'a Ast, rot: &'a Ast,
               ctx: &'a Context, int_set_sort: &'a Sort) -> Ast<'a>
    {
        let mut set = ctx.named_const(name, int_set_sort);

        for &(x, y) in pts.iter() {
            // We shift both coordinates by 50, then scale y by 100
            // to get a distinct mapping from R^2 => R
            let px = ctx.from_i64( x as i64);
            let nx = ctx.from_i64(-x as i64);
            let py = ctx.from_i64( y as i64);
            let ny = ctx.from_i64(-y as i64);

            let r0 = ctx.from_i64(0);
            let r1 = ctx.from_i64(1);
            let r2 = ctx.from_i64(2);

            let offset = ctx.from_i64(50);
            let scale = ctx.from_i64(100);

            let x_ = dx.add(&[&offset,
                &rot._eq(&r0)
                    .ite(&px, &rot._eq(&r1)
                    .ite(&py, &rot._eq(&r2)
                    .ite(&nx,
                         &ny)))]);

            let y_ = dy.add(&[&offset,
                &rot._eq(&r0)
                    .ite(&py, &rot._eq(&r1)
                    .ite(&nx, &rot._eq(&r2)
                    .ite(&ny,
                         &px)))]);

            let q = x_.add(&[&y_.mul(&[&scale])]);
            set = set.set_add(&q);
        }
        return set;
    }
}

////////////////////////////////////////////////////////////////////////////////

fn main() {
    // stacks[i] is all of the stackups of height i + 1
    let mut stacks = vec![
        (0..(NUM_COPIES as u16 + 1).pow(PIECE_TYPES as u32))
        .map(Stack::from_int)
        .collect::<HashSet<_>>()];

    let s = Stack::from_int(2);
    s.validate();

    let mut count = 0;
    for (i, a) in stacks.last().unwrap().iter().enumerate() {
        if i % 100 == 0 {
            println!("{} / {}", i, (NUM_COPIES as u16 + 1).pow(PIECE_TYPES as u32));
        }
        for b in stacks[0].iter() {
            if let Some(c) = a.onto(b) {
                count += 1;
            }
        }
    }
    println!("{}", count);

    let mut count = 0;
    for i in 0..(NUM_COPIES as u16 + 1).pow(PIECE_TYPES as u32) {
        if i % 100 == 0 {
            println!("{} / {}", i, (NUM_COPIES as u16 + 1).pow(PIECE_TYPES as u32));
        }
        for j in 0..(NUM_COPIES as u16 + 1).pow(PIECE_TYPES as u32) {
            let a = Stack::from_int(i);
            let b = Stack::from_int(j);
            let i_ = i;
            let j_ = j;

            let mut i = i;
            let mut j = j;
            let mut compatible = true;
            let mut pieces_below = 0;
            let mut tiles_above = 0;
            let mut tiles_below = 0;
            let mut t = 0;
            let n = NUM_COPIES as u16 + 1;
            while i > 0 || j > 0 {
                pieces_below += j % n;
                tiles_above += TILE_COUNT[t] as u16 * (i % n);
                tiles_below += TILE_COUNT[t] as u16 * (j % n);
                compatible &= (i % n) + (j % n) < n;
                i /= n;
                j /= n;
                t += 1;
            }
            compatible &= tiles_below >= tiles_above;

            if a.onto(&b).is_some() && !compatible {
                println!("Error at {} onto {}; {} onto {}", i_, j_, a.0, b.0);
                println!("tiles: {} {}", tiles_above, tiles_below);
            }
            if compatible {
                count += 1;
            }
        }
    }
    println!("Got count {}", count);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_from_int() {
        let s = Stack::from_int(0);
        assert_eq!(s.0, 0);

        let s = Stack::from_int(1);
        assert_eq!(s.0, 1);

        let s = Stack::from_int(2);
        assert_eq!(s.0, 1 | (1 << Z_BITS));

        let s = Stack::from_int(NUM_COPIES as u16 + 1);
        assert_eq!(s.0, (1 << (NUM_COPIES as StackInt * Z_BITS)));
    }

    #[test]
    fn stack_place() {
        let mut s = Stack::new();
        s.place(0, 0);
        assert_eq!(s.0, 1);
        s.place(0, 0);
        assert_eq!(s.0, 1 | (1 << Z_BITS));

        // Test order-independence
        let mut a = Stack::new();
        a.place(4, 3);
        a.place(4, 5);
        a.place(5, 1);
        a.place(5, 2);

        let mut b = Stack::new();
        b.place(5, 2);
        b.place(4, 5);
        b.place(4, 3);
        b.place(5, 1);

        assert_eq!(a, b);
    }

    #[test]
    fn stack_get_z() {
        let mut s = Stack::new();
        s.place(0, 0);
        assert_eq!(s.get_z(0), Some(0));
        assert_eq!(s.get_z(1), None);

        s.place(5, 6);
        assert_eq!(s.get_z(5 * NUM_COPIES), Some(6));
    }

    #[test]
    fn stack_placed() {
        let mut s = Stack::new();
        s.place(0, 0);
        s.place(1, 2);
        s.place(1, 3);
        assert_eq!(s.placed(0), 1);
        assert_eq!(s.placed(1), 2);
        assert_eq!(s.placed(2), 0);
    }

    #[test]
    fn stack_onto() {
        let mut a = Stack::new();
        a.place(0, 0);
        a.place(1, 2);
        a.place(1, 3);

        let mut b = Stack::new();
        b.place(0, 0);

        let c = a.onto(&b);
        assert!(c.is_some());
        assert_eq!(c.unwrap().height(), a.height() + b.height());

        let mut b = Stack::new();
        b.place(1, 0);
        assert_eq!(a.onto(&b), None);
    }
}
