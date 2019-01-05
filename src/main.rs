#[macro_use] extern crate lazy_static;

use std::collections::HashSet;

// PIECE_TYPES is the number of unique piece shapes
const PIECE_TYPES: u8 = 10;

// This is the number of individual pieces per shape
const NUM_COPIES: u8 = 2;

// PIECE_COUNT is the number of pieces; two per type
const PIECE_COUNT: u8 = NUM_COPIES * PIECE_TYPES;

lazy_static! {
pub static ref PIECE_SHAPES: [Vec<(u32, u32)>; PIECE_TYPES as usize] = [
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
    fn from_int(t: u16) -> Stack {
        assert!(t < (NUM_COPIES as u16 + 1).pow(PIECE_TYPES as u32));
        let mut out = Stack(0);
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

    // Sets the z position a particular piece
    fn place(&mut self, i: u8, level: u8) {
        assert!(i < PIECE_TYPES);
        for j in (i * NUM_COPIES)..((i + 1) * NUM_COPIES) {
            if self.get_z(i * NUM_COPIES + j).is_none() {
                self.0 |= (level as StackInt + 1) <<
                          (((i as StackInt * NUM_COPIES as StackInt) + j as StackInt) * Z_BITS);
                break;
            }
        }
    }

    // Returns the z position a particular piece
    fn get_z(&self, i: u8) -> Option<u8> {
        assert!(i < PIECE_COUNT);
        (((self.0 >> (i as StackInt * Z_BITS)) & 0xF) as u8).checked_sub(1)
    }

    // Returns the number of pieces with shape t that have been placed
    // (this is always in the range 0 to 2, inclusive)
    fn placed(&self, t: u8) -> u8 {
        assert!(t < PIECE_TYPES);
        (0..NUM_COPIES).filter_map(
            |i| self.get_z(t * NUM_COPIES + i)).count() as u8
    }

    // Returns the total height of this stack, assuming it is well-constructed
    // If there no pieces placed, returns 0
    fn height(&self) -> u8 {
        (0..PIECE_COUNT).filter_map(|i| self.get_z(i))
                        .max()
                        .map(|i| i + 1)
                        .unwrap_or(0)
    }
}

fn main() {
    let mut stacks = vec![
        (0..(NUM_COPIES as u16 + 1).pow(PIECE_TYPES as u32))
        .map(Stack::from_int)
        .collect::<HashSet<_>>()];

    let mut count = 0;
    for i in 0..3_u16.pow(PIECE_TYPES as u32) {
        if i % 100 == 0 {
            println!("{} / {}", i, 3_u16.pow(PIECE_TYPES as u32));
        }
        for j in 0..3_u16.pow(PIECE_TYPES as u32) {
            let mut i = i;
            let mut j = j;
            let mut compatible = true;
            let mut tiles_above = 0;
            let mut tiles_below = 0;
            let mut t = 0;
            while i > 0 {
                tiles_above += TILE_COUNT[t] as u16 * (i % 3);
                tiles_below += TILE_COUNT[t] as u16 * (j % 3);
                compatible &= (i % 3) + (j % 3) <= 2;
                i /= 3;
                j /= 3;
                t += 1;
            }
            compatible &= tiles_below >= tiles_above;
            if compatible {
                count += 1;
            }
        }
    }
    println!("Got count {}", count);
}
