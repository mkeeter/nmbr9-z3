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


pub static ref PIECE_AREA: [u8; PIECE_TYPES as usize] = {
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

    // Tries to place this stack onto the bottom stack
    //
    // Returns None if we run out of pieces, but performs no
    // other validity checking.
    fn onto(&self, bottom: &Stack) -> Option<Stack> {
        if bottom.0 == 0 {
            return None;
        }

        // First sanity-check: return None if we run out of pieces
        for t in 0..PIECE_TYPES {
            if self.placed(t) + bottom.placed(t) > NUM_COPIES {
                return None;
            }
        }
        // Second sanity-check: return None if there's an area mismatch
        let my_base_area: u32 = (0..PIECE_COUNT)
            .filter(|i| self.get_z(*i).unwrap_or(0xFF) == 0)
            .map(|i| PIECE_AREA[(i / NUM_COPIES) as usize] as u32)
            .sum();

        let h = bottom.height();
        let bottom_top_area: u32 = (0..PIECE_COUNT)
            .filter(|i| bottom.get_z(*i).unwrap_or(0xFF) == h - 1)
            .map(|i| PIECE_AREA[(i / NUM_COPIES) as usize] as u32)
            .sum();

        if bottom_top_area < my_base_area {
            return None;
        }

        let mut out = Stack::new();
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
}

fn main() {
    // stacks[i] is all of the stackups of height i + 1
    let mut stacks = vec![
        (0..(NUM_COPIES as u16 + 1).pow(PIECE_TYPES as u32))
        .map(Stack::from_int)
        .collect::<HashSet<_>>()];

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
                tiles_above += PIECE_AREA[t] as u16 * (i % n);
                tiles_below += PIECE_AREA[t] as u16 * (j % n);
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
