#[macro_use] extern crate lazy_static;

const PIECE_COUNT: u32 = 10;

lazy_static! {
pub static ref PIECE_SHAPES: [Vec<(u32, u32)>; PIECE_COUNT as usize] = [
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


pub static ref TILE_COUNT: [u32; PIECE_COUNT as usize] = {
    let mut out = [0; PIECE_COUNT as usize];
    for (i, p) in PIECE_SHAPES.iter().enumerate() {
        out[i] = p.len() as u32;
    }
    return out;
};
}   // end of lazy_static

fn main() {
    let mut count = 0;
    for i in 0..3_u32.pow(PIECE_COUNT) {
        if i % 100 == 0 {
            println!("{} / {}", i, 3_u32.pow(PIECE_COUNT));
        }
        for j in 0..3_u32.pow(PIECE_COUNT) {
            let mut i = i;
            let mut j = j;
            let mut compatible = true;
            let mut tiles_above = 0;
            let mut tiles_below = 0;
            let mut t = 0;
            while i > 0 {
                tiles_above += TILE_COUNT[t] * (i % 3);
                tiles_below += TILE_COUNT[t] * (j % 3);
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
