#[macro_use] extern crate itertools;
use rayon::prelude::*;

extern crate z3;
use z3::*;

use std::collections::{HashSet, HashMap};
use std::fmt;

fn rotated(ts: &[(i32, i32)], rot: usize) -> Vec<(i32, i32)> {
    let m = match rot {
        0 => &[ 1,  0,  0,  1],
        1 => &[ 0,  1, -1,  0],
        2 => &[-1,  0,  0, -1],
        3 => &[ 0, -1,  1,  0],
        _ => unimplemented!("oh no")
    };
    ts.iter().map(|(x, y)| (m[0] * x + m[1] * y, m[2] * x + m[3] * y)).collect()
}

fn to_vec<T>(vs: &[Ast], model: &Model, f: &Fn(Ast) -> Option<T>) -> Vec<T>
    where T: Default
{
    vs.iter().map(
        |x| model.eval(x)
            .and_then(|i| f(i))
            .unwrap_or(T::default())
        ).collect()
}

fn main() {
    #[allow(non_snake_case)]
    let PIECE_SHAPES: [Vec<(i32, i32)>; 10] = [
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

    // We need to check every piece + rotation against every other
    // piece + rotation to build our adjacency and overlapping tables.
    let (is_overlapping, is_adjacent) = iproduct!(
            PIECE_SHAPES.iter().enumerate(), 0..4,
            PIECE_SHAPES.iter().enumerate(), 0..4)
        .par_bridge()
        .map(|((i, pi), ri, (j, pj), rj)| {
            let pi = rotated(pi, ri);
            let pj = rotated(pj, rj);

            // Find coordinates where the two pieces are overlapping
            let over = iproduct!(-10..=10, -10..=10).filter(|(dx, dy)| {
                let si: HashSet<_> = pi.iter()
                    .map(|(x, y)| (x + dx, y + dy))
                    .collect();
                let sj: HashSet<_> = pj.iter().cloned().collect();
                si.intersection(&sj).count() > 0
            }).collect::<Vec<(i32, i32)>>();

            // Find coordinates where the two pieces are adjacent
            let adj = iproduct!(-10..=10, -10..=10).filter(|(dx, dy)| {
                let si: HashSet<_> = pi.iter()
                    .flat_map(|(x, y)|
                        [(0, 1), (0, -1), (1, 0), (-1, 0)].iter()
                        .map(move |(dx, dy)| (x + dx, y + dy)))
                    .map(|(x, y)| (x + dx, y + dy))
                    .collect();
                let sj: HashSet<_> = pj.iter().cloned().collect();
                si.intersection(&sj).count() > 0
            }).collect::<Vec<(i32, i32)>>();

            let key = (i, ri, j, rj);
            let mut o = HashMap::new();
            o.insert(key, over);

            let mut a = HashMap::new();
            a.insert(key, adj);

            (o, a)
        })
        .reduce(
            || (HashMap::new(), HashMap::new()),
            |a, b| (a.0.into_iter().chain(b.0).collect(),
                    a.1.into_iter().chain(b.1).collect())
        );

    const N: usize = 1;

    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let int_sort = ctx.int_sort();

    // Here are all of our state variables!
    let xs: Vec<_> = iproduct!(0..N, 0..4).map(
        |(i, r)| ctx.named_int_const(&format!("x_{}_{}", i, r))).collect();
    let ys: Vec<_> = iproduct!(0..N, 0..4).map(
        |(i, r)| ctx.named_int_const(&format!("y_{}_{}", i, r))).collect();
    let zs: Vec<_> = iproduct!(0..N, 0..4).map(
        |(i, r)| ctx.named_int_const(&format!("z_{}_{}", i, r))).collect();
    let active: Vec<_> = iproduct!(0..N, 0..4).map(
        |(i, r)| ctx.named_bool_const(&format!("a_{}_{}", i, r))).collect();

    let mut s = Optimize::new(&ctx);

    // First, add a constraint that only one of each four rotations is active
    for i in 0..N {
        let active = active[i*4..(i+1)*4].iter().collect::<Vec<_>>();
        let coeffs = vec![0, 1, 1, 1, 1];
        let f = ctx.from_bool(false);
        let cond = f.pb_eq(&active, coeffs, 1);
        s.assert(&cond);
    }

    for i in 0..(4*N) {
        let (ni, ri) = (i / 8, i % 4);

        for j in 0..(4*N) {
            // Skip the same piece
            if i / 4 == j / 4 {
                continue;
            }

            let (nj, rj) = (j / 8, j % 4);
            let key = (i, ri, j, rj);
            let dx = xs[i].sub(&[&xs[j]]);
            let dy = ys[i].sub(&[&ys[j]]);
        }
    }

    if (s.check()) {
        let model = s.get_model();

        let xs = to_vec(&xs, &model, &|i| i.as_i64());
        let ys = to_vec(&ys, &model, &|i| i.as_i64());
        let zs = to_vec(&zs, &model, &|i| i.as_i64());
        let active = to_vec(&active, &model, &|i| i.as_bool());

        for a in active.iter() {
            println!("{:?}", a);
        }
    }
    println!("Hello, world {} ", is_overlapping.len());
}
