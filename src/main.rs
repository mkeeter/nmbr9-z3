#[macro_use] extern crate itertools;
use rayon::prelude::*;

extern crate z3;
use z3::*;

use std::collections::{HashSet, HashMap};
use std::fmt;

const N: usize = 4;
const R: usize = 4;

fn rotated(ts: &[(i64, i64)], rot: usize) -> Vec<(i64, i64)> {
    let m = match rot {
        0 => &[ 1,  0,  0,  1],
        1 => &[ 0,  1, -1,  0],
        2 => &[-1,  0,  0, -1],
        3 => &[ 0, -1,  1,  0],
        _ => unimplemented!("oh no")
    };
    ts.iter().map(|(x, y)| (m[0] * x + m[1] * y, m[2] * x + m[3] * y)).collect()
}

fn from_ast_vec<T>(vs: &[Ast], model: &Model, f: &Fn(Ast) -> Option<T>) -> Vec<T>
    where T: Default
{
    vs.iter().map(
        |x| model.eval(x)
            .and_then(|i| f(i))
            .unwrap_or(T::default())
        ).collect()
}

/*
 *  Builds an Ast that is true if (x,y) appears in the given table
 */
fn any_xy_match<'a>(ctx: &'a Context, x: Ast<'a>, y: Ast<'a>,
                    table: &[(i64, i64)]) -> Ast<'a> {
    let o = table
        .iter()
        .map(|(tx, ty)| (x._eq(&ctx.from_i64(*tx)),
                         y._eq(&ctx.from_i64(*ty))))
        .map(|(ex, ey)| ex.and(&[&ey]))
        .collect::<Vec<_>>();

    o[0].or(&o[1..].iter().collect::<Vec<_>>())
}

const COLORS: [&str; 10] = [
    "\x1b[7m",      // 0: bright white
    "\x1b[47m",     // 1: slightly greyer
    "\x1b[43m",     // 2: orange
    "\x1b[103m",    // 3: bright orange
    "\x1b[42m",     // 4: green
    "\x1b[44m",     // 5: blue
    "\x1b[104m",    // 6: blue-grey
    "\x1b[45m",     // 7: purple
    "\x1b[105m",    // 8: pink
    "\x1b[101m",    // 9: red
];

fn main() {
    #[allow(non_snake_case)]
    let PIECE_SHAPES: [Vec<(i64, i64)>; 10] = [
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
            PIECE_SHAPES.iter().enumerate(), 0..R,
            PIECE_SHAPES.iter().enumerate(), 0..R)
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
            }).collect::<Vec<(i64, i64)>>();

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
            }).collect::<Vec<(i64, i64)>>();

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

    let cfg = Config::new();
    let ctx = Context::new(&cfg);

    // Here are all of our state variables!
    let xs: Vec<_> = (0..N)
        .map(|i| ctx.named_int_const(&format!("x_{}", i)))
        .collect();
    let ys: Vec<_> = (0..N)
        .map(|i| ctx.named_int_const(&format!("y_{}", i)))
        .collect();
    let zs: Vec<_> = (0..N)
        .map(|i| ctx.named_int_const(&format!("z_{}", i)))
        .collect();
    let active: Vec<_> = iproduct!(0..N, 0..R)
        .map(|(i, r)| ctx.named_bool_const(&format!("a_{}_{}", i, r)))
        .collect();

    let s = Optimize::new(&ctx);

    // First, add a constraint that only one of each four rotations is active
    for i in 0..N {
        let active = active[i*R..(i+1)*R].iter().collect::<Vec<_>>();
        let cond = active[0]
            .pb_eq(&active[1..], vec![1; active.len() + 1], 1);
        s.assert(&cond);
    }

    // A piece is lonely if it is the only one at its Z level
    let lonely : Vec<_> = (0..N).into_iter().map(|i| {
        let same_z: Vec<_> = (0..N).into_iter()
            .filter(|j| *j != i)
            .map(|j| zs[i]._eq(&zs[j]))
            .collect();
        same_z[0]
            .or(&same_z[1..].iter().collect::<Vec<_>>())
            .not()})
        .collect();

    for i in 0..(R*N) {
        let (ni, ri) = (i / R / 2, i % R);

        let data = (0..R*N).into_iter()
            .filter(|j| j / R != i / R)
            .map(|j| {
                let (nj, rj) = (j / R / 2, j % R);
                let key = (ni, ri, nj, rj);

                let dx = xs[i / R].sub(&[&xs[j / R]]);
                let dy = ys[i / R].sub(&[&ys[j / R]]);

                let is_over = any_xy_match(&ctx, dx.clone(), dy.clone(),
                                           &is_overlapping[&key]);

                let is_overlapping = active[i].and(&[
                    &active[j],
                    &zs[i / R]._eq(&zs[j / R]),
                    &is_over]);

                let is_above = active[i].and(&[
                    &active[j],
                    &zs[i / R]._eq(&zs[j / R].add(&[&ctx.from_i64(1)])),
                    &is_over]);

                let is_adjacent = active[i].and(&[
                    &active[j],
                    &zs[i / R]._eq(&zs[j / R]),
                    &any_xy_match(&ctx, dx.clone(), dy.clone(),
                                  &is_adjacent[&key])]);

                (is_overlapping, is_adjacent, is_above)
            })
            .collect::<Vec<_>>();

        let any_overlapping = data[0].0
            .or(&data[1..].iter().map(|p| &p.0).collect::<Vec<_>>());

        let any_adjacent = lonely[i / R]
            .or(&data.iter().map(|p| &p.1).collect::<Vec<_>>());

        let above_two = data[0].2
            .pb_ge(&data[1..].iter().map(|p| &p.2).collect::<Vec<_>>(),
                   vec![1; data.len() + 1], 2);

        let cond = any_overlapping.not().and(&[
            &any_adjacent,
            &above_two.or(
                &[&zs[i / R]._eq(&ctx.from_i64(0))])]);

        s.assert(&active[i].not().or(&[&cond]));
    }
    println!("{}", s.to_string());

    if s.check() {
        let model = s.get_model();

        let xs = from_ast_vec(&xs, &model, &|i| i.as_i64());
        let ys = from_ast_vec(&ys, &model, &|i| i.as_i64());
        let zs = from_ast_vec(&zs, &model, &|i| i.as_i64());
        let active = from_ast_vec(&active, &model, &|i| i.as_bool());

        let mut tiles = HashMap::new();
        for i in 0..R*N {
            if active[i] {
                let j = i / R / 2;
                for (px, py) in rotated(&PIECE_SHAPES[j], i % R).iter() {
                    tiles.insert((xs[i / R] + px, ys[i / R] + py, zs[i / R]), j);
                }
            }
        }
        let xmin = tiles.keys().map(|(x, _, _)| *x).min().unwrap_or(0);
        let xmax = tiles.keys().map(|(x, _, _)| *x).max().unwrap_or(0);
        let ymin = tiles.keys().map(|(_, y, _)| *y).min().unwrap_or(0);
        let ymax = tiles.keys().map(|(_, y, _)| *y).max().unwrap_or(0);
        let zmin = tiles.keys().map(|(_, _, z)| *z).min().unwrap_or(0);
        let zmax = tiles.keys().map(|(_, _, z)| *z).max().unwrap_or(0);

        for z in zmin..=zmax {
            for y in ymin..=ymax {
                for x in xmin..=xmax {
                    let s = if x == xmin || x == xmax ||
                               y == ymin || y == ymax
                        { ". " } else { "  " };

                    if let Some(i) = tiles.get(&(x, y, z)) {
                        print!("{}{}\x1b[0m", COLORS[*i], s);
                    } else {
                        print!("{}", s);
                    }
                }
                print!("\n");
            }
        }
    } else {
        println!("unsat");
    }
}
