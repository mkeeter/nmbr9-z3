#[macro_use] extern crate itertools;
use rayon::prelude::*;

extern crate z3;
use z3::*;

use std::collections::{HashSet, HashMap};
use std::fmt;

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

fn build_ast_vec<'a>(f: &Fn(&str) -> Ast<'a>, name: &str, n: usize) -> Vec<Ast<'a>> {
    iproduct!(0..n, 0..4)
        .map(|(i, r)| f(&format!("{}_{}_{}", name, i, r)))
        .collect()
}

/*
 *  Builds an Ast that is true if (x,y) appears in the given table
 */
fn any_xy_match<'a>(ctx: &'a Context, x: Ast<'a>, y: Ast<'a>,
                    table: &[(i64, i64)]) -> Ast<'a> {
    ctx.from_bool(false)
        .or(&table
            .iter()
            .map(|(tx, ty)| (x._eq(&ctx.from_i64(*tx)),
                             y._eq(&ctx.from_i64(*ty))))
            .map(|(ex, ey)| ex.and(&[&ey]))
            .collect::<Vec<_>>()
            .iter()
            .collect::<Vec<_>>())
}

fn main() {
    #[allow(non_snake_case)]
    let PIECE_SHAPES: [Vec<(i64, i64)>; 10] = [
        // 0
        //vec![(0, 0)],
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

    const N: usize = 2;

    let cfg = Config::new();
    let ctx = Context::new(&cfg);

    // Here are all of our state variables!
    let xs = build_ast_vec(&|s| ctx.named_int_const(s), "x", N);
    let ys = build_ast_vec(&|s| ctx.named_int_const(s), "y", N);
    let zs = build_ast_vec(&|s| ctx.named_int_const(s), "z", N);
    let active = build_ast_vec(&|s| ctx.named_bool_const(s), "active", N);

    let s = Optimize::new(&ctx);

    // First, add a constraint that only one of each four rotations is active
    for i in 0..N {
        let active = active[i*4..(i+1)*4].iter().collect::<Vec<_>>();
        let cond = ctx.from_bool(false)
            .pb_eq(&active, vec![1; active.len() + 1], 1);
        s.assert(&cond);
    }

    // A piece is lonely if it is the only one at its Z level
    let lonely : Vec<_> = (0..(4*N)).into_iter().map(|i| {
        let same_z: Vec<_> = (0..4*N).into_iter()
            .filter(|j| j / 4 != i / 4)
            .map(|j| zs[i]._eq(&zs[j]))
            .collect();
        ctx.from_bool(false)
            .or(&same_z.iter().collect::<Vec<_>>())
            .not()})
        .collect();

    for i in 0..(4*N) {
        let (ni, ri) = (i / 8, i % 4);

        let data = (0..4*N).into_iter()
            .filter(|j| j / 4 != i / 4)
            .map(|j| {
                let (nj, rj) = (j / 8, j % 4);
                let key = (ni, ri, nj, rj);

                println!("Got key {:?}", key);

                let dx = xs[i].sub(&[&xs[j]]);
                let dy = ys[i].sub(&[&ys[j]]);

                let is_over = any_xy_match(&ctx, dx.clone(), dy.clone(),
                                           &is_overlapping[&key]);

                let is_overlapping = active[i].and(&[
                    &active[j],
                    &zs[i]._eq(&zs[j]),
                    &is_over]);

                let is_above = active[i].and(&[
                    &active[j],
                    &zs[i]._eq(&zs[j].sub(&[&ctx.from_i64(1)])),
                    &is_over]);

                let is_adjacent = active[i].and(&[
                    &active[j],
                    &zs[i]._eq(&zs[j]),
                    &any_xy_match(&ctx, dx.clone(), dy.clone(),
                                  &is_adjacent[&key])]);

                (is_overlapping, is_adjacent, is_above)
            })
            .collect::<Vec<_>>();

        let any_overlapping = ctx
            .from_bool(false)
            .or(&data.iter().map(|p| &p.0).collect::<Vec<_>>());

        let any_adjacent = ctx
            .from_bool(false)
            .or(&data.iter().map(|p| &p.1).collect::<Vec<_>>());

        let above_two = ctx.from_bool(false)
            .pb_ge(&data.iter().map(|p| &p.2).collect::<Vec<_>>(),
                   vec![1; data.len() + 1], 2);

        s.assert(&any_overlapping.not());
        s.assert(&any_adjacent.or(&[&lonely[i]]));
        //s.assert(&above_two.or(&[&zs[i]._eq(&ctx.from_i64(0))]));
    }
    println!("{}", s.to_string());

    if s.check() {
        let model = s.get_model();

        let xs = from_ast_vec(&xs, &model, &|i| i.as_i64());
        let ys = from_ast_vec(&ys, &model, &|i| i.as_i64());
        let zs = from_ast_vec(&zs, &model, &|i| i.as_i64());
        let active = from_ast_vec(&active, &model, &|i| i.as_bool());

        for a in active.iter() {
            println!("{:?}", a);
        }
        for a in xs.iter() {
            println!("{:?}", a);
        }
    } else {
        println!("unsat");
    }
    println!("{:?}", is_overlapping[&(0, 0, 0, 0)]);
    println!("Hello, world {} ", is_overlapping.len());
}
