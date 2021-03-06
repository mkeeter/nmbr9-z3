use z3::*;

use std::collections::{HashSet, HashMap};

/*  A PieceIndex represents a particular piece, in the range 0-9 */
#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
struct Piece(usize);

/*  A Rotation represents one of four possible rotations */
#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct Rotation(usize);

/*  A PieceIndex is a combination of piece + rotation */
#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct PieceIndex(Piece, Rotation);

use rand::prelude::*;
use rayon::prelude::*;

struct Tables {
    /*  Every piece's shape */
    shapes: HashMap<PieceIndex, Vec<(i64, i64)>>,

    /*  A simple look-up table that stores area per piece */
    area: HashMap<PieceIndex, usize>,

    /*  overlapping[(a,b)] returns a list of lists of all possible offsets
     *  (as xy pairs) that produce a particular number of overlapping tiles */
    overlap: HashMap<(PieceIndex, PieceIndex),
                     Vec<(usize, Vec<(i64, i64)>)>>,

    /*  adjacent[(a,b)] returns a list of all possible offsets (as xy pairs)
     *  that result in the two pieces being adjacent (and not overlapping) */
    adjacent: HashMap<(PieceIndex, PieceIndex), Vec<(i64, i64)>>,
}

impl Tables {
    fn new() -> Tables {
        let shapes: [Vec<(i64, i64)>; 10] = [
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

        let mut area = HashMap::new();
        let mut piece_shapes = HashMap::new();
        for i in 0..shapes.len() {
            for r in 0..4 {
                let index = PieceIndex(Piece(i), Rotation(r));
                area.insert(index, shapes[i].len());
                piece_shapes.insert(index, Self::rotated(&shapes[i], r));
            }
        }

        // Build a map of where pieces overlap, and by how much
        let mut overlapping_map = HashMap::new();
        for (i, pi) in piece_shapes.iter() {
            let si: HashSet<_> = pi.iter()
                .cloned()
                .collect();
            for (j, pj) in piece_shapes.iter() {
                let target = overlapping_map.entry((*i, *j))
                    .or_insert(HashMap::new());
                for dx in -4..=4 {
                    for dy in -4..=4 {
                        let num = pj.iter()
                            .map(|(x, y)| (x + dx, y + dy))
                            .collect::<HashSet<_>>()
                            .intersection(&si)
                            .count();
                        if num > 0 {
                            target.entry(num)
                                .or_insert(Vec::new())
                                .push((dx, dy));
                        }
                    }
                }
            }
        }

        // Convert to the final form of overlap map
        let mut overlapping = HashMap::new();
        for (k, v) in overlapping_map {
            let scores: Vec<(usize, Vec<(i64, i64)>)> = v.iter()
                .map(|(k, v)| (*k, v.iter().cloned().collect()))
                .collect();
            overlapping.insert(k, scores);
        }

        // Build a map of where pieces are adjacent
        let mut adjacent = HashMap::new();
        for (i, pi) in piece_shapes.iter() {
            let si: HashSet<_> = pi.iter()
                .flat_map(|(x, y)| [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    .iter()
                    .map(move |(dx, dy)| (x + dx, y + dy)))
                .collect();

            for (j, pj) in piece_shapes.iter() {
                let target = adjacent.entry((*i, *j))
                    .or_insert(Vec::new());

                let overlap: HashSet<_> = overlapping[&(*i, *j)].iter()
                    .flat_map(|(_, v)| v.iter())
                    .cloned()
                    .collect();

                for dx in -5..=5 {
                    for dy in -5..=5 {
                        let any_adjacent = pj.iter()
                            .map(|(x, y)| (x + dx, y + dy))
                            .collect::<HashSet<_>>()
                            .intersection(&si)
                            .count() > 0;
                        if any_adjacent && !overlap.contains(&(dx, dy)) {
                            target.push((dx, dy))
                        }
                    }
                }
            }
        }

        Tables {
            shapes:     piece_shapes,
            area:       area,
            overlap:    overlapping,
            adjacent:   adjacent
        }
    }

    fn rotated(ts: &[(i64, i64)], rot: usize) -> Vec<(i64, i64)> {
        let m = match rot {
            0 => &[ 1,  0,  0,  1],
            1 => &[ 0,  1, -1,  0],
            2 => &[-1,  0,  0, -1],
            3 => &[ 0, -1,  1,  0],
            _ => unimplemented!("oh no")
        };
        let out: Vec<_> = ts.iter()
            .map(|(x, y)| (m[0] * x + m[1] * y, m[2] * x + m[3] * y))
            .collect();
        let xmin = out.iter().map(|(x, _)| *x).min().unwrap_or(0);
        let ymin = out.iter().map(|(_, y)| *y).min().unwrap_or(0);
        out.into_iter()
            .map(|(x, y)| (x - xmin, y - ymin))
            .collect()
    }
}

// The front of the vec is the top of the stack
#[derive(Clone, Debug)]
struct Stackup(Vec<Vec<Piece>>);

impl Stackup {
    fn from_ternary(mut i: usize) -> Stackup {
        let mut layer0 = Vec::new();
        let mut j = 0;
        while i > 0 {
            for k in 0..(i % 3) {
                layer0.push(Piece(j));
            }
            j += 1;
            i /= 3;
        }
        Stackup(vec![layer0])
    }

    fn as_u128(&self) -> u128 {
        let mut out: u128 = 0;
        for (z, layer) in self.0.iter().enumerate() {
            for p in layer.iter() {
                if out & ((0xF << (p.0 * 8)) as u128) == 0 {
                    out |= ((z + 1) << (p.0 * 8)) as u128;
                } else {
                    out |= ((z + 1) << (p.0 * 8 + 4)) as u128;
                }
            }
        }
        out
    }

    fn xy_bits(&self) -> u32 {
        let most_pieces = self.0.iter().map(|v| v.len()).max().unwrap_or(0);
        ((most_pieces - 1) as f64 * 4.0 + 1.0).log2().ceil() as u32
    }

    fn model_size(&self) -> u32 {
        let bits = self.xy_bits();
        let pieces: u32 = self.0.iter().map(|v| v.len() as u32).sum();
        (bits * 2 + 2) * pieces
    }

    // Stacks this Stackup onto another one, returning None if check() fails
    fn onto(&self, below: &Stackup, t: &Tables) -> Option<Stackup> {
        assert!(self.0.len() > 0);
        assert!(below.0.len() > 0);
        assert!(self.0[0].len() > 0);
        assert!(below.0[0].len() > 0);

        // Only the top layer is allowed to have less than two pieces
        if below.0[0].len() < 2 {
            return None;
        }

        // Area must be monotonically decreasing
        let area_above: usize = self.0.last()
            .unwrap()
            .iter()
            .map(|p| t.area[&PieceIndex(*p, Rotation(0))])
            .sum();
        let area_below: usize = below.0[0]
            .iter()
            .map(|p| t.area[&PieceIndex(*p, Rotation(0))])
            .sum();
        if area_below < area_above {
            return None;
        }

        // We can't have more than two of any particular piece
        let mut count = [0; 10];
        for layer in self.0.iter().chain(below.0.iter()) {
            for v in layer.iter() {
                if count[v.0] == 2 {
                    return None;
                } else {
                    count[v.0] += 1;
                }
            }
        }

        // Otherwise, we're good - build the new Stackup and return it
        let mut out = self.clone();
        for layer in below.0.iter() {
            out.0.push(layer.clone());
        }
        Some(out)
    }

    // Expensive validity check, using Z3
    fn validate(&self, t: &Tables) -> bool {
        println!("Checking {:?}", self);
        let cfg = Config::new();
        let ctx = Context::new(&cfg);

        let most_pieces = self.0.iter().map(|v| v.len()).max().unwrap_or(0);
        let bits = self.xy_bits();
        let bv_sort = ctx.bitvector_sort(bits);
        let rot_sort = ctx.bitvector_sort(2);

        let pts: Vec<_> = self.0.iter().enumerate()
            .map(|(i, t)| t.iter()
                 .enumerate()
                 .map(|(j, pt)|
                      (*pt,
                       ctx.named_bitvector_const(&format!("x_{}_{}", i, j), bits),
                       ctx.named_bitvector_const(&format!("y_{}_{}", i, j), bits),
                       ctx.named_bitvector_const(&format!("r_{}_{}", i, j), 2)))
                 .collect::<Vec<_>>())
            .collect();

        let solver = Optimize::new(&ctx);
        for layer in pts.iter() {
            Self::add_layer_constraints(&solver, &bv_sort, &rot_sort, layer, t);
        }
        for (above, below) in pts.iter().zip(pts.iter().skip(1)) {
            Self::add_interlayer_constraints(&ctx, &solver, &bv_sort, &rot_sort,
                                             above, below, t);
        }

        // Constrain one piece, to reduce the search space
        solver.assert(&pts[0][0].1._eq(&bv_sort.from_i64(0)));
        solver.assert(&pts[0][0].2._eq(&bv_sort.from_i64(0)));
        solver.assert(&pts[0][0].3._eq(&rot_sort.from_i64(0)));

        if solver.check() {
            println!("sat");
            Self::print_model(&solver.get_model(), &pts, t, bits);
            return true;
        } else {
            println!("unsat");
            return false;
        }
    }

    fn print_model(model: &Model, layers: &[Vec<(Piece, Ast, Ast, Ast)>],
                   t: &Tables, bits: u32)
    {
        let mut tiles = HashMap::new();

        for (z, layer) in layers.iter().enumerate() {
            for (p, x, y, r) in layer.iter() {
                let mut dx  = model.eval(x).and_then(|i| i.as_i64()).unwrap_or(0);
                if dx & (1 << (bits - 1)) != 0 {
                    dx -= 1 << bits;
                }
                let mut dy  = model.eval(y).and_then(|i| i.as_i64()).unwrap_or(0);
                if dy & (1 << (bits - 1)) != 0 {
                    dy -= 1 << bits;
                }
                let rot = model.eval(r).and_then(|i| i.as_i64()).unwrap_or(0);
                let index = PieceIndex(*p, Rotation(rot as usize));
                for (px, py) in t.shapes[&index].iter() {
                    tiles.insert((dx + px, dy + py, z), p.0);
                }
            }
        }

        let xmin = tiles.keys().map(|(x, _, _)| *x).min().unwrap_or(0);
        let xmax = tiles.keys().map(|(x, _, _)| *x).max().unwrap_or(0);
        let ymin = tiles.keys().map(|(_, y, _)| *y).min().unwrap_or(0);
        let ymax = tiles.keys().map(|(_, y, _)| *y).max().unwrap_or(0);

        for (z, _) in layers.iter().enumerate() {
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
            print!("\n");
        }
    }

    fn add_layer_constraints(solver: &Optimize, bv_sort: &Sort, rot_sort: &Sort,
                             pts: &[(Piece, Ast, Ast, Ast)], t: &Tables) {
        // Single-piece layers have no constraints
        if pts.len() < 2 {
            return;
        }

        for (i, (ap, ax, ay, ar)) in pts.iter().enumerate() {
            let mut adjacent = Vec::new();
            for (bp, bx, by, br) in pts.iter().skip(i + 1) {
                let dx = bx.bvsub(&ax);
                let dy = by.bvsub(&ay);

                for rot_a in 0..4 {
                    for rot_b in 0..4 {
                        let rot_matched = ar._eq(&rot_sort.from_i64(rot_a))
                            .and(&[&br._eq(&rot_sort.from_i64(rot_b))]);

                        let key = (PieceIndex(*ap, Rotation(rot_a as usize)),
                                   PieceIndex(*bp, Rotation(rot_b as usize)));
                        let mut overlap = Vec::new();
                        // We can directly assert that these pieces aren't allowed
                        // to overlap each other.
                        for (tx, ty) in t.overlap[&key]
                            .iter()
                            .flat_map(|o| o.1.iter())
                        {
                            overlap.push(dx._eq(&bv_sort.from_i64(*tx))
                                           .and(&[&dy._eq(&bv_sort.from_i64(*ty))]));
                        }
                        // Build a clause saying that we have an overlap (and
                        // that the rotation is valid), then assert not that.
                        solver.assert(&overlap[0]
                            .or(&overlap[1..].iter().collect::<Vec<_>>())
                            .and(&[&rot_matched])
                            .not());


                        // We build a list of possible adjacencies, then `or` them
                        // together as an assertion at the end.
                        for (tx, ty) in t.adjacent[&key].iter() {
                            adjacent.push(dx._eq(&bv_sort.from_i64(*tx))
                                            .and(&[&dy._eq(&bv_sort.from_i64(*ty)),
                                                   &rot_matched]));
                        }
                    }
                }
            }
            if adjacent.len() > 0 {
                solver.assert(&adjacent[0]
                              .or(&adjacent[1..].iter().collect::<Vec<_>>()));
            }
        }

    }

    fn add_interlayer_constraints(ctx: &Context, solver: &Optimize,
                                  bv_sort: &Sort, rot_sort: &Sort,
                                  above: &[(Piece, Ast, Ast, Ast)],
                                  below: &[(Piece, Ast, Ast, Ast)],
                                  t: &Tables)
    {
        let mut count = Vec::new();
        let mut total_area = 0;
        let zero = ctx.from_i64(0);
        for (ap, ax, ay, ar) in above.iter() {
            let area = t.area[&PieceIndex(*ap, Rotation(0))];
            total_area += area;
            for (bp, bx, by, br) in below.iter() {
                let dx = bx.bvsub(&ax);
                let dy = by.bvsub(&ay);

                for rot_a in 0..4 {
                    for rot_b in 0..4 {
                        let rot_matched = ar._eq(&rot_sort.from_i64(rot_a))
                            .and(&[&br._eq(&rot_sort.from_i64(rot_b))]);

                        let key = (PieceIndex(*ap, Rotation(rot_a as usize)),
                                   PieceIndex(*bp, Rotation(rot_b as usize)));

                        for (score, ts) in t.overlap[&key].iter() {
                            // Skip the fully-overlapping areas, because
                            // they violate the above-two constraint.
                            if *score == area {
                                continue;
                            }
                            let score = ctx.from_i64(*score as i64);
                            for (tx, ty) in ts.iter() {
                                count.push(
                                    dx._eq(&bv_sort.from_i64(*tx))
                                       .and(&[&dy._eq(&bv_sort.from_i64(*ty)),
                                              &rot_matched])
                                       .ite(&score, &zero));
                            }
                        }
                    }
                }
            }
        }

        solver.assert(&count[0].add(&count[1..].iter().collect::<Vec<_>>())
                               ._eq(&ctx.from_i64(total_area as i64)));
    }
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
    println!("Building tables!");
    let t = Tables::new();
    println!("Done");

    let mut base = (1..(3 as usize).pow(5))
        .map(|t| Stackup::from_ternary(t))
        .collect::<Vec<Stackup>>();

    let mut targets = Vec::new();
    for (i, above) in base.iter().enumerate() {
        println!("{}: {}", i, targets.len());
        let mut v = base.par_iter()
            .filter_map(|b| above.onto(b, &t))
            .collect::<Vec<_>>();
        targets.append(&mut v);
    }
    targets.sort_by_key(|s| s.model_size());
    let targets = targets.iter()
        .filter(|b| b.validate(&t))
        .collect::<Vec<_>>();

    println!("Got {} targets", targets.len());
    for t in targets {
    }

    let seed = [
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
    ];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    for i in 0..10 {
        let mut s = Stackup(Vec::new());
        s.0.push(Vec::new());
        s.0[0].push(Piece(rng.gen::<usize>() % 10));

        s.0.push(Vec::new());
        s.0[1].push(Piece(rng.gen::<usize>() % 10));
        s.0[1].push(Piece(rng.gen::<usize>() % 10));

        s.validate(&t);
    }
}
