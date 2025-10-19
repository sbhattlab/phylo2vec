use phylo2vec::matrix::base::sample_matrix;
use phylo2vec::matrix::convert as mconvert;
use phylo2vec::vector::convert as vconvert;
use std::env;

const DEFAULT_N_LEAVES: usize = 100000;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n_leaves = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(DEFAULT_N_LEAVES)
    } else {
        DEFAULT_N_LEAVES
    };
    // For more stability, we use a fixed vector: 0, 2*i, 4*i, ...
    let v = (0..(n_leaves - 1)).map(|i| 2 * i).collect::<Vec<usize>>();
    // Convert vector to newick and back
    let n_from_v = vconvert::to_newick(&v);
    let v2 = vconvert::from_newick(&n_from_v);
    assert_eq!(v, v2);

    let mut m = sample_matrix(n_leaves, false);
    // Set topology to v
    for i in 0..(n_leaves - 1) {
        m[(i, 0)] = v[i] as f64;
    }
    // Convert matrix to newick and back
    let n_from_m = mconvert::to_newick(&m.view());
    let m2 = mconvert::from_newick(&n_from_m);
    assert_eq!(m, m2);
}
