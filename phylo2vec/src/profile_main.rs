use std::env;

use ndarray::ArrayView2;

use phylo2vec::matrix::base::sample_matrix;
use phylo2vec::matrix::convert as mconvert;
use phylo2vec::vector::convert as vconvert;

const DEFAULT_N_LEAVES: usize = 100000;
const DEFAULT_MODE: &str = "vector";

fn profile_vector(v: &[usize]) {
    // Convert vector to newick and back
    let n_from_v = vconvert::to_newick(v);
    let v2 = vconvert::from_newick(&n_from_v);
    assert_eq!(v, v2);
}

fn profile_matrix(m: &ArrayView2<f64>) {
    // Convert matrix to newick and back
    let n_from_m = mconvert::to_newick(m);
    let m2 = mconvert::from_newick(&n_from_m);
    assert_eq!(m, m2);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Set mode (default: "vector")
    let mode = if args.len() > 1 {
        args[1].as_str()
    } else {
        DEFAULT_MODE
    };

    // Set n_leaves (default: DEFAULT_N_LEAVES)
    let n_leaves = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(DEFAULT_N_LEAVES)
    } else {
        DEFAULT_N_LEAVES
    };

    // For more stability, we use a fixed vector: 0, 2*i, 4*i, ...
    let v = (0..n_leaves).map(|i| 2 * i).collect::<Vec<usize>>();

    match mode {
        "matrix" => {
            let mut m = sample_matrix(n_leaves, false);
            // Set topology to v
            for i in 0..(n_leaves - 1) {
                m[(i, 0)] = v[i] as f64;
            }
            profile_matrix(&m.view());
        }
        "vector" => {
            profile_vector(&v);
        }
        _ => eprintln!(
            "Unrecognized mode: {}. Usage: profile_main [matrix|vector] [n_leaves]",
            mode
        ),
    }
}
