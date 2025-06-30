use phylo2vec::vector::convert::{from_newick, to_newick};
use std::env;

const DEFAULT_N_LEAVES: usize = 100000;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n_leaves = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(DEFAULT_N_LEAVES)
    } else {
        DEFAULT_N_LEAVES
    };
    // For more stability, we use a fixed vector
    let v = (0..(n_leaves - 1)).map(|i| 2 * i).collect::<Vec<usize>>();
    let n = to_newick(&v);
    let re_v = from_newick(&n);
    println!("vector length: {:?}", re_v.len());
}
