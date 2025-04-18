use phylo2vec::{
    tree_vec::ops::{to_newick_from_vector, to_vector},
    utils::sample_vector,
};
use std::env;

const DEFAULT_N_LEAVES: usize = 100000;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n_leaves = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(DEFAULT_N_LEAVES)
    } else {
        DEFAULT_N_LEAVES
    };
    let v = sample_vector(n_leaves, true);
    let n = to_newick_from_vector(&v);
    let re_v = to_vector(&n);
    println!("vector length: {:?}", re_v.len());
}
