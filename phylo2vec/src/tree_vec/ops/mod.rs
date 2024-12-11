pub mod avl;
pub mod newick;
pub mod vector;

use crate::tree_vec::types::Ancestry;

pub use vector::{
    build_vector, find_coords_of_first_leaf, get_ancestry, get_pairs, get_pairs_avl,
    order_cherries, order_cherries_no_parents,
};

pub use newick::build_newick;

/// Recover a rooted tree (in Newick format) from a Phylo2Vec vector
pub fn to_newick(v: &Vec<usize>) -> String {
    let ancestry: Ancestry = get_ancestry(&v);
    build_newick(&ancestry)
}
