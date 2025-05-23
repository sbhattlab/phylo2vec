pub mod avl;
pub mod matrix;
pub mod newick;
pub mod vector;

use crate::{
    tree_vec::types::{Ancestry, Pairs},
    utils::check_m,
};
use matrix::parse_matrix;
use newick::build_newick_with_bls;

pub use vector::{
    build_vector, cophenetic_distances, find_coords_of_first_leaf, from_ancestry, from_edges,
    from_pairs, get_ancestry, get_edges, get_pairs, order_cherries, order_cherries_no_parents,
};

pub use newick::{build_newick, get_cherries, has_parents};

/// Recover a rooted tree (in Newick format) from a Phylo2Vec vector
pub fn to_newick_from_vector(v: &[usize]) -> String {
    let pairs: Pairs = get_pairs(v);
    build_newick(&pairs)
}

/// Recover a rooted tree (in Newick format) from a Phylo2Vec matrix
pub fn to_newick_from_matrix(m: &[Vec<f32>]) -> String {
    // First, check the matrix structure for validity
    check_m(m);

    let (v, bls) = parse_matrix(m);
    let pairs = get_pairs(&v);
    build_newick_with_bls(&pairs, &bls)
}

/// Recover a Phylo2Vec vector from a rooted tree (in Newick format)
pub fn to_vector(newick: &str) -> Vec<usize> {
    let mut ancestry: Ancestry = get_cherries(newick).expect("failed to get cherries");

    if has_parents(newick) {
        order_cherries(&mut ancestry);
    } else {
        order_cherries_no_parents(&mut ancestry);
    }

    build_vector(&ancestry)
}

/// Adds a new leaf to the tree
///
/// # Arguments
/// * `tree` - The tree to add the leaf to
/// * `leaf` - Index of the new leaf to add
/// * `branch` - Index of the branch to attach the leaf to
///
/// # Result
/// Modifies the tree structure by adding the new leaf and updating indices
pub fn add_leaf(v: &mut Vec<usize>, leaf: usize, branch: usize) -> Vec<usize> {
    v.push(branch);

    let mut ancestry_add = get_ancestry(v);

    let mut found_first_leaf = false;
    for row in ancestry_add.iter_mut() {
        for val in row.iter_mut() {
            if !found_first_leaf && *val == v.len() {
                // Find the indices of the first leaf
                // and then set the value to the new leaf
                *val = leaf;
                found_first_leaf = true;
            } else if *val >= leaf {
                *val += 1;
            }
        }
    }

    // ancestry_add[leaf_coords][leaf_col] = leaf as isize;
    // let ancestry_add_ref = &mut ancestry_add;
    order_cherries(&mut ancestry_add);
    order_cherries_no_parents(&mut ancestry_add);
    build_vector(&ancestry_add)
}

/// Removes a leaf from the tree
///
/// # Arguments
/// * `tree` - The tree to remove the leaf from
/// * `leaf` - Index of the leaf to remove
///
/// # Returns
/// The index of the sister node of the removed leaf
///
/// # Side effects
/// Modifies the tree structure by removing the leaf and updating indices
pub fn remove_leaf(v: &mut [usize], leaf: usize) -> (Vec<usize>, usize) {
    let ancestry = get_ancestry(v);
    let leaf_coords = find_coords_of_first_leaf(&ancestry, leaf);
    let leaf_row = leaf_coords.0;
    let leaf_col = leaf_coords.1;

    // Find the parent of the leaf to remove
    let parent = ancestry[leaf_row][2];
    let sister = ancestry[leaf_row][1 - leaf_col];
    let num_cherries = ancestry.len();

    let mut ancestry_rm = Vec::with_capacity(num_cherries - 1);

    for r in 0..num_cherries - 1 {
        let mut new_row = if r < leaf_row {
            ancestry[r]
        } else {
            ancestry[r + 1]
        };

        for node in new_row.iter_mut() {
            if *node == parent {
                *node = sister;
            }

            // Subtract 1 for leaves > "leaf"
            // (so that the vector is still valid)
            if *node > leaf {
                *node -= 1;
                if *node >= parent {
                    *node -= 1;
                }
            }
        }

        ancestry_rm.push(new_row);
    }

    order_cherries(&mut ancestry_rm);
    order_cherries_no_parents(&mut ancestry_rm);
    let new_vec = build_vector(&ancestry_rm);

    (new_vec, sister)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::sample_vector;
    use rstest::*;

    /// Test the conversion of vector to Newick format
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)6)8,2)9,(1,4)7)10;")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)6)7)8)9)10;")]
    #[case(vec![0, 0, 1], "((0,2)5,(1,3)4)6;")]
    fn test_to_newick_from_vector(#[case] v: Vec<usize>, #[case] expected: &str) {
        let newick = to_newick_from_vector(&v);
        assert_eq!(newick, expected);
    }

    /// Test the conversion of a matrix to a Newick string
    #[rstest]
    #[case(vec![
        vec![0.0, 0.9, 0.4],
        vec![0.0, 0.8, 3.12],
        vec![3.0, 0.4, 0.5],
    ], "(((0:0.9,2:0.4)4:0.8,3:3.12)5:0.4,1:0.5)6;")]
    #[case(vec![
        vec![0.0, 0.1, 0.2],
    ], "(0:0.1,1:0.2)2;")]
    #[case(vec![
        vec![0.0, 1.0, 3.0],
        vec![0.0, 0.1, 0.2],
        vec![1.0, 0.5, 0.7],
    ], "((0:0.1,2:0.2)5:0.5,(1:1,3:3)4:0.7)6;")]
    fn test_to_newick_from_matrix(#[case] m: Vec<Vec<f32>>, #[case] expected: &str) {
        let newick = to_newick_from_matrix(&m);
        assert_eq!(newick, expected);
    }

    /// Test the conversion of a Newick string to a vector
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)6)8,2)9,(1,4)7)10;")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)6)7)8)9)10;")]
    #[case(vec![0, 0, 1], "((0,2)5,(1,3)4)6;")]
    fn test_to_vector(#[case] expected: Vec<usize>, #[case] newick: &str) {
        let vector = to_vector(newick);
        assert_eq!(vector, expected);
    }

    #[rstest]
    #[case(vec![0], vec![vec![0, 2], vec![2, 0]])]
    // #[case(vec![0], true, vec![vec![0, 1], vec![1, 0]])]
    #[case(vec![0, 1, 2], vec![vec![0, 3, 4, 4], vec![3, 0, 3, 3], vec![4, 3, 0, 2], vec![4, 3, 2, 0]])]
    // #[case(vec![0, 1, 2], true, vec![vec![0, 2, 3, 3], vec![2, 0, 3, 3], vec![3, 3, 0, 2], vec![3, 3, 2, 0]])]
    #[case(vec![0, 0, 1], vec![vec![0, 4, 2, 4], vec![4, 0, 4, 2], vec![2, 4, 0, 4], vec![4, 2, 4, 0]])]
    // #[case(vec![0, 0, 1], true, vec![vec![0, 3, 2, 3], vec![3, 0, 3, 2], vec![2, 3, 0, 3], vec![3, 2, 3, 0]])]
    fn test_cophenetic_distances(
        #[case] v: Vec<usize>,
        // #[case] unrooted: bool,
        #[case] expected: Vec<Vec<usize>>,
    ) {
        assert_eq!(cophenetic_distances(&v), expected);
    }

    /// Test the conversion of a Newick string without parents to a vector
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)),2),(1,4));")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)))));")]
    #[case(vec![0, 0, 1], "((0,2),(1,3));")]
    fn test_to_vector_no_parents(#[case] expected: Vec<usize>, #[case] newick: &str) {
        let vector = to_vector(newick);
        assert_eq!(vector, expected);
    }

    /// Test the addition of a new leaf to the tree
    ///
    /// Tests are using 6 leaf tree with different leaf and branch indices
    #[rstest]
    #[case(vec![0, 1, 2, 5, 4, 2], 5, 3, vec![0, 1, 2, 5, 3, 4, 2])]
    #[case(vec![0, 1, 2, 5, 4, 2], 7, 0, vec![0, 1, 2, 5, 4, 2, 0])]
    #[case(vec![0, 1, 2, 5, 4, 2], 7, 2, vec![0, 1, 2, 5, 4, 2, 2])]
    fn test_add_leaf(
        #[case] mut v: Vec<usize>,
        #[case] leaf: usize,
        #[case] branch: usize,
        #[case] expected: Vec<usize>,
    ) {
        let new_vec = add_leaf(&mut v, leaf, branch);
        assert_eq!(new_vec, expected);
    }

    /// Test the removal of a leaf from the tree
    ///
    /// Tests are using 6 leaf tree with different leaf and sister branch indices
    #[rstest]
    #[case(vec![0, 1, 2, 5, 4, 2], 5, 4, vec![0, 1, 2, 5, 2])]
    #[case(vec![0, 1, 2, 5, 4, 2], 6, 2, vec![0, 1, 2, 5, 4])]
    #[case(vec![0, 1, 2, 5, 4, 2], 0, 11, vec![0, 1, 4, 3, 1])]
    fn test_remove_leaf(
        #[case] mut v: Vec<usize>,
        #[case] leaf: usize,
        #[case] branch: usize,
        #[case] expected: Vec<usize>,
    ) {
        let (new_vec, sister) = remove_leaf(&mut v, leaf);
        assert_eq!(new_vec, expected);
        assert_eq!(sister, branch);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_ancestry(#[case] num_leaves: usize) {
        let v = sample_vector(num_leaves, false);

        let ancestry = get_ancestry(&v);
        let v2 = from_ancestry(&ancestry);

        assert_eq!(v, v2);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_pairs(#[case] num_leaves: usize) {
        let v = sample_vector(num_leaves, false);

        let pairs = get_pairs(&v);
        let v2 = from_pairs(&pairs);

        assert_eq!(v, v2);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_edges(#[case] num_leaves: usize) {
        let v = sample_vector(num_leaves, false);

        let edges = get_edges(&v);
        let v2 = from_edges(&edges);

        assert_eq!(v, v2);
    }
}
