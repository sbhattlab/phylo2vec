pub mod avl;
pub mod newick;
pub mod vector;

use crate::tree_vec::types::Ancestry;

pub use vector::{
    build_vector, find_coords_of_first_leaf, get_ancestry, get_pairs, get_pairs_avl,
    order_cherries, order_cherries_no_parents,
};

pub use newick::{build_newick, get_cherries, get_cherries_no_parents, has_parents};

/// Recover a rooted tree (in Newick format) from a Phylo2Vec vector
pub fn to_newick(v: &Vec<usize>) -> String {
    let ancestry: Ancestry = get_ancestry(&v);
    build_newick(&ancestry)
}

/// Recover a Phylo2Vec vector from a rooted tree (in Newick format)
pub fn to_vector(newick: &str) -> Vec<usize> {
    let mut ancestry: Ancestry;

    if has_parents(&newick) {
        ancestry = get_cherries(newick);
        order_cherries(&mut ancestry);
    } else {
        ancestry = get_cherries_no_parents(newick);
        order_cherries_no_parents(&mut ancestry);
    }

    return build_vector(&ancestry);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    /// Test the conversion of vector to Newick format
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)6)8,2)9,(1,4)7)10;")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)6)7)8)9)10;")]
    #[case(vec![0, 0, 1], "((0,2)5,(1,3)4)6;")]
    fn test_to_newick(#[case] v: Vec<usize>, #[case] expected: &str) {
        let newick = to_newick(&v);
        assert_eq!(newick, expected);
    }

    /// Test the conversion of a newick string to a vector
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)6)8,2)9,(1,4)7)10;")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)6)7)8)9)10;")]
    #[case(vec![0, 0, 1], "((0,2)5,(1,3)4)6;")]
    fn test_to_vector(#[case] expected: Vec<usize>, #[case] newick: &str) {
        let vector = to_vector(&newick);
        assert_eq!(vector, expected);
    }

    /// Test the conversion of a newick string without parents to a vector
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)),2),(1,4));")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)))));")]
    #[case(vec![0, 0, 1], "((0,2),(1,3));")]
    fn test_to_vector_no_parents(#[case] expected: Vec<usize>, #[case] newick: &str) {
        let vector = to_vector(&newick);
        assert_eq!(vector, expected);
    }
}
