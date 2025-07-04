/// Functions to convert Phylo2Vec matrices to other tree representations
use ndarray::{Array2, ArrayView2, Axis};

use crate::matrix::base::{check_m, parse_matrix};
use crate::newick::{has_parents, parse_with_bls};
use crate::types::Pairs;
use crate::vector::convert::{
    build_vector, order_cherries, order_cherries_no_parents, prepare_cache, to_pairs,
};

/// Recover a rooted tree (in Newick format) from a Phylo2Vec matrix
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::matrix::convert::to_newick;
/// let m = array![
///     [0.0, 0.5, 0.8],
///     [1.0, 0.7, 0.6],
/// ];
/// let newick = to_newick(&m.view());
/// assert_eq!(newick, "(0:0.7,(1:0.5,2:0.8)3:0.6)4;");
/// ```
pub fn to_newick(m: &ArrayView2<f64>) -> String {
    // First, check the matrix structure for validity
    check_m(m);

    let (v, bls) = parse_matrix(m);
    let pairs = to_pairs(&v);
    build_newick_with_bls(&pairs, &bls)
}

/// Build newick string from the ancestry matrix and branch lengths
fn build_newick_with_bls(pairs: &Pairs, branch_lengths: &[[f64; 2]]) -> String {
    let num_leaves = pairs.len() + 1;

    let mut cache = prepare_cache(pairs);

    for (i, (&(c1, c2), &[bl1, bl2])) in pairs.iter().zip(branch_lengths.iter()).enumerate() {
        let s2 = std::mem::take(&mut cache[c2]);
        let sp = (num_leaves + i).to_string();
        let sb1 = bl1.to_string();
        let sb2 = bl2.to_string();

        cache[c1].push(':');
        cache[c1].push_str(&sb1);
        cache[c1].push(',');
        cache[c1].push_str(&s2);
        cache[c1].push(':');
        cache[c1].push_str(&sb2);
        cache[c1].push(')');
        cache[c1].push_str(&sp);
    }

    format!("{};", cache[0])
}

/// Converts a Newick string to a matrix representation.
///
/// # Arguments
///
/// * `newick` - A string representing a phylogenetic tree in Newick format.
///
/// # Returns
///
/// An `Array2<f64>` where each row contains the tree's vector representation value and associated branch lengths.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::matrix::convert::from_newick;
/// let newick = "(0:0.1,1:0.2)2;";
/// let matrix = from_newick(newick);
///
/// assert_eq!(matrix, array![[0.0, 0.1, 0.2]]);
/// ```
///
/// # Notes
///
/// Assumes a valid Newick string. Relies on helper functions for processing.
pub fn from_newick(newick: &str) -> Array2<f64> {
    // Get the cherries and branch lengths
    let (mut cherries, mut bls) =
        parse_with_bls(newick).expect("failed to get cherries with branch lengths and no parents");

    // Order the cherries in the ancestry matrix
    let (row_idxs, bl_rows_to_swap) = if has_parents(newick) {
        // Case 1: Newick string with parent nodes
        order_cherries(&mut cherries)
    } else {
        // Case 2: Newick string without parent nodes
        order_cherries_no_parents(&mut cherries)
    };

    // Build the vector
    let vector = build_vector(&cherries);

    // Swap the branch lengths for the specified rows
    // See `order_cherries` and `order_cherries_no_parents` for details
    for i in bl_rows_to_swap {
        bls[i] = [bls[i][1], bls[i][0]];
    }

    // Reorder the branch lengths based on the sorted indices
    let reordered_bls: Vec<[f64; 2]> = row_idxs
        .iter()
        .map(|&idx| bls[idx]) // Access each element of `bls` using the index from `indices`
        .collect();

    let mut matrix = Array2::<f64>::zeros((vector.len(), 3));

    for (i, mut row) in matrix.axis_iter_mut(Axis(0)).enumerate() {
        row[0] = vector[i] as f64; // Ancestry value
        row[1] = reordered_bls[i][0]; // Branch length 1
        row[2] = reordered_bls[i][1]; // Branch length 2
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use rstest::rstest;

    use crate::matrix::base::sample_matrix;

    // Test for the `from_newick` function
    // Verifies correct matrix generation from a Newick string.
    #[rstest]
    #[case("(0:0.1,1:0.2)2;", array![[0.0, 0.1, 0.2]] )]
    #[case("(((0:0.9,2:0.4)4:0.8,3:3.0)5:0.4,1:0.5)6;", array![
        [0.0, 0.9, 0.4],
        [0.0, 0.8, 3.0],
        [3.0, 0.4, 0.5],
    ])]
    #[case("(0:0.7,(1:0.5,2:0.8)3:0.6)4;", array![
        [0.0, 0.5, 0.8],
        [1.0, 0.7, 0.6],
    ])]
    #[case("((((0:0.30118185,3:0.69665915)5:0.69915295,4:0.059750594)6:0.0017238181,1:0.34410468)7:0.2021168,2:0.7084421)8;", array![
        [0.0, 0.30118185, 0.69665915],
        [2.0, 0.69915295, 0.059750594],
        [0.0, 0.0017238181, 0.34410468],
        [4.0, 0.2021168, 0.7084421]
    ])]
    fn test_from_newick(#[case] newick: String, #[case] expected_matrix: Array2<f64>) {
        let matrix = from_newick(&newick);
        // Check if the matrix matches the expected matrix
        assert_eq!(matrix, expected_matrix);
    }

    // Test for the `from_newick` function for Newick strings without parent nodes
    #[rstest]
    #[case("(0:0.5,1:0.6);", array![
        [0.0, 0.5, 0.6],
    ])]
    #[case("((0:0.1,2:0.2):0.3,(1:0.5,3:0.7):0.4);", array![
        [0.0, 0.5, 0.7],
        [0.0, 0.1, 0.2],
        [1.0, 0.3, 0.4],
    ])]
    fn test_from_newick_no_parents(
        #[case] newick_no_parents: String,
        #[case] expected_matrix: Array2<f64>,
    ) {
        let matrix = from_newick(&newick_no_parents);

        // Check if the matrix matches the expected matrix
        assert_eq!(matrix, expected_matrix);
    }

    // Test for an empty Newick string in the `from_newick` function
    // Ensures that an empty Newick string results in an empty matrix.
    #[rstest]
    #[case("".to_string(), Array2::<f64>::zeros((0, 3)))]
    fn test_empty_newick_from_newick(#[case] newick: String, #[case] expected_matrix: Array2<f64>) {
        let matrix = from_newick(&newick);

        // Empty Newick should result in an empty matrix
        assert_eq!(matrix, expected_matrix);
    }

    /// Test the conversion of a matrix to a Newick string
    #[rstest]
    #[case(array![
        [0.0, 0.9, 0.4],
        [0.0, 0.8, 3.12],
        [3.0, 0.4, 0.5],
    ], "(((0:0.9,2:0.4)4:0.8,3:3.12)5:0.4,1:0.5)6;")]
    #[case(array![[0.0, 0.1, 0.2]], "(0:0.1,1:0.2)2;")]
    #[case(array![
        [0.0, 1.0, 3.0],
        [0.0, 0.1, 0.2],
        [1.0, 0.5, 0.7],
    ], "((0:0.1,2:0.2)5:0.5,(1:1,3:3)4:0.7)6;")]
    fn test_to_newick(#[case] m: Array2<f64>, #[case] expected: &str) {
        let newick = to_newick(&m.view());
        assert_eq!(newick, expected);
    }

    // Test convert to newick and back to matrix
    #[rstest]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_newick(#[case] n_leaves: usize) {
        // Sample a matrix
        let m = sample_matrix(n_leaves, false);

        // Convert to Newick
        let newick = to_newick(&m.view());

        // Convert back to matrix
        let m2 = from_newick(&newick);

        // Check if the original and converted matrices are equal
        assert_eq!(m, m2);
    }
}
