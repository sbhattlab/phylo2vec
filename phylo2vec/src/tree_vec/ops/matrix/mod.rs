use crate::tree_vec::ops::newick::{get_cherries_no_parents_with_bls, get_cherries_with_bls};
use crate::tree_vec::ops::vector::{build_vector, order_cherries, order_cherries_no_parents};
use crate::tree_vec::types::Ancestry;

/// Converts a Newick string to a matrix representation.
///
/// # Arguments
///
/// * `newick` - A string representing a phylogenetic tree in Newick format.
///
/// # Returns
///
/// A `Vec<Vec<f32>>` where each row contains the tree's vector representation value and associated branch lengths.
///
/// # Example
///
/// ```
/// use phylo2vec::tree_vec::ops::matrix::to_matrix;
/// let newick = "(0:0.1,1:0.2)2;";
/// let matrix = to_matrix(newick);
/// ```
///
/// # Notes
///
/// Assumes a valid Newick string. Relies on helper functions for processing.
pub fn to_matrix(newick: &str) -> Vec<Vec<f32>> {
    // Get the ancestry and branch lengths
    let (mut ancestry, bls) =
        get_cherries_with_bls(newick).expect("failed to get cherries with branch lengths");
    let indices = _get_sorted_indices(&ancestry);

    order_cherries(&mut ancestry); // Order the cherries in the ancestry matrix based on parent values
    let vector = build_vector(&ancestry); // Build the ordered  vector

    let reordered_bls: Vec<[f32; 2]> = indices
        .iter()
        .map(|&idx| bls[idx]) // Access each element of `bls` using the index from `indices`
        .collect();

    // Combine the vector with the branch lengths into a matrix
    let mut matrix: Vec<Vec<f32>> = Vec::new();

    for i in 0..vector.len() {
        let row = vec![vector[i] as f32, reordered_bls[i][0], reordered_bls[i][1]];
        matrix.push(row);
    }

    matrix
}

// Matrix construction for the "no parents" case
pub fn to_matrix_no_parents(newick: &str) -> Vec<Vec<f32>> {
    let (mut cherries, bls) = get_cherries_no_parents_with_bls(newick)
        .expect("failed to get cherries with branch lengths and no parents");
    let idxs = order_cherries_no_parents(&mut cherries);

    // Extract branch lengths from the Newick string or ensure they are included in get_cherries
    let reordered_bls: Vec<[f32; 2]> = idxs
        .iter()
        .map(|&idx| bls[idx]) // Access each element of `bls` using the index from `indices`
        .collect();

    let vector = build_vector(&cherries);

    let mut matrix: Vec<Vec<f32>> = Vec::new();

    for i in 0..vector.len() {
        let row = vec![vector[i] as f32, reordered_bls[i][0], reordered_bls[i][1]];
        matrix.push(row);
    }

    matrix
}

// Helper function that takes an ancestry array, and returns an array of indices,
//sorted by the parent values in the ancestry array.
fn _get_sorted_indices(ancestry: &Ancestry) -> Vec<usize> {
    let num_cherries = ancestry.len();

    // Create a vector of indices from 0 to num_cherries - 1
    let mut indices: Vec<usize> = (0..num_cherries).collect();

    // Sort the indices based on the parent value (ancestry[i][2])
    indices.sort_by_key(|&i| ancestry[i][2]);
    indices
}

/// Parses a matrix into its vector and branch length components.
///
/// # Arguments
///
/// * `matrix` - A phylogenetic tree represented in matrix format.
///
/// # Returns
///
/// * A Vec<usize> - the tree's vector representation and
/// * A Vec<[f32; 2]> - the vector's associated branch lengths
///
pub fn parse_matrix(matrix: &[Vec<f32>]) -> (Vec<usize>, Vec<[f32; 2]>) {
    let mut vector = Vec::new();
    let mut branch_lengths = Vec::new();

    for row in matrix.iter() {
        // Extract vector (ancestry) value and convert it to usize
        vector.push(row[0] as usize);

        // Extract branch lengths
        branch_lengths.push([row[1], row[2]]);
    }

    (vector, branch_lengths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // Test for the `to_matrix` function
    // Verifies correct matrix generation from a Newick string.
    #[rstest]
    #[case("(0:0.1,1:0.2)2;", vec![
        vec![0.0, 0.1, 0.2],
    ])]
    #[case("(((0:0.9,2:0.4)4:0.8,3:3.0)5:0.4,1:0.5)6;", vec![
        vec![0.0, 0.9, 0.4],
        vec![0.0, 0.8, 3.0],
        vec![3.0, 0.4, 0.5],
    ])]
    #[case("(0:0.7,(1:0.5,2:0.8)3:0.6)4;", vec![
        vec![0.0, 0.5, 0.8],
        vec![1.0, 0.7, 0.6],
    ])]
    #[case("((((0:0.30118185,3:0.69665915)5:0.69915295,4:0.059750594)6:0.0017238181,1:0.34410468)7:0.2021168,2:0.7084421)8;", vec![
        vec![0.0, 0.30118185, 0.69665915],
        vec![2.0, 0.69915295, 0.059750594],
        vec![0.0, 0.0017238181, 0.34410468],
        vec![4.0, 0.2021168, 0.7084421]
    ])]
    fn test_to_matrix(#[case] newick: String, #[case] expected_matrix: Vec<Vec<f32>>) {
        let matrix = to_matrix(&newick);

        // Check if the matrix matches the expected matrix
        assert_eq!(matrix, expected_matrix);
    }

    // Test for the `to_matrix_no_parents` function
    // Verifies correct matrix generation from a Newick string without parent nodes.
    #[rstest]
    #[case("(0:0.5,1:0.6);", vec![
        vec![0.0, 0.5, 0.6],
    ])]
    #[case("((0:0.1,2:0.2):0.3,(1:0.5,3:0.7):0.4);", vec![
        vec![0.0, 0.5, 0.7],
        vec![0.0, 0.1, 0.2],
        vec![1.0, 0.3, 0.4],
    ])]
    fn test_to_matrix_no_parents(
        #[case] newick_no_parents: String,
        #[case] expected_matrix: Vec<Vec<f32>>,
    ) {
        let matrix = to_matrix_no_parents(&newick_no_parents);

        // Check if the matrix matches the expected matrix
        assert_eq!(matrix, expected_matrix);
    }

    // Test for an empty Newick string in the `to_matrix` function
    // Ensures that an empty Newick string results in an empty matrix.
    #[rstest]
    #[case("".to_string(), vec![])]
    fn test_empty_newick_to_matrix(#[case] newick: String, #[case] expected_matrix: Vec<Vec<f32>>) {
        let matrix = to_matrix(&newick);

        // Empty Newick should result in an empty matrix
        assert_eq!(matrix, expected_matrix);
    }

    // Test for an empty Newick string in the `to_matrix_no_parents` function
    // Ensures that an empty Newick string results in an empty matrix when no parents are considered.
    #[rstest]
    #[case("".to_string(), vec![])]
    fn test_empty_newick_to_matrix_no_parents(
        #[case] newick_no_parents: String,
        #[case] expected_matrix: Vec<Vec<f32>>,
    ) {
        let matrix = to_matrix_no_parents(&newick_no_parents);

        // Empty Newick should result in an empty matrix
        assert_eq!(matrix, expected_matrix);
    }
}
