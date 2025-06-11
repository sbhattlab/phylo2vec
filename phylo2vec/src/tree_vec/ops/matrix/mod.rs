use crate::tree_vec::ops::newick::{get_cherries_with_bls, has_parents};
use crate::tree_vec::ops::vector::{
    _cophenetic_distances, build_vector, order_cherries, order_cherries_no_parents,
};

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
    // Get the cherries and branch lengths
    let (mut cherries, mut bls) = get_cherries_with_bls(newick)
        .expect("failed to get cherries with branch lengths and no parents");

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
    let reordered_bls: Vec<[f32; 2]> = row_idxs
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

/// Get the cophenetic distances from a Phylo2Vec matrix.
/// Output is a pairwise distance matrix of dimensions n x n
/// where n = number of leaves.
///
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::matrix::cophenetic_distances_with_bls;
/// let m = vec![
///        vec![0.0, 0.4, 0.5],
///        vec![2.0, 0.1, 0.2],
///        vec![2.0, 0.3, 0.6],
///    ];
/// let dist = cophenetic_distances_with_bls(&m);
/// ```
pub fn cophenetic_distances_with_bls(m: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let (v, bls) = parse_matrix(m);
    _cophenetic_distances(&v, Some(&bls))
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

    // Test for the `to_matrix` function for Newick strings without parent nodes
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
        let matrix = to_matrix(&newick_no_parents);

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

    // Test cophenetic distances with branch lengths in the `cophenetic_distances_with_bls` function
    // Verifies correct cophenetic distance calculation from a matrix with branch lengths.
    #[rstest]
    #[case(vec![vec![0.0, 1.0, 10.0]], vec![vec![0.0, 11.0], vec![11.0, 0.0]])]
    #[case(vec![
        vec![0.0, 0.4, 0.5],
        vec![2.0, 0.1, 0.2],
        vec![2.0, 0.3, 0.6],
    ], vec![
        vec![0.0, 0.3, 1.4, 1.5],
        vec![0.3, 0.0, 1.5, 1.6],
        vec![1.4, 1.5, 0.0, 0.9],
        vec![1.5, 1.6, 0.9, 0.0]
    ])]
    #[case(vec![
        vec![0.0, 0.6, 0.2],
        vec![0.0, 0.1, 0.4],
        vec![2.0, 0.2, 0.6],
        vec![3.0, 0.8, 0.9],
    ], vec![
        vec![0.0, 1.9, 0.9, 1.8, 1.4],
        vec![1.9, 0.0, 2.4, 3.3, 2.9],
        vec![0.9, 2.4, 0.0, 1.1, 0.7],
        vec![1.8, 3.3, 1.1, 0.0, 0.8],
        vec![1.4, 2.9, 0.7, 0.8, 0.0]
    ])]
    #[case(vec![
        vec![0.0, 1.0, 0.2],
        vec![2.0, 0.9, 0.7],
        vec![1.0, 0.1, 0.2],
        vec![6.0, 0.8, 0.3]
    ], vec![
        vec![0.0, 2.6, 1.2, 1.8, 2.1],
        vec![2.6, 0.0, 2.0, 1.2, 2.9],
        vec![1.2, 2.0, 0.0, 1.2, 1.3],
        vec![1.8, 1.2, 1.2, 0.0, 2.1],
        vec![2.1, 2.9, 1.3, 2.1, 0.0]
    ])]
    #[case(
        vec![
            vec![0.0, 0.9, 0.9],
            vec![1.0, 0.5, 0.6],
            vec![0.0, 0.7, 0.9],
            vec![4.0, 0.6, 0.7],
            vec![4.0, 0.1, 0.1],
            vec![2.0, 0.7, 0.6],
            vec![6.0, 0.7, 0.3],
        ],
        vec![
            vec![0.0, 2.4, 2.8, 1.3, 1.5, 1.7, 3.8, 3.8],
            vec![2.4, 0.0, 1.8, 2.5, 2.5, 2.7, 2.8, 2.8],
            vec![2.8, 1.8, 0.0, 2.9, 2.9, 3.1, 2.0, 2.0],
            vec![1.3, 2.5, 2.9, 0.0, 1.6, 1.8, 3.9, 3.9],
            vec![1.5, 2.5, 2.9, 1.6, 0.0, 1.6, 3.9, 3.9],
            vec![1.7, 2.7, 3.1, 1.8, 1.6, 0.0, 4.1, 4.1],
            vec![3.8, 2.8, 2.0, 3.9, 3.9, 4.1, 0.0, 1.8],
            vec![3.8, 2.8, 2.0, 3.9, 3.9, 4.1, 1.8, 0.0],
        ]
    )]
    fn test_cophenetic_distances_with_bls(
        #[case] matrix: Vec<Vec<f32>>,
        //#[case] unrooted: bool,
        #[case] expected_distances: Vec<Vec<f32>>,
    ) {
        let distances = cophenetic_distances_with_bls(&matrix);

        let rtol = 1e-5;
        let atol = 1e-8;

        // Check if the distances match the expected distances
        // assert_eq!(distances, expected_distances);
        let n_leaves = distances.len();
        for i in 0..n_leaves {
            for j in 0..n_leaves {
                let diff = (distances[i][j] - expected_distances[i][j]).abs();

                assert!(diff <= (atol + rtol * expected_distances[i][j].abs()));
            }
        }
    }
}
