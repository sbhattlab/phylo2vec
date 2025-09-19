/// Base functions for `phylo2vec` matrices: sampling, validation, and checking properties.
use ndarray::{s, Array2, ArrayView2, Axis};
use rand::{distr::Uniform, prelude::Distribution};

use crate::vector::base::{check_v, sample_vector};

/// Sample a matrix with `n_leaves` elements.
///
/// If ordering is True, sample an ordered tree, by default ordering is False
///
/// ordering=True: `v_i` in {0, 1, ..., i} for i in (0, n_leaves-1)
///
/// ordering=False: `v_i` in {0, 1, ..., 2*i} for i in (0, n_leaves-1)
///
/// # Examples
///
/// ```
/// use phylo2vec::matrix::base::sample_matrix;
/// let v = sample_matrix(10, false);
/// let v2 = sample_matrix(5, true);
/// ```
pub fn sample_matrix(n_leaves: usize, ordered: bool) -> Array2<f64> {
    // Use the existing sample function to generate v
    let v = sample_vector(n_leaves, ordered);
    let bl_size = (v.len(), 2); // 2 columns for the branch lengths

    // Generate the branch lengths using the uniform distribution in (0, 1)
    let mut bls = Vec::with_capacity(bl_size.0);

    let mut rng = rand::rng();
    let uniform_dist = Uniform::new(0.0, 1.0).unwrap(); // Uniform distribution in the range (0, 1)

    for _ in 0..bl_size.0 {
        let mut row = Vec::with_capacity(bl_size.1);
        for _ in 0..bl_size.1 {
            row.push(uniform_dist.sample(&mut rng));
        }
        bls.push(row);
    }

    // Combine `v` and `bls` into a matrix
    let mut m = Array2::<f64>::zeros((v.len(), 3));

    for (i, mut row) in m.axis_iter_mut(Axis(0)).enumerate() {
        row[0] = v[i] as f64; // First column is the vector part
        row[1] = bls[i][0]; // Second column is the first branch length
        row[2] = bls[i][1]; // Third column is the second branch length
    }

    m
}

/// Input validation of a `phylo2vec` matrix
///
/// The input is checked for the `phylo2vec` constraints on the vector part (first column)
/// and positive branch lengths in the remaining columns.
///
/// # Panics
///
/// Panics if:
/// - Any element of the vector (first column of matrix) fails the `phylo2vec` constraints.
/// - Any branch length (columns 2 and 3) is not positive.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::matrix::base::check_m;
///
/// check_m(&array![
///     [0.0, 0.0, 0.0],
///     [0.0, 0.1, 0.2],
///     [1.0, 0.5, 0.7]].view()
/// );
/// ```
pub fn check_m(matrix: &ArrayView2<f64>) {
    assert!(!matrix.is_empty(), "Matrix should not be empty");

    // Validate the vector part (first column)
    let vector: Vec<usize> = matrix.outer_iter().map(|row| row[0] as usize).collect();
    check_v(&vector);

    assert!(
        matrix.shape()[1] == 3,
        "Matrix must have at least 3 columns (vector and two branch lengths)"
    );

    // Ensure all branch lengths (remaining columns) are non-negative
    for row in matrix.outer_iter() {
        assert!(
            row[1] >= 0.0 && row[2] >= 0.0,
            "Branch lengths must be positive"
        );
    }
}

/// Parses a matrix into its vector and branch length components.
///
/// # Arguments
///
/// * `matrix` - A phylogenetic tree represented in matrix format.
///
/// # Returns
///
/// * `vector` - the tree's vector representation
/// * `branch_lengths` - the vector's associated branch lengths
///
pub fn parse_matrix(matrix: &ArrayView2<f64>) -> (Vec<usize>, Vec<[f64; 2]>) {
    let vector = matrix
        .slice(s![.., 0])
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<usize>>();
    let branch_lengths = matrix
        .slice(s![.., 1..3])
        .outer_iter()
        .map(|row| [row[0], row[1]])
        .collect::<Vec<[f64; 2]>>();

    (vector, branch_lengths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    use ndarray::array;

    #[rstest]
    #[case(array![
        [0.0, 0.1, 0.4],
        [0.0, 0.2, 0.5],
        [1.0, 0.3, 0.6]
    ])]
    #[case(array![
        [0.0, 0.33, 0.72],
        [0.0, 0.61, 0.14],
        [2.0, 0.90, 0.45],
        [1.0, 0.12, 0.34],
        [8.0, 0.78, 0.23]
    ])]
    #[should_panic]
    // Empty array
    #[case(array![[]])]
    #[should_panic]
    // Wrong dimensions
    #[case(array![
        [0.0, 0.0, 9.0, 1.0],
        [0.1, 0.2, 0.3, 0.7],
        [0.4, 0.5, 0.6, 0.8]
    ])]
    #[should_panic]
    // Error due to v
    #[case(array![
        [0.0, 0.0, 0.1, 0.2],
        [0.0, 0.3, 0.4, 0.5],
        [9.0, 0.6, 0.7, 0.8],
        [1.0, 0.9, 1.0, 1.1]
    ])]
    #[should_panic]
    // Error due to negative branch lengths
    #[case(array![
        [0.0, 0.33, 0.72],
        [0.0, 0.61, 0.14],
        [2.0, 0.90, 0.45],
        [1.0, 0.12, 0.34],
        [8.0, -0.78, 0.23]
    ])]
    fn test_check_m(#[case] m: Array2<f64>) {
        check_m(&m.view());
    }

    #[rstest]
    #[case(50, true)]
    #[case(50, false)]
    fn test_sample_matrix(#[case] n_leaves: usize, #[case] ordering: bool) {
        // Assuming sample_matrix generates a matrix
        let matrix = sample_matrix(n_leaves, ordering);

        // Check the number of rows in the matrix
        assert_eq!(matrix.shape()[0], n_leaves - 1);

        // Check if the number of columns in the matrix is correct (e.g., 2)
        let num_columns = matrix.shape()[1];
        assert!(num_columns > 0, "Matrix should have at least one column.");

        // Check that the branch lengths are positive
        assert_eq!(matrix.abs(), matrix);
    }
}
