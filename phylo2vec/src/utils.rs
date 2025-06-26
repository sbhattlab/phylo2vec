use ndarray::{Array2, ArrayView2, Axis};
use rand::{distributions::Uniform, prelude::Distribution, Rng};

/// Sample a vector with `n_leaves` elements.
///
/// If ordering is True, sample an ordered tree, by default ordering is False
/// ordering=True: v_i in {0, 1, ..., i} for i in (0, n_leaves-1)
/// ordering=False: v_i in {0, 1, ..., 2*i} for i in (0, n_leaves-1)
///
/// # Examples
///
/// ```
/// use phylo2vec::utils::sample_vector;
/// let v = sample_vector(10, false);
/// let v2 = sample_vector(5, true);
/// ```
pub fn sample_vector(n_leaves: usize, ordering: bool) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::with_capacity(n_leaves);
    let mut rng = rand::thread_rng();

    match ordering {
        true => {
            for i in 0..(n_leaves - 1) {
                v.push(rng.gen_range(0..(i + 1)));
            }
        }
        false => {
            for i in 0..(n_leaves - 1) {
                v.push(rng.gen_range(0..(2 * i + 1)));
            }
        }
    }

    v
}

/// Sample a matrix with `n_leaves` elements.
///
/// If ordering is True, sample an ordered tree, by default ordering is False
/// ordering=True: v_i in {0, 1, ..., i} for i in (0, n_leaves-1)
/// ordering=False: v_i in {0, 1, ..., 2*i} for i in (0, n_leaves-1)
///
/// # Examples
///
/// ```
/// use phylo2vec::utils::sample_matrix;
/// let v = sample_matrix(10, false);
/// let v2 = sample_matrix(5, true);
/// ```
pub fn sample_matrix(n_leaves: usize, ordered: bool) -> Array2<f32> {
    // Use the existing sample function to generate v
    let v = sample_vector(n_leaves, ordered);
    let bl_size = (v.len(), 2); // 2 columns for the branch lengths

    // Generate the branch lengths using the uniform distribution in (0, 1)
    let mut bls = Vec::with_capacity(bl_size.0);

    let mut rng = rand::thread_rng();
    let uniform_dist = Uniform::new(0.0, 1.0); // Uniform distribution in the range (0, 1)

    for _ in 0..bl_size.0 {
        let mut row = Vec::with_capacity(bl_size.1);
        for _ in 0..bl_size.1 {
            row.push(uniform_dist.sample(&mut rng) as f32);
        }
        bls.push(row);
    }

    // Combine `v` and `bls` into a matrix
    let mut m = Array2::<f32>::zeros((v.len(), 3));

    for (i, mut row) in m.axis_iter_mut(Axis(0)).enumerate() {
        row[0] = v[i] as f32; // First column is the vector part
        row[1] = bls[i][0]; // Second column is the first branch length
        row[2] = bls[i][1]; // Third column is the second branch length
    }

    m
}

/// Input validation of a Phylo2Vec vector
///
/// The input is checked to satisfy the Phylo2Vec constraints
///
/// # Panics
///
/// Panics if any element of the input vector is out of bounds
///
/// # Examples
///
/// ```
/// use phylo2vec::utils::check_v;
/// check_v(&vec![0, 0, 1]);
/// ```
pub fn check_v(v: &[usize]) {
    for (i, vi) in v.iter().enumerate() {
        _check_max(i, *vi);
    }
}

/// Input validation of a Phylo2Vec matrix
///
/// The input is checked for the Phylo2Vec constraints on the vector part (first column)
/// and positive branch lengths in the remaining columns.
///
/// # Panics
///
/// Panics if:
/// - Any element of the vector (first column of matrix) fails the Phylo2Vec constraints.
/// - Any branch length (columns 2 and 3) is not positive.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::utils::{check_m};
///
/// check_m(&array![
///     [0.0, 0.0, 0.0],
///     [0.0, 0.1, 0.2],
///     [1.0, 0.5, 0.7]].view()
/// );
///
pub fn check_m(matrix: &ArrayView2<f32>) {
    // Validate the vector part (first column)
    let vector: Vec<usize> = matrix.outer_iter().map(|row| row[0] as usize).collect();
    check_v(&vector);

    // Ensure all branch lengths (remaining columns) are non-negative
    for row in matrix.outer_iter() {
        assert!(
            row[1] >= 0.0 && row[2] >= 0.0,
            "Branch lengths must be positive"
        );
    }
}

/// Validate the maximum value of a Phylo2Vec vector element
///
/// # Panics
///
/// Panics if the value is out of bounds (max = 2 * idx)
fn _check_max(idx: usize, value: usize) {
    let absolute_max = 2 * idx;
    assert!(
        value <= absolute_max,
        "Validation failed: v[{}] = {} is out of bounds (max = {})",
        idx,
        value,
        absolute_max
    );
}

/// Check if a Phylo2Vec vector is unordered
///
/// # Panics
///
/// Panics if any element of the input vector is out of bounds
///
/// # Returns
///
/// Returns true if the vector is unordered, false otherwise
///
/// # Examples
///
/// ```
/// use phylo2vec::utils::is_unordered;
///
/// let unordered = is_unordered(&vec![0, 0, 0, 3, 2, 9, 4, 1, 12]);
///
/// assert_eq!(unordered, true);
///
/// let unordered = is_unordered(&vec![0, 0, 0, 1, 3, 3, 1, 4, 4]);
///
/// assert_eq!(unordered, false);
/// ```
pub fn is_unordered(v: &[usize]) -> bool {
    for (i, vi) in v.iter().enumerate() {
        _check_max(i, *vi);
        if v[i] > i + 1 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(50, true, 1)]
    #[case(50, false, 2)]
    fn test_sample_vector(#[case] n_leaves: usize, #[case] ordering: bool, #[case] scale: usize) {
        let v = sample_vector(n_leaves, ordering);
        assert_eq!(v.len(), n_leaves - 1);
        check_v(&v);
        for (i, vi) in v.iter().enumerate() {
            assert!(*vi <= scale * i);
        }
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

    #[rstest]
    #[case(vec![0, 0, 1])]
    #[case(vec![0, 0, 2, 1, 8])]
    #[should_panic]
    #[case(vec![0, 0, 9, 1])]
    fn test_check_v(#[case] v: Vec<usize>) {
        check_v(&v);
    }

    #[rstest]
    #[case(vec![0, 0, 0, 1, 3, 3, 1, 4, 4], false)]
    #[case(vec![0, 0, 0, 3, 2, 9, 4, 1, 12], true)]
    #[should_panic]
    #[case(vec![0, 0, 1, 10, 2, 9, 4, 1, 12], true)]
    fn test_is_unordered(#[case] v: Vec<usize>, #[case] expected: bool) {
        assert_eq!(is_unordered(&v), expected);
    }
}
