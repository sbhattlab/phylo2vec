/// Functions to derive common graph/tree theory objects from Phylo2Vec matrices
use crate::matrix::base::parse_matrix;
use crate::vector::graph::{_cophenetic_distances, _pre_precision, _vcv};

use ndarray::{Array2, ArrayView2};

/// Get the cophenetic distances from a Phylo2Vec matrix.
/// Output is a pairwise distance matrix of dimensions n x n
/// where n = number of leaves.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::matrix::graph::cophenetic_distances;
/// let m = array![
///        [0.0, 0.4, 0.5],
///        [2.0, 0.1, 0.2],
///        [2.0, 0.3, 0.6],
///    ];
/// let dist = cophenetic_distances(&m.view(), false);
/// ```
pub fn cophenetic_distances(m: &ArrayView2<f64>, unrooted: bool) -> Array2<f64> {
    let (v, bls) = parse_matrix(m);
    _cophenetic_distances(&v, Some(&bls), unrooted)
}

/// Get a precursor of the precision matrix
/// for a Phylo2Vec matrix.
/// Output is a matrix of dimensions 2 * (n - 1) x 2 * (n - 1)
/// where n = number of leaves.
/// The precision matrix can be obtained using Schur's complement
pub fn pre_precision(m: &ArrayView2<f64>) -> Array2<f64> {
    let (v, bls) = parse_matrix(m);
    _pre_precision(&v, Some(&bls))
}

/// Get the variance-covariance matrix
/// for a Phylo2Vec matrix.
/// Output is a matrix of dimensions n x n
/// where n = number of leaves.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::matrix::graph::vcv;
/// let m = array![
/// [0.0, 0.4, 0.5],
/// [2.0, 0.1, 0.2],
/// [2.0, 0.3, 0.6],
/// ];
/// let vcv_matrix = vcv(&m.view());
/// assert_eq!(vcv_matrix, array![
/// [0.4, 0.3, 0.0, 0.0],
/// [0.3, 0.5, 0.0, 0.0],
/// [0.0, 0.0, 1.0, 0.6],
/// [0.0, 0.0, 0.6, 1.1]
/// ]);
/// ```
pub fn vcv(m: &ArrayView2<f64>) -> Array2<f64> {
    let (v, bls) = parse_matrix(m);
    _vcv(&v, Some(&bls))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use rstest::rstest;

    fn allclose(a: ArrayView2<f64>, b: ArrayView2<f64>) -> bool {
        let rtol = 1e-5;
        let atol = 1e-8;
        // Check if the distances match the expected distances
        let (r, c) = a.dim();
        for i in 0..r {
            for j in 0..c {
                let diff = (a[[i, j]] - b[[i, j]]).abs();

                if diff > (atol + rtol * b[[i, j]].abs()) {
                    return false;
                }
            }
        }

        true
    }

    // Test cophenetic distances with branch lengths in the `cophenetic_distances_with_bls` function
    // Verifies correct cophenetic distance calculation from a matrix with branch lengths.
    #[rstest]
    #[case(array![[0.0, 1.0, 10.0]], false, array![[0.0, 11.0], [11.0, 0.0]])]
    #[case(array![[0.0, 1.0, 10.0]], true, array![[0.0, 11.0], [11.0, 0.0]])]
    #[case(array![
        [0.0, 1.0, 3.0],
        [0.0, 0.1, 2.0],
        [1.0, 4.0, 5.0],
    ], false, array![
        [0.0, 10.1, 2.1, 12.1],
        [10.1, 0.0, 12.0, 4.0],
        [2.1, 12.0, 0.0, 14.0],
        [12.1, 4.0, 14.0, 0.0]]
    )]
    #[case(array![
        [0.0, 1.0, 3.0],
        [0.0, 0.1, 2.0],
        [1.0, 5.0, 4.0],
    ], true, array![
        [0.0, 5.1, 2.1, 7.1],
        [5.1, 0.0, 7.0, 4.0],
        [2.1, 7.0, 0.0, 9.0],
        [7.1, 4.0, 9.0, 0.0]]
    )]
    #[case(array![
        [0.0, 0.4, 0.5],
        [2.0, 0.1, 0.2],
        [2.0, 0.3, 0.6],
    ], false, array![
        [0.0, 0.3, 1.4, 1.5],
        [0.3, 0.0, 1.5, 1.6],
        [1.4, 1.5, 0.0, 0.9],
        [1.5, 1.6, 0.9, 0.0]
    ])]
    #[case(array![
        [0.0, 0.6, 0.2],
        [0.0, 0.1, 0.4],
        [2.0, 0.2, 0.6],
        [3.0, 0.8, 0.9],
    ], false, array![
        [0.0, 1.9, 0.9, 1.8, 1.4],
        [1.9, 0.0, 2.4, 3.3, 2.9],
        [0.9, 2.4, 0.0, 1.1, 0.7],
        [1.8, 3.3, 1.1, 0.0, 0.8],
        [1.4, 2.9, 0.7, 0.8, 0.0]
    ])]
    #[case(array![
        [0.0, 1.0, 0.2],
        [2.0, 0.9, 0.7],
        [1.0, 0.1, 0.2],
        [6.0, 0.8, 0.3]
    ], false, array![
        [0.0, 2.6, 1.2, 1.8, 2.1],
        [2.6, 0.0, 2.0, 1.2, 2.9],
        [1.2, 2.0, 0.0, 1.2, 1.3],
        [1.8, 1.2, 1.2, 0.0, 2.1],
        [2.1, 2.9, 1.3, 2.1, 0.0]
    ])]
    #[case(
        array![
            [0.0, 0.9, 0.9],
            [1.0, 0.5, 0.6],
            [0.0, 0.7, 0.9],
            [4.0, 0.6, 0.7],
            [4.0, 0.1, 0.1],
            [2.0, 0.7, 0.6],
            [6.0, 0.7, 0.3],
        ], false,
        array![
            [0.0, 2.4, 2.8, 1.3, 1.5, 1.7, 3.8, 3.8],
            [2.4, 0.0, 1.8, 2.5, 2.5, 2.7, 2.8, 2.8],
            [2.8, 1.8, 0.0, 2.9, 2.9, 3.1, 2.0, 2.0],
            [1.3, 2.5, 2.9, 0.0, 1.6, 1.8, 3.9, 3.9],
            [1.5, 2.5, 2.9, 1.6, 0.0, 1.6, 3.9, 3.9],
            [1.7, 2.7, 3.1, 1.8, 1.6, 0.0, 4.1, 4.1],
            [3.8, 2.8, 2.0, 3.9, 3.9, 4.1, 0.0, 1.8],
            [3.8, 2.8, 2.0, 3.9, 3.9, 4.1, 1.8, 0.0],
        ]
    )]
    fn test_cophenetic_distances_with_bls(
        #[case] matrix: Array2<f64>,
        #[case] unrooted: bool,
        #[case] expected_distances: Array2<f64>,
    ) {
        let distances = cophenetic_distances(&matrix.view(), unrooted);
        println!("Computed distances:\n{:?}", distances);

        assert!(allclose(distances.view(), expected_distances.view()));
    }

    #[rstest]
    #[case(array![[0.0, 1.0, 10.0]], array![[1.0, 0.0], [0.0, 10.0]])]
    #[case(array![
        [0.0, 0.4, 0.5],
        [2.0, 0.1, 0.2],
        [2.0, 0.3, 0.6],
    ], array![
        [0.4, 0.3, 0.0, 0.0],
        [0.3, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.6],
        [0.0, 0.0, 0.6, 1.1]
    ])]
    #[case(array!
        [[0.0, 0.5, 0.5],
         [2.0, 1.5, 1.5],
         [2.0, 1.0, 2.0]
        ],
        array![[2.5, 1.0, 0.0, 0.0],
               [1.0, 2.5, 0.0, 0.0],
               [0.0, 0.0, 2.5, 2.0],
               [0.0, 0.0, 2.0, 2.5]]

    )]
    fn test_vcv(#[case] matrix: Array2<f64>, #[case] expected_vcv: Array2<f64>) {
        let try_vcv = vcv(&matrix.view());
        assert!(
            allclose(try_vcv.view(), expected_vcv.view()),
            "VCV mismatch: Computed VCV = {:?}, Expected VCV = {:?}",
            try_vcv,
            expected_vcv
        );
    }
}
