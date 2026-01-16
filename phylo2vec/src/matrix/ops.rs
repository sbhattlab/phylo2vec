/// Operations on Phylo2Vec matrices
use ndarray::ArrayView2;

use crate::matrix::base::parse_matrix;
use crate::vector::ops::{_get_node_depth, _get_node_depths};

/// Get the depths of all nodes in a Phylo2Vec matrix.
///
/// The depth of a node is the length of the path from the root to that node
/// (i.e., distance from root). This follows the BEAST/ETE convention.
///
/// The root has depth 0, and depths increase as you move toward the leaves.
///
/// For matrices, actual branch lengths are used.
///
/// # Returns
/// A vector of depths for all nodes (length = 2 * n_leaves - 1).
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::matrix::ops::get_node_depths;
///
/// // Tree: ((0:0.3,2:0.7)3:0.5,1:0.2)4  (v = [0, 0])
/// let m = array![
///     [0.0, 0.3, 0.7],
///     [0.0, 0.5, 0.2],
/// ];
/// let depths = get_node_depths(&m.view());
/// // Root (node 4) has depth 0
/// assert_eq!(depths[4], 0.0);
/// // Node 3 is 0.5 from root
/// assert!((depths[3] - 0.5).abs() < 1e-10);
/// // Leaf 0 is 0.5 + 0.3 = 0.8 from root
/// assert!((depths[0] - 0.8).abs() < 1e-10);
/// // Leaf 1 is 0.2 from root
/// assert!((depths[1] - 0.2).abs() < 1e-10);
/// // Leaf 2 is 0.5 + 0.7 = 1.2 from root
/// assert!((depths[2] - 1.2).abs() < 1e-10);
/// ```
pub fn get_node_depths(m: &ArrayView2<f64>) -> Vec<f64> {
    let (v, bls) = parse_matrix(m);
    _get_node_depths(&v, Some(&bls))
}

/// Get the depth of a node in a Phylo2Vec matrix.
///
/// The depth of a node is the length of the path from the root to that node
/// (i.e., distance from root). This follows the BEAST/ETE convention.
///
/// The root has depth 0, and depths increase as you move toward the leaves.
///
/// For matrices, actual branch lengths are used.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::matrix::ops::get_node_depth;
///
/// // Tree: ((0:0.3,2:0.7)3:0.5,1:0.2)4  (v = [0, 0])
/// let m = array![
///     [0.0, 0.3, 0.7],
///     [0.0, 0.5, 0.2],
/// ];
/// // Root (node 4) has depth 0
/// assert_eq!(get_node_depth(&m.view(), 4), 0.0);
/// // Node 3 is 0.5 from root
/// assert!((get_node_depth(&m.view(), 3) - 0.5).abs() < 1e-10);
/// // Leaf 0 is 0.5 + 0.3 = 0.8 from root
/// assert!((get_node_depth(&m.view(), 0) - 0.8).abs() < 1e-10);
/// ```
pub fn get_node_depth(m: &ArrayView2<f64>, node: usize) -> f64 {
    let (v, bls) = parse_matrix(m);
    _get_node_depth(&v, Some(&bls), node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::base::sample_matrix;
    use crate::vector::ops::get_node_depth as get_node_depth_vector;
    use ndarray::array;
    use rstest::rstest;

    #[rstest]
    // Tree: (0:0.7,(1:0.5,2:0.8)3:0.6)4;
    // Depth is distance from root to node
    // Root (node 4) has depth 0
    #[case(array![
        [0.0, 0.5, 0.8],
        [1.0, 0.7, 0.6],
    ], 4, 0.0)]
    // Node 3 is 0.6 from root
    #[case(array![
        [0.0, 0.5, 0.8],
        [1.0, 0.7, 0.6],
    ], 3, 0.6)]
    // Leaf 0 is 0.7 from root
    #[case(array![
        [0.0, 0.5, 0.8],
        [1.0, 0.7, 0.6],
    ], 0, 0.7)]
    // Leaf 1 is 0.6 + 0.5 = 1.1 from root
    #[case(array![
        [0.0, 0.5, 0.8],
        [1.0, 0.7, 0.6],
    ], 1, 1.1)]
    // Leaf 2 is 0.6 + 0.8 = 1.4 from root
    #[case(array![
        [0.0, 0.5, 0.8],
        [1.0, 0.7, 0.6],
    ], 2, 1.4)]
    // Tree: (((0:0.9,2:0.4)4:0.8,3:3.0)5:0.4,1:0.5)6;
    // Root (node 6) has depth 0
    #[case(array![
        [0.0, 0.9, 0.4],
        [0.0, 0.8, 3.0],
        [3.0, 0.4, 0.5],
    ], 6, 0.0)]
    // Node 5 is 0.4 from root
    #[case(array![
        [0.0, 0.9, 0.4],
        [0.0, 0.8, 3.0],
        [3.0, 0.4, 0.5],
    ], 5, 0.4)]
    // Node 4 is 0.4 + 0.8 = 1.2 from root
    #[case(array![
        [0.0, 0.9, 0.4],
        [0.0, 0.8, 3.0],
        [3.0, 0.4, 0.5],
    ], 4, 1.2)]
    // Leaf 1 is 0.5 from root
    #[case(array![
        [0.0, 0.9, 0.4],
        [0.0, 0.8, 3.0],
        [3.0, 0.4, 0.5],
    ], 1, 0.5)]
    // Leaf 3 is 0.4 + 3.0 = 3.4 from root
    #[case(array![
        [0.0, 0.9, 0.4],
        [0.0, 0.8, 3.0],
        [3.0, 0.4, 0.5],
    ], 3, 3.4)]
    // Leaf 0 is 0.4 + 0.8 + 0.9 = 2.1 from root
    #[case(array![
        [0.0, 0.9, 0.4],
        [0.0, 0.8, 3.0],
        [3.0, 0.4, 0.5],
    ], 0, 2.1)]
    // Leaf 2 is 0.4 + 0.8 + 0.4 = 1.6 from root
    #[case(array![
        [0.0, 0.9, 0.4],
        [0.0, 0.8, 3.0],
        [3.0, 0.4, 0.5],
    ], 2, 1.6)]
    fn test_get_node_depth(
        #[case] m: ndarray::Array2<f64>,
        #[case] node: usize,
        #[case] expected_depth: f64,
    ) {
        let depth = get_node_depth(&m.view(), node);
        assert!(
            (depth - expected_depth).abs() < 1e-10,
            "Expected depth {expected_depth}, got {depth}"
        );
    }

    #[rstest]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_get_node_depth_root_is_zero(#[case] n_leaves: usize) {
        let m = sample_matrix(n_leaves, false);
        let root = 2 * m.nrows(); // Root node index
        let depth = get_node_depth(&m.view(), root);
        assert_eq!(depth, 0.0, "Root should have depth 0");
    }

    #[rstest]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    fn test_get_node_depths_root_is_zero(#[case] n_leaves: usize) {
        let m = sample_matrix(n_leaves, false);
        let root = 2 * m.nrows();
        let depths = get_node_depths(&m.view());
        assert_eq!(depths[root], 0.0, "Root should have depth 0");
    }

    #[rstest]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    fn test_get_node_depth_topological_matches_vector(#[case] n_leaves: usize) {
        let m = sample_matrix(n_leaves, false);
        let v: Vec<usize> = m.column(0).iter().map(|&x| x as usize).collect();

        // Create a matrix with all branch lengths = 1.0 to test topological depth
        let mut m_topo = m.clone();
        for mut row in m_topo.rows_mut() {
            row[1] = 1.0;
            row[2] = 1.0;
        }

        let max_node = 2 * n_leaves - 1;

        // Test that matrix depth with unit branch lengths matches vector depth
        for node in 0..max_node {
            let depth_matrix = get_node_depth(&m_topo.view(), node);
            let depth_vector = get_node_depth_vector(&v, node);
            assert_eq!(depth_matrix, depth_vector, "Depth mismatch for node {node}");
        }
    }
}
