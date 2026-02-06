use crate::vector::ops::_get_node_depths;

/// Sackin index from Sackin (1972)
/// The Sackin index denotes sum of depths of all leaves
///
/// # Examples
/// ```
/// use phylo2vec::vector::balance::sackin;
/// let v = vec![0, 0, 0]; // ladder tree with 4 leaves
/// assert_eq!(sackin(&v), 9);
/// let v = vec![0, 2, 2]; // balanced tree with 4 leaves
/// assert_eq!(sackin(&v), 8);
/// ```
pub fn sackin(v: &[usize]) -> usize {
    let n_leaves = v.len() + 1;
    let depths = _get_node_depths(v, None);
    depths[..n_leaves].iter().sum::<f64>() as usize
}

pub fn leaf_depth_variance(v: &[usize]) -> f64 {
    let n_leaves = v.len() + 1;
    let depths = _get_node_depths(v, None);

    // Calculate variance of depths of leaves (nodes 0 to n_leaves-1)
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &d in &depths[..n_leaves] {
        sum += d;
        sum_sq += d * d;
    }
    let n = n_leaves as f64;
    sum_sq / n - (sum / n).powi(2)
}

/// B2 index from Shao and Sokal (1990)
/// General definition or a finite rooted phylogeny:
/// B2 = -sum_{i=1}^L p_i log_2(p_i)
/// L is the number of leaves
/// p_i is the probability of a random walk from the root to leaf i, where at each internal node, the walk chooses one of its children with equal probability.
///
/// Simplification for a binary rooted tree:
/// For a binary tree, a random walk has left/right choices at each internal node,
/// so the probability of reaching leaf i is p_i = 2^{-d_i}, where d_i is the depth of leaf i.
/// B2 = sum_{i=1}^L 2^{-d_i}
/// d_i is the depth of leaf i
/// See https://doi.org/10.1007/s00285-021-01662-7 for more details
pub fn b2(v: &[usize]) -> f64 {
    let n_leaves = v.len() + 1;
    let depths = _get_node_depths(v, None);

    let mut b2 = 0.0;
    for &d in &depths[..n_leaves] {
        b2 += d * 2f64.powi(-d as i32);
    }
    b2
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    fn close(a: f64, b: f64) -> bool {
        let rtol = 1e-5;
        let atol = 1e-8;
        (a - b).abs() <= (atol + rtol * b.abs())
    }

    /// Test Sackin index
    ///
    /// Extremal values:
    ///   Ladder tree (n leaves):   S = n(n + 1) / 2 - 1
    ///   Balanced tree (n leaves): S = n * log2(n)
    ///
    /// Source: https://doi.org/10.1007/s00026-021-00539-2
    #[rstest]
    // Small trees
    #[case(vec![0], 2)]
    #[case(vec![0, 2, 2, 3], 12)] // n=5
    // Ladder trees: S = n(n + 1) / 2 - 1
    #[case(vec![0, 0], 5)]
    #[case(vec![0, 0, 0], 9)]
    #[case(vec![0, 0, 0, 0], 14)] // n=5  -> 5*6/2 - 1 = 14
    #[case(vec![0; 49], 1274)] // n=50 -> 50*51/2 - 1 = 1274
    #[case(vec![0; 99], 5049)] // n=100 -> 100*101/2 - 1 = 5049
    // Balanced trees: S = n * log2(n)
    #[case(vec![0, 2, 2], 8)]
    #[case(vec![0, 2, 2, 6, 4, 6, 6], 24)] // n=8  -> 8*3 = 24
    #[case(vec![0, 2, 2, 6, 4, 6, 6, 14, 8, 10, 10, 14, 12, 14, 14], 64)] // n=16 -> 16*4 = 64
    fn test_sackin(#[case] v: Vec<usize>, #[case] expected: usize) {
        let result = sackin(&v);
        assert_eq!(result, expected);
    }

    /// Test leaf depth variance
    ///
    /// Extremal values:
    ///   Ladder tree (n leaves):   Var = (n - 1)(n - 2)(n^2 + 3n - 6) / (12n^2)
    ///   Balanced tree (n leaves): Var = 0 (all leaves have the same depth)
    ///
    /// Source: https://doi.org/10.1186/s12859-020-3405-1
    #[rstest]
    // Small trees
    #[case(vec![0], 0.0)] // n=2
    #[case(vec![0, 2, 2, 3], 0.24)] // n=5
    // Balanced trees: Var = 0
    #[case(vec![0, 2, 2], 0.0)] // n=4
    #[case(vec![0, 2, 2, 6, 4, 6, 6], 0.0)] // n=8
    #[case(vec![0, 2, 2, 6, 4, 6, 6, 14, 8, 10, 10, 14, 12, 14, 14], 0.0)] // n=16
    // Ladder trees: Var = (n - 1)(n - 2)(n^2 + 3n - 6) / (12n^2)
    #[case(vec![0, 0], 0.2222222)] // n=3
    #[case(vec![0, 0, 0], 0.6875)] // n=4
    #[case(vec![0, 0, 0, 0], 1.36)] // n=5
    #[case(vec![0; 49], 207.2896)] // n=50

    fn test_leaf_depth_variance(#[case] v: Vec<usize>, #[case] expected: f64) {
        let result = leaf_depth_variance(&v);
        assert!(
            close(result, expected),
            "Expected {}, got {}",
            expected,
            result
        );
    }

    /// Test B2 index
    ///
    /// Extremal values:
    ///   Ladder tree (n leaves):   B2 = 2 - 2^(2 - n)
    ///   Balanced tree (n leaves): B2 = log2(n)
    ///
    /// Source: https://doi.org/10.1007/s00285-021-01662-7
    #[rstest]
    // Small trees
    #[case(vec![0], 1.0)]
    #[case(vec![0, 2, 2, 3], 2.25)]
    // Ladder trees: B2 = 2 - 2^(2 - n)
    #[case(vec![0, 0], 1.5)] // n=3  -> 2 - 2^(-1) = 1.5
    #[case(vec![0, 0, 0], 1.75)] // n=4  -> 2 - 2^(-2) = 1.75
    #[case(vec![0, 0, 0, 0], 1.875)] // n=5  -> 2 - 2^(-3) = 1.875
    #[case(vec![0, 0, 0, 0, 0], 1.9375)] // n=6  -> 2 - 2^(-4) = 1.9375
    #[case(vec![0; 49], 2.0)] // n=50  -> ~2.0
    #[case(vec![0; 99], 2.0)] // n=100 -> ~2.0
    // Balanced trees: B2 = log2(n)
    #[case(vec![0, 2, 2], 2.0)] // n=4  -> log2(4) = 2
    #[case(vec![0, 2, 2, 6, 4, 6, 6], 3.0)] // n=8  -> log2(8) = 3
    #[case(vec![0, 2, 2, 6, 4, 6, 6, 14, 8, 10, 10, 14, 12, 14, 14], 4.0)] // n=16 -> log2(16) = 4
    fn test_b2(#[case] v: Vec<usize>, #[case] expected: f64) {
        let result = b2(&v);
        assert!(
            close(result, expected),
            "Expected {}, got {}",
            expected,
            result
        );
    }
}
