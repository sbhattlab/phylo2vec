/// Functions to derive common graph/tree theory objects from Phylo2Vec vectors
use ndarray::{s, Array2};

use crate::vector::convert::to_ancestry;

/// Generic function to calculate the cophenetic distance of a tree from
/// a Phylo2Vec vector with or without branch lengths.
/// Output is a pairwise distance matrix of dimensions n x n
/// where n = number of leaves.
/// Each distance corresponds to the sum of the branch lengths between two leaves
/// Inspired from the `cophenetic` function in the `ape` package: https://github.com/emmanuelparadis/ape
pub fn _cophenetic_distances(v: &[usize], bls: Option<&Vec<[f32; 2]>>) -> Array2<f32> {
    let k = v.len();
    let ancestry = to_ancestry(v);

    let bls = match bls {
        Some(b) => b.to_vec(),
        None => vec![[1.0; 2]; v.len()],
    };

    // Note: unrooted option was removed.
    // Originally implemented to match tr.unroot() in ete3
    // But now prefer to operate such that unrooting
    // preserves total branch lengths (compatible with ape
    // and ete3, see https://github.com/etetoolkit/ete/pull/344)

    // Dist shape: N_nodes x N_nodes
    let mut dist = Array2::<f32>::zeros((2 * k + 1, 2 * k + 1));
    let mut all_visited: Vec<usize> = Vec::new();

    for i in 0..k {
        let [c1, c2, p] = ancestry[k - i - 1];
        let [bl1, bl2] = bls[k - i - 1];

        if !all_visited.is_empty() {
            for &visited in &all_visited[0..all_visited.len() - 1] {
                let dist1 = dist[[p, visited]] + bl1;
                let dist2 = dist[[p, visited]] + bl2;

                dist[[c1, visited]] = dist1;
                dist[[visited, c1]] = dist1;
                dist[[c2, visited]] = dist2;
                dist[[visited, c2]] = dist2;
            }
        }

        dist[[c1, c2]] = bl1 + bl2;
        dist[[c2, c1]] = bl1 + bl2;

        dist[[c1, p]] = bl1;
        dist[[p, c1]] = bl1;

        dist[[c2, p]] = bl2;
        dist[[p, c2]] = bl2;

        all_visited.push(c1);
        all_visited.push(c2);
        all_visited.push(p);
    }

    let n_leaves = k + 1;

    // Return the upper-left n_leaves x n_leaves submatrix
    dist.slice(s![0..n_leaves, 0..n_leaves]).to_owned()
}

/// Get the cophenetic distances from the Phylo2Vec vector
/// Output is a pairwise distance matrix of dimensions n x n
/// where n = number of leaves.
///
/// # Example
/// ```
/// use phylo2vec::vector::graph::cophenetic_distances;
///
/// let v = vec![0, 0, 0, 1, 3, 3, 1, 4, 4];
/// let dist = cophenetic_distances(&v);
/// ```
pub fn cophenetic_distances(v: &[usize]) -> Array2<f32> {
    _cophenetic_distances(v, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rstest::rstest;

    #[rstest]
    #[case(vec![0], array![[0.0, 2.0], [2.0, 0.0]])]
    #[case(vec![0, 1, 2], array![[0.0, 3.0, 4.0, 4.0], [3.0, 0.0, 3.0, 3.0], [4.0, 3.0, 0.0, 2.0], [4.0, 3.0, 2.0, 0.0]])]
    #[case(vec![0, 0, 1], array![[0.0, 4.0, 2.0, 4.0], [4.0, 0.0, 4.0, 2.0], [2.0, 4.0, 0.0, 4.0], [4.0, 2.0, 4.0, 0.0]])]
    #[case(vec![0, 1, 0, 4, 4, 2, 6], array![
        [0.0, 5.0, 6.0, 2.0, 4.0, 4.0, 7.0, 7.0],
        [5.0, 0.0, 3.0, 5.0, 5.0, 5.0, 4.0, 4.0],
        [6.0, 3.0, 0.0, 6.0, 6.0, 6.0, 3.0, 3.0],
        [2.0, 5.0, 6.0, 0.0, 4.0, 4.0, 7.0, 7.0],
        [4.0, 5.0, 6.0, 4.0, 0.0, 2.0, 7.0, 7.0],
        [4.0, 5.0, 6.0, 4.0, 2.0, 0.0, 7.0, 7.0],
        [7.0, 4.0, 3.0, 7.0, 7.0, 7.0, 0.0, 2.0],
        [7.0, 4.0, 3.0, 7.0, 7.0, 7.0, 2.0, 0.0]
    ])]
    fn test_cophenetic_distances(
        #[case] v: Vec<usize>,
        // #[case] unrooted: bool,
        #[case] expected: Array2<f32>,
    ) {
        assert_eq!(cophenetic_distances(&v), expected);
    }
}
