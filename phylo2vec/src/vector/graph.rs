/// Functions to derive common graph/tree theory objects from Phylo2Vec vectors
use std::collections::HashMap;

use bit_set::BitSet;
use ndarray::{s, Array1, Array2};

use crate::types::Pairs;
use crate::vector::convert::{to_ancestry, to_edges, to_pairs};

fn ones(k: usize) -> Vec<[f64; 2]> {
    vec![[1.0; 2]; k]
}

/// Generic function to calculate the cophenetic distance of a tree from
/// a Phylo2Vec vector with or without branch lengths.
/// Output is a pairwise distance matrix of dimensions n x n
/// where n = number of leaves.
/// Each distance corresponds to the sum of the branch lengths between two leaves
/// Inspired from the `cophenetic` function in the `ape` package: <https://github.com/emmanuelparadis/ape>
pub fn _cophenetic_distances(v: &[usize], bls: Option<&Vec<[f64; 2]>>) -> Array2<f64> {
    let k = v.len();
    let ancestry = to_ancestry(v);

    let bls = match bls {
        Some(b) => b.to_vec(),
        None => ones(k),
    };

    // Note: unrooted option was removed.
    // Originally implemented to match tr.unroot() in ete3
    // But now prefer to operate such that unrooting
    // preserves total branch lengths (compatible with ape
    // and ete3, see <https://github.com/etetoolkit/ete/pull/344>)

    // Dist shape: N_nodes x N_nodes
    let mut dist = Array2::<f64>::zeros((2 * k + 1, 2 * k + 1));
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
/// use ndarray::array;
/// use phylo2vec::vector::graph::cophenetic_distances;
///
/// let v = vec![0, 1, 2];
/// let dist = cophenetic_distances(&v);
/// assert_eq!(dist, array![[0.0, 3.0, 4.0, 4.0], [3.0, 0.0, 3.0, 3.0], [4.0, 3.0, 0.0, 2.0], [4.0, 3.0, 2.0, 0.0]]);
/// ```
pub fn cophenetic_distances(v: &[usize]) -> Array2<f64> {
    _cophenetic_distances(v, None)
}

/// Generic function to calculate a block matrix
/// from which the precision matrix can be derived using Schur's complement.
/// Works with a Phylo2Vec vector with or without branch lengths.
/// Output is a matrix of dimensions 2 * (n - 1) x 2 * (n - 1)
/// where n = number of leaves.
/// Inspired from the `inverseA` function in the `MCMCglmm` R package: <https://github.com/cran/MCMCglmm>
pub fn _pre_precision(v: &[usize], bls: Option<&Vec<[f64; 2]>>) -> Array2<f64> {
    let k = v.len();

    let bls = match bls {
        Some(b) => b.to_vec(),
        None => ones(k),
    };

    let edges = to_edges(v);

    let size = 2 * k;
    let mut out = Array2::<f64>::zeros((size, size));

    for i in 0..k - 1 {
        let (c1, p) = edges[2 * i];
        let (c2, _) = edges[2 * i + 1];
        let [bl1, bl2] = bls[i];

        let inv_bl1 = 1.0 / bl1;
        let inv_bl2 = 1.0 / bl2;

        out[[c1, c1]] += inv_bl1;
        out[[c2, c2]] += inv_bl2;
        out[[p, p]] += inv_bl1 + inv_bl2;
        out[[c1, p]] -= inv_bl1;
        out[[p, c1]] -= inv_bl1;
        out[[c2, p]] -= inv_bl2;
        out[[p, c2]] -= inv_bl2;
    }

    // Process root edge lengths
    let (c1, _) = edges[size - 2];
    let (c2, _) = edges[size - 1];
    let [bl1, bl2] = bls[k - 1];
    out[[c1, c1]] += 1.0 / bl1;
    out[[c2, c2]] += 1.0 / bl2;

    out
}

pub fn pre_precision(v: &[usize]) -> Array2<f64> {
    _pre_precision(v, None)
}

fn get_descendants_and_edges(pairs: &Pairs) -> (HashMap<usize, BitSet>, Vec<(usize, usize)>) {
    let k = pairs.len();

    // Create a mapping from internal node to leaves
    let mut desc: HashMap<usize, BitSet> = HashMap::with_capacity(2 * k + 1);
    for i in 0..=k {
        let mut bs = BitSet::new();
        bs.insert(i);
        desc.insert(i, bs);
    }

    let mut edges = Vec::with_capacity(2 * k);

    let mut parents: Vec<usize> = (0..=2 * k).collect();

    for (i, &(c1, c2)) in pairs.iter().enumerate() {
        let next_parent = k + 1 + i;

        // Push descendant via binary union
        let union_c1_c2 = desc[&parents[c1]].union(&desc[&parents[c2]]);
        desc.insert(next_parent, union_c1_c2.collect());

        // Push current edges
        edges.push((parents[c1], next_parent));
        edges.push((parents[c2], next_parent));

        // Update the parents of current children
        parents[c1] = next_parent;
        parents[c2] = next_parent;
    }

    (desc, edges)
}

/// Generic function to calculate a variance-covariance matrix
/// for a Phylo2Vec vector with or without branch lengths.
/// Output is a matrix of dimensions n x n
/// where n = number of leaves.
/// Adapted from `vcv.phylo` function in the `ape` package: <https://github.com/emmanuelparadis/ape>
/// Example:
/// ```
/// use ndarray::array;
/// use phylo2vec::vector::graph::vcv;
/// let v = vec![0, 1, 2];
/// let vcv_matrix = vcv(&v);
/// let arr = array![
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 2.0, 1.0, 1.0],
///     [0.0, 1.0, 3.0, 2.0],
///     [0.0, 1.0, 2.0, 3.0]
/// ];
/// assert_eq!(vcv_matrix, arr);
/// ```
pub fn _vcv(v: &[usize], bls: Option<&Vec<[f64; 2]>>) -> Array2<f64> {
    let k = v.len();
    let n_leaves = k + 1;
    let bls = match bls {
        Some(b) => b.to_vec(),
        None => ones(k),
    };

    let pairs = to_pairs(v);
    let (desc, edges) = get_descendants_and_edges(&pairs);

    // Distance from each node to the root
    let mut root_to_x = Array1::<f64>::zeros(2 * k + 1);
    // Output variance-covariance matrix
    let mut out = Array2::<f64>::zeros((n_leaves, n_leaves));

    for i in (0..2 * k).rev() {
        let (ci, p) = edges[i];
        let var = root_to_x[p];
        root_to_x[ci] = var + bls[i / 2][i % 2];

        if i % 2 == 1 {
            let j = i - 1;
            let (cj, _) = edges[j];

            let left = &desc[&cj];
            let right = &desc[&ci];

            for l in left.iter() {
                for r in right.iter() {
                    out[[l, r]] = var;
                    out[[r, l]] = var;
                }
            }
        }
    }

    // np.fill_diagonal(out, root_to_x)
    out.diag_mut().assign(&root_to_x.slice(s![..n_leaves]));

    out
}

pub fn vcv(v: &[usize]) -> Array2<f64> {
    _vcv(v, None)
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
        #[case] expected: Array2<f64>,
    ) {
        assert_eq!(cophenetic_distances(&v), expected);
    }

    #[rstest]
    #[case(vec![0], array![[1.0, 0.0], [0.0, 1.0]])]
    #[case(vec![0, 1, 2], array![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 1.0, 1.0],
        [0.0, 1.0, 3.0, 2.0],
        [0.0, 1.0, 2.0, 3.0]
    ])]
    #[case(vec![0, 1, 1, 4], array![
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 1.0, 3.0, 2.0],
        [0.0, 1.0, 2.0, 1.0, 1.0],
        [0.0, 3.0, 1.0, 4.0, 2.0],
        [0.0, 2.0, 1.0, 2.0, 3.0]
    ])]
    fn test_vcv(
        #[case] v: Vec<usize>,
        // #[case] unrooted: bool,
        #[case] expected: Array2<f64>,
    ) {
        assert_eq!(vcv(&v), expected);
    }

    #[rstest]
    #[case(vec![0], array![[1.0, 0.0], [0.0, 1.0]])]
    #[case(vec![0, 1, 2], array![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, -1.0, -1.0, 3.0, -1.0],
        [0.0, -1.0, 0.0, 0.0, -1.0, 3.0]])]
    #[case(vec![0, 1, 1, 4], array![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, -1.0, 0.0, 3.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 3.0, -1.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 3.0]])]
    fn test_pre_precision(
        #[case] v: Vec<usize>,
        // #[case] unrooted: bool,
        #[case] expected: Array2<f64>,
    ) {
        assert_eq!(pre_precision(&v), expected);
    }
}
