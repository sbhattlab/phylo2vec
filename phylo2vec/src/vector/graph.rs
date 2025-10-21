/// Functions to derive common graph/tree theory objects from Phylo2Vec vectors
use std::collections::HashMap;

use bit_set::BitSet;
use ndarray::{array, s, Array1, Array2};

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
pub fn _cophenetic_distances(
    v: &[usize],
    bls: Option<&Vec<[f64; 2]>>,
    unrooted: bool,
) -> Array2<f64> {
    let k = v.len();

    // Special case for 1-leaf tree
    if k == 0 {
        return array![[0.0]];
    }

    let bls = match bls {
        Some(b) => b.to_vec(),
        None => ones(k),
    };

    // Special case for a 2-leaf tree
    // Cannot be unrooted (see ete3)
    // so we return the sum of their branch lengths
    if k == 1 {
        return array![[0.0, bls[0][0] + bls[0][1]], [bls[0][0] + bls[0][1], 0.0]];
    }

    let mut ancestry = to_ancestry(v);

    if unrooted {
        let nrows = ancestry.len();
        let ncols = ancestry[0].len();
        ancestry[nrows - 1][ncols - 1] = ancestry[nrows - 1].iter().max().unwrap() - 1;
    }

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
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::vector::graph::cophenetic_distances;
///
/// let v = vec![0, 1, 2];
/// let unrooted = false;
/// let dist = cophenetic_distances(&v, unrooted);
/// assert_eq!(dist, array![[0.0, 3.0, 4.0, 4.0], [3.0, 0.0, 3.0, 3.0], [4.0, 3.0, 0.0, 2.0], [4.0, 3.0, 2.0, 0.0]]);
/// ```
pub fn cophenetic_distances(v: &[usize], unrooted: bool) -> Array2<f64> {
    _cophenetic_distances(v, None, unrooted)
}

/// Generic function to calculate a block matrix
/// from which the precision matrix can be derived using Schur's complement.
/// Works with a Phylo2Vec vector with or without branch lengths.
/// Output is a matrix of dimensions 2 * (n - 1) x 2 * (n - 1)
/// where n = number of leaves.
/// Inspired from the `inverseA` function in the `MCMCglmm` R package: <https://github.com/cran/MCMCglmm>
pub fn _pre_precision(v: &[usize], bls: Option<&Vec<[f64; 2]>>) -> Array2<f64> {
    let k = v.len();

    assert!(
        k > 0,
        "Precision matrix not supported for trees with < 2 leaves."
    );

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

/// Get a precursor of the precision matrix
/// for a Phylo2Vec vector.
/// Output is a matrix of dimensions 2 * (n - 1) x 2 * (n - 1)
/// where n = number of leaves.
/// The precision matrix can be obtained using Schur's complement
pub fn pre_precision(v: &[usize]) -> Array2<f64> {
    _pre_precision(v, None)
}

/// Get all leaves under each internal node as well as all edges from a list of pairs
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
pub fn _vcv(v: &[usize], bls: Option<&Vec<[f64; 2]>>) -> Array2<f64> {
    let k = v.len();

    assert!(
        k > 0,
        "Variance-covariance matrix not supported for trees with < 2 leaves."
    );

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

/// Get the variance-covariance matrix
/// for a Phylo2Vec vector.
/// Output is a matrix of dimensions n x n
/// where n = number of leaves.
///
/// # Examples
///
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
pub fn vcv(v: &[usize]) -> Array2<f64> {
    _vcv(v, None)
}

/// The `Incidence` struct represents an incidence matrix for a phylogenetic tree
/// derived from a Phylo2Vec vector.
/// Note that the Phylo2Vec vector represents a rooted binary tree, which can be
/// viewed as a directed acyclic graph (DAG). Duplicates are not allowed.
///
/// # Fields
///
/// - `n_nodes`: The total number of nodes in the tree. This is calculated as `2*k + 1`,
///   where `k` is the length of the input Phylo2Vec vector.
/// - `n_edges`: The total number of edges in the tree. This is calculated as `2*k`,
///   where `k` is the length of the input Phylo2Vec vector.
/// - `edges`: A vector of tuples where each tuple `(usize, usize)` represents a directed
///   edge in the tree. In each tuple, the first element is the child node and the second
///   element is the parent node.
///
/// # Methods
///
/// - `new(v: &[usize]) -> Self`
///   - Constructs a new `Incidence` instance from a Phylo2Vec vector `v`.
/// - `to_dense(&self) -> Array2<i8>`
///   - Returns the incidence matrix in a dense format as a 2D ndarray of type `i8`.
/// - `to_coo(&self) -> (Vec<i8>, Vec<usize>, Vec<usize>)`
///   - Constructs the incidence matrix in Coordinate (COO) format, returning the data,
///     row indices, and column indices of non-zero entries.
/// - `to_dok(&self) -> HashMap<(usize, usize), i8>`
///   - Provides a dictionary-of-keys (DOK) representation of the incidence matrix,
///     where the keys are `(row, column)` pairs mapping to their corresponding non-zero value.
/// - `to_csr(&self) -> (Vec<i8>, Vec<usize>, Vec<usize>)`
///   - Produces the incidence matrix in Compressed Sparse Row (CSR) format, returning
///     the non-zero values, column indices, and row pointer array.
///
/// # Usage
///
/// The `Incidence` struct is useful for converting a phylogenetic tree represented
/// by a Phylo2Vec vector into various matrix formats, which can then be used in further
/// algorithms that require a graph representation of the tree, such as network flow
/// computations, spectral analysis, or other numerical methods on sparse matrices.
pub struct Incidence {
    n_nodes: usize,
    n_edges: usize,
    edges: Vec<(usize, usize)>,
}

impl Incidence {
    pub fn new(v: &[usize]) -> Self {
        let k = v.len();
        let edges = to_edges(v);
        let n_nodes = 2 * k + 1;
        let n_edges = 2 * k;
        Incidence {
            n_nodes,
            n_edges,
            edges,
        }
    }

    /// Create an incidence matrix from a Phylo2Vec vector
    /// Using a dense representation.
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use phylo2vec::vector::graph::Incidence;
    /// let v = vec![0, 1, 2];
    /// let inc = Incidence::new(&v);
    /// let dense_matrix = inc.to_dense();
    /// assert_eq!(
    ///   dense_matrix,
    ///   array![[ 1,  0,  0,  0,  0,  0],
    ///          [ 0,  1,  0,  0,  0,  0],
    ///          [ 0,  0,  1,  0,  0,  0],
    ///          [ 0,  0,  0,  1,  0,  0],
    ///          [ 0,  0, -1, -1,  1,  0],
    ///          [ 0, -1,  0,  0, -1,  1],
    ///          [-1,  0,  0,  0,  0, -1]
    ///         ]
    /// );
    /// ```
    pub fn to_dense(&self) -> Array2<i8> {
        // n_nodes x n_edges
        let mut out = Array2::<i8>::zeros((self.n_nodes, self.n_edges));

        for &(u, v) in self.edges.iter() {
            // edge u leaves node v
            out[[v, u]] = -1;
            // edge u enters node u
            out[[u, u]] = 1;
        }

        out
    }

    /// Create an incidence matrix from a Phylo2Vec vector
    /// Using the COO format (a.k.a triplet format)
    /// # Examples
    ///
    /// ```
    /// use phylo2vec::vector::graph::Incidence;
    /// let v = vec![0, 1, 2];
    /// let inc = Incidence::new(&v);
    /// let (data, rows, cols) = inc.to_coo();
    /// assert_eq!(
    ///   data,
    ///   vec![-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
    /// );
    /// assert_eq!(
    ///  rows,
    ///  vec![4, 2, 4, 3, 5, 1, 5, 4, 6, 0, 6, 5]
    /// );
    /// assert_eq!(
    /// cols,
    /// vec![2, 2, 3, 3, 1, 1, 4, 4, 0, 0, 5, 5]
    /// );
    /// ```
    pub fn to_coo(&self) -> (Vec<i8>, Vec<usize>, Vec<usize>) {
        let mut data: Vec<i8> = Vec::with_capacity(2 * self.n_edges);
        let mut rows: Vec<usize> = Vec::with_capacity(2 * self.n_edges);
        let mut cols: Vec<usize> = Vec::with_capacity(2 * self.n_edges);

        for &(u, v) in self.edges.iter() {
            // edge u leaves node v
            data.push(-1);
            rows.push(v);
            cols.push(u);

            // edge u enters node u
            data.push(1);
            rows.push(u);
            cols.push(u);
        }

        (data, rows, cols)
    }

    /// Create an incidence matrix from a Phylo2Vec vector
    /// Using the DOK (dictionary-of-keys) format
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use phylo2vec::vector::graph::Incidence;
    /// let v = vec![0, 1];
    /// let inc = Incidence::new(&v);
    /// let dok = inc.to_dok();
    /// assert_eq!(dok, HashMap::from([
    ///     ((0, 0), 1),
    ///     ((1, 1), 1),
    ///     ((2, 2), 1),
    ///     ((3, 3), 1),
    ///     ((4, 0), -1),
    ///     ((3, 2), -1),
    ///     ((3, 1), -1),
    ///     ((4, 3), -1)
    /// ]));
    /// ```
    pub fn to_dok(&self) -> HashMap<(usize, usize), i8> {
        let mut out: HashMap<(usize, usize), i8> = HashMap::with_capacity(2 * self.n_edges);

        for &(u, v) in self.edges.iter() {
            // edge u leaves node v
            out.entry((v, u)).or_insert(-1);
            // edge u enters node u
            out.entry((u, u)).or_insert(1);
        }

        out
    }

    /// Create an incidence matrix from a Phylo2Vec vector
    /// Using the CSR (compressed sparse row) format
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use phylo2vec::vector::graph::Incidence;
    /// let v = vec![0, 1];
    /// let inc = Incidence::new(&v);
    /// let (data, indices, indptr) = inc.to_csr();
    /// assert_eq!(data, vec![1, 1, 1, -1, -1, 1, -1, -1]);
    /// assert_eq!(indices, vec![0, 1, 2, 1, 2, 3, 0, 3]);
    /// assert_eq!(indptr, vec![0, 1, 2, 3, 6, 8]);
    /// ```
    pub fn to_csr(&self) -> (Vec<i8>, Vec<usize>, Vec<usize>) {
        // Each edge yields two nonzero entries
        let mut triplets: Vec<(usize, usize, i8)> = Vec::with_capacity(2 * self.n_edges);
        for &(u, v) in self.edges.iter() {
            // Entry from edge leaving node v_node: row = v_node, col = u, value = -1
            triplets.push((v, u, -1));
            // Entry from edge entering node i: row = i, col = i, value = 1
            triplets.push((u, u, 1));
        }
        // Sort triplets by row index
        triplets.sort_unstable_by_key(|&(row, _, _)| row);

        let nnz = triplets.len();
        let mut data: Vec<i8> = Vec::with_capacity(nnz);
        let mut indices: Vec<usize> = Vec::with_capacity(nnz);
        // Initialize row pointer with zeros; length: n_nodes+1
        let mut indptr = vec![0; self.n_nodes + 1];

        // Build CSR structure: iterate over rows in order
        let mut current_row = 0;
        for &(row, col, value) in triplets.iter() {
            // Update indptr for rows with no entries until the current row
            while current_row < row {
                indptr[current_row + 1] = data.len();
                current_row += 1;
            }
            data.push(value);
            indices.push(col);
        }
        // Finalize remaining row pointers
        while current_row < self.n_nodes {
            indptr[current_row + 1] = data.len();
            current_row += 1;
        }

        (data, indices, indptr)
    }

    /// Create an incidence matrix from a Phylo2Vec vector
    /// Using the CSC (compressed sparse column) format
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use phylo2vec::vector::graph::Incidence;
    /// let v = vec![0, 1, 2];
    /// let inc = Incidence::new(&v);
    /// let (data, indices, indptr) = inc.to_csc();
    /// assert_eq!(data, vec![1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]);
    /// assert_eq!(indices, vec![0, 6, 1, 5, 2, 4, 3, 4, 4, 5, 5, 6]);
    /// assert_eq!(indptr, vec![0, 2, 4, 6, 8, 10, 12]);
    /// ```
    pub fn to_csc(&self) -> (Vec<i8>, Vec<usize>, Vec<usize>) {
        // Each edge yields two nonzero entries
        let mut triplets: Vec<(usize, usize, i8)> = Vec::with_capacity(2 * self.n_edges);
        for &(u, v) in self.edges.iter() {
            // Entry from edge entering node i: row = i, col = i, value = 1
            triplets.push((u, u, 1));
            // Entry from edge leaving node v_node: row = v_node, col = u, value = -1
            triplets.push((v, u, -1));
        }
        // Sort triplets by row index
        triplets.sort_unstable_by_key(|&(_, col, _)| col);

        let nnz = triplets.len();
        let mut data: Vec<i8> = Vec::with_capacity(nnz);
        let mut row_indices: Vec<usize> = Vec::with_capacity(nnz);
        let mut col_ptrs = vec![0; self.n_edges + 1];

        let mut current_col = 0;
        for &(row, col, value) in triplets.iter() {
            // Fill in any columns without entries
            while current_col < col {
                col_ptrs[current_col + 1] = data.len();
                current_col += 1;
            }
            data.push(value);
            row_indices.push(row);
        }

        // Finalize remaining column pointers
        while current_col < self.n_edges {
            col_ptrs[current_col + 1] = data.len();
            current_col += 1;
        }

        (data, row_indices, col_ptrs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(vec![], false, array![[0.0]])]
    #[case(vec![0], false, array![[0.0, 2.0], [2.0, 0.0]])]
    #[case(vec![0], true, array![[0.0, 2.0], [2.0, 0.0]])]
    #[case(vec![0, 0, 1], false, array![
        [0.0, 4.0, 2.0, 4.0],
        [4.0, 0.0, 4.0, 2.0],
        [2.0, 4.0, 0.0, 4.0],
        [4.0, 2.0, 4.0, 0.0]
    ])]
    #[case(vec![0, 0, 1], true, array![
        [0.0, 3.0, 2.0, 3.0],
        [3.0, 0.0, 3.0, 2.0],
        [2.0, 3.0, 0.0, 3.0],
        [3.0, 2.0, 3.0, 0.0]
    ])]
    #[case(vec![0, 1, 2], false, array![
        [0.0, 3.0, 4.0, 4.0],
        [3.0, 0.0, 3.0, 3.0],
        [4.0, 3.0, 0.0, 2.0],
        [4.0, 3.0, 2.0, 0.0]
    ])]
    #[case(vec![0, 1, 2], true, array![
        [0.0, 2.0, 3.0, 3.0],
        [2.0, 0.0, 3.0, 3.0],
        [3.0, 3.0, 0.0, 2.0],
        [3.0, 3.0, 2.0, 0.0]
    ])]
    #[case(vec![0, 1, 0, 4, 4, 2, 6], false, array![
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
        #[case] unrooted: bool,
        #[case] expected: Array2<f64>,
    ) {
        assert_eq!(cophenetic_distances(&v, unrooted), expected);
    }

    #[rstest]
    #[case(vec![0], array![[1.0, 0.0], [0.0, 1.0]])]
    #[case(vec![0, 0, 1], array![[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]])]
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
    #[case(vec![0, 1, 0, 4, 4, 2, 6], array![
        [3.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        [2.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 4.0, 3.0],
        [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0]])]
    fn test_vcv(#[case] v: Vec<usize>, #[case] expected: Array2<f64>) {
        assert_eq!(vcv(&v), expected);
    }

    #[rstest]
    #[should_panic]
    #[case(vec![])]
    fn test_vcv_empty(#[case] v: Vec<usize>) {
        vcv(&v);
    }

    #[rstest]
    #[case(vec![0], array![[1.0, 0.0], [0.0, 1.0]])]
    #[case(vec![0, 1, 2], array![
        [1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  0.0,  0.0, -1.0],
        [0.0,  0.0,  1.0,  0.0, -1.0,  0.0],
        [0.0,  0.0,  0.0,  1.0, -1.0,  0.0],
        [0.0,  0.0, -1.0, -1.0,  3.0, -1.0],
        [0.0, -1.0,  0.0,  0.0, -1.0,  3.0]])]
    #[case(vec![0, 1, 1, 4], array![
        [1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, -1.0],
        [0.0,  0.0,  0.0,  1.0,  0.0, -1.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  0.0,  1.0,  0.0, -1.0,  0.0],
        [0.0, -1.0,  0.0, -1.0,  0.0,  3.0, -1.0,  0.0],
        [0.0,  0.0,  0.0,  0.0, -1.0, -1.0,  3.0, -1.0],
        [0.0,  0.0, -1.0,  0.0,  0.0,  0.0, -1.0,  3.0]])]
    fn test_pre_precision(#[case] v: Vec<usize>, #[case] expected: Array2<f64>) {
        assert_eq!(pre_precision(&v), expected);
    }

    #[rstest]
    #[should_panic]
    #[case(vec![])]
    fn test_pre_precision_empty(#[case] v: Vec<usize>) {
        pre_precision(&v);
    }

    #[rstest]
    #[case(vec![0], array![[1, 0], [0, 1], [-1, -1]])]
    #[case(vec![0, 1], array![
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0, -1, -1,  1],
        [-1,  0,  0, -1]
    ])]
    #[case(vec![0, 1, 2], array![
        [ 1,  0,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  1,  0,  0],
        [ 0,  0, -1, -1,  1,  0],
        [ 0, -1,  0,  0, -1,  1],
        [-1,  0,  0,  0,  0, -1]
    ])]
    #[case(vec![0, 2, 2], array![
        [ 1,   0,  0,  0,  0,  0],
        [ 0,   1,  0,  0,  0,  0],
        [ 0,   0,  1,  0,  0,  0],
        [ 0,   0,  0,  1,  0,  0],
        [ 0,   0, -1, -1,  1,  0],
        [-1,  -1,  0,  0,  0,  1],
        [ 0,   0,  0,  0, -1, -1]])]
    fn test_incidence_dense(#[case] v: Vec<usize>, #[case] expected: Array2<i8>) {
        let incid = Incidence::new(&v);
        assert_eq!(incid.to_dense(), expected);
    }

    #[rstest]
    #[case(vec![0], vec![-1, 1, -1, 1], vec![2, 0, 2, 1], vec![0, 0, 1, 1])]
    #[case(vec![0, 1], vec![-1, 1, -1, 1, -1, 1, -1, 1], vec![3, 1, 3, 2, 4, 0, 4, 3], vec![1, 1, 2, 2, 0, 0, 3, 3])]
    #[case(vec![0, 1, 2], vec![-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1], vec![4, 2, 4, 3, 5, 1, 5, 4, 6, 0, 6, 5], vec![2, 2, 3, 3, 1, 1, 4, 4, 0, 0, 5, 5])]
    #[case(vec![0, 2, 2], vec![-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1], vec![4, 2, 4, 3, 5, 0, 5, 1, 6, 5, 6, 4], vec![2, 2, 3, 3, 0, 0, 1, 1, 5, 5, 4, 4])]
    fn test_incidence_coo(
        #[case] v: Vec<usize>,
        #[case] expected_data: Vec<i8>,
        #[case] expected_rows: Vec<usize>,
        #[case] expected_cols: Vec<usize>,
    ) {
        let incid = Incidence::new(&v);
        let (data, rows, cols) = incid.to_coo();
        assert_eq!(data, expected_data);
        assert_eq!(rows, expected_rows);
        assert_eq!(cols, expected_cols);
    }

    #[rstest]
    #[case(
        vec![0],
        HashMap::from(
            [
                ((0, 0), 1),
                ((2, 0), -1),
                ((2, 1), -1),
                ((1, 1), 1)
            ]
        )
    )]
    #[case(
        vec![0, 1, 2],
        HashMap::from(
            [
                ((0, 0), 1),
                ((1, 1), 1),
                ((2, 2), 1),
                ((3, 3), 1),
                ((4, 4), 1),
                ((5, 5), 1),
                ((4, 2), -1),
                ((4, 3), -1),
                ((5, 1), -1),
                ((5, 4), -1),
                ((6, 0), -1),
                ((6, 5), -1)
            ]
        )
    )]
    #[case(
        vec![0, 2, 2],
        HashMap::from(
            [
                ((0, 0), 1),
                ((1, 1), 1),
                ((2, 2), 1),
                ((3, 3), 1),
                ((4, 4), 1),
                ((5, 5), 1),
                ((4, 2), -1),
                ((4, 3), -1),
                ((5, 0), -1),
                ((5, 1), -1),
                ((6, 5), -1),
                ((6, 4), -1)
            ]
        )
    )]
    fn test_incidence_dok(#[case] v: Vec<usize>, #[case] expected: HashMap<(usize, usize), i8>) {
        let incid = Incidence::new(&v);
        assert_eq!(incid.to_dok(), expected);
    }

    #[rstest]
    #[case(vec![0], vec![1, 1, -1, -1], vec![0, 1, 0, 1], vec![0, 1, 2, 4])]
    #[case(vec![0, 1], vec![1, 1, 1, -1, -1, 1, -1, -1], vec![0, 1, 2, 1, 2, 3, 0, 3], vec![0, 1, 2, 3, 6, 8])]
    #[case(vec![0, 1, 2], vec![ 1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1, -1], vec![0, 1, 2, 3, 2, 3, 4, 1, 4, 5, 0, 5], vec![ 0,  1,  2,  3,  4,  7, 10, 12])]
    #[case(vec![0, 2, 2], vec![ 1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1, -1], vec![0, 1, 2, 3, 2, 3, 4, 0, 1, 5, 5, 4], vec![ 0,  1,  2,  3,  4,  7, 10, 12])]
    fn test_incidence_csr(
        #[case] v: Vec<usize>,
        #[case] expected_data: Vec<i8>,
        #[case] expected_indices: Vec<usize>,
        #[case] expected_indptr: Vec<usize>,
    ) {
        let incid = Incidence::new(&v);
        let (data, indices, indptr) = incid.to_csr();
        assert_eq!(data, expected_data);
        assert_eq!(indices, expected_indices);
        assert_eq!(indptr, expected_indptr);
    }

    #[rstest]
    #[case(vec![0], vec![1, -1, 1, -1], vec![0, 2, 1, 2], vec![0, 2, 4])]
    #[case(vec![0, 1], vec![1, -1, 1, -1, 1, -1, 1, -1], vec![0, 4, 1, 3, 2, 3, 3, 4], vec![0, 2, 4, 6, 8])]
    #[case(vec![0, 1, 2], vec![1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], vec![0, 6, 1, 5, 2, 4, 3, 4, 4, 5, 5, 6], vec![0, 2, 4, 6, 8, 10, 12])]
    #[case(vec![0, 2, 2], vec![1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], vec![0, 5, 1, 5, 2, 4, 3, 4, 4, 6, 5, 6], vec![0, 2, 4, 6, 8, 10, 12])]
    fn test_incidence_csc(
        #[case] v: Vec<usize>,
        #[case] expected_data: Vec<i8>,
        #[case] expected_indices: Vec<usize>,
        #[case] expected_indptr: Vec<usize>,
    ) {
        let incid = Incidence::new(&v);
        let (data, indices, indptr) = incid.to_csc();
        assert_eq!(data, expected_data);
        assert_eq!(indices, expected_indices);
        assert_eq!(indptr, expected_indptr);
    }
}
