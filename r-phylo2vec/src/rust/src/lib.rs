use std::collections::HashMap;
use std::result::Result;

use extendr_api::prelude::*;
use ndarray::Array2;

use phylo2vec::matrix::base as mbase;
use phylo2vec::matrix::convert as mconvert;
use phylo2vec::matrix::graph as mgraph;
use phylo2vec::matrix::ops as mops;
use phylo2vec::newick;
use phylo2vec::vector::balance as vbalance;
use phylo2vec::vector::base as vbase;
use phylo2vec::vector::convert as vconvert;
use phylo2vec::vector::distance as vdist;
use phylo2vec::vector::graph as vgraph;
use phylo2vec::vector::ops as vops;

enum RTree {
    Matrix(RMatrix<f64>),
    Vector(Vec<i32>),
}

impl TryFrom<Robj> for RTree {
    type Error = Error;

    fn try_from(robj: Robj) -> Result<Self, Self::Error> {
        if robj.is_matrix() {
            Ok(RTree::Matrix(robj.try_into()?))
        } else {
            Ok(RTree::Vector(robj.try_into()?))
        }
    }
}

// Convert Vec<i32> to Vec<usize>
fn as_usize(v: Vec<i32>) -> Vec<usize> {
    v.iter().map(|&x| x as usize).collect()
}

// Convert Vec<usize> to Vec<i32>
fn as_i32(v: Vec<usize>) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

// Convert R matrix to Rust Array2<f64>.
fn convert_from_rmatrix(matrix: &Robj) -> Result<Array2<f64>, &'static str> {
    let data = matrix.as_real_slice().ok_or("Expected numeric matrix")?;
    let dims = matrix.dim().ok_or("Matrix is missing dimensions")?;

    let (nrows, ncols) = (dims[0].inner() as usize, dims[1].inner() as usize);

    if data.len() != nrows * ncols {
        return Err("Matrix dimensions do not match data length");
    }
    // Create a 2D array with the specified dimensions
    let mut array = Array2::<f64>::zeros((nrows, ncols));
    for i in 0..nrows {
        for j in 0..ncols {
            array[[i, j]] = data[j * nrows + i];
        }
    }

    Ok(array)
}

/// Sample a random tree topology via phylo2vec
///
/// In ordered trees, leaf i grows from a sister leaf (i.e., attaches to a leaf branch j <= i)
/// In unordered trees, leaf i grows from any node (i.e., attaches to any branch j <= 2*i)
///
/// @param n_leaves Number of leaves (must be at least 2)
/// @param ordered Whether to sample an ordered tree
/// @return A phylo2vec vector representing the sampled tree topology (length: n_leaves - 1)
/// @export
#[extendr]
fn sample_vector(n_leaves: isize, ordered: bool) -> Result<Vec<i32>, Error> {
    if n_leaves < 2 {
        Err(Error::OutOfRange("n_leaves must be at least 2".into()))
    } else {
        Ok(as_i32(vbase::sample_vector(n_leaves as usize, ordered)))
    }
}

/// Sample a random tree with branch lengths via phylo2vec
///
/// @param n_leaves Number of leaves (must be at least 2)
/// @param ordered Whether to sample an ordered tree
/// @return A phylo2vec matrix representing the sampled tree with branch lengths (shape: (n_leaves - 1, 3))
/// @export
#[extendr]
fn sample_matrix(n_leaves: isize, ordered: bool) -> Result<RMatrix<f64>, Error> {
    if n_leaves < 2 {
        Err(Error::OutOfRange("n_leaves must be at least 2".into()))
    } else {
        let matrix = mbase::sample_matrix(n_leaves as usize, ordered);
        let shape = matrix.shape();
        Ok(RMatrix::new_matrix(shape[0], shape[1], |r, c| {
            matrix[[r, c]]
        }))
    }
}

/// Recover a rooted tree (in Newick format) from a phylo2vec tree
///
/// @param tree A phylo2vec tree
/// @return Newick string representation of the tree
/// @export
#[extendr]
fn to_newick(tree: RTree) -> String {
    match tree {
        RTree::Matrix(m) => {
            let matrix = convert_from_rmatrix(&m).unwrap();
            mconvert::to_newick(&matrix.view())
        }
        RTree::Vector(v) => {
            let v_usize = as_usize(v);
            vconvert::to_newick(&v_usize)
        }
    }
}

// Convert a newick string to a phylo2vec vector
#[extendr]
fn to_vector(newick: &str) -> Vec<i32> {
    let v = vconvert::from_newick(newick);
    as_i32(v)
}

// Convert a newick string to a phylo2vec matrix
#[extendr]
fn to_matrix(newick: &str) -> RMatrix<f64> {
    let matrix = mconvert::from_newick(newick);
    let shape = matrix.shape();
    RMatrix::new_matrix(shape[0], shape[1], |r, c| matrix[[r, c]])
}

/// Validate a phylo2vec vector
///
/// Raises an error if the vector is invalid.
///
/// @param vector phylo2vec vector representation of a tree topology
/// @export
#[extendr]
fn check_v(vector: Vec<i32>) {
    let v_usize = as_usize(vector);
    vbase::check_v(&v_usize);
}

/// Validate a phylo2vec matrix
///
/// Raises an error if the matrix is invalid.
///
/// @param vector phylo2vec matrix representation of a tree (with branch lengths)
/// @export
#[extendr]
fn check_m(vector: RMatrix<f64>) {
    let matrix = convert_from_rmatrix(&vector).unwrap();
    mbase::check_m(&matrix.view());
}

/// Get the ancestry matrix of a phylo2vec vector
///
/// @param vector phylo2vec vector representation of a tree topology
/// @return Ancestry representation (shape: (n_leaves - 1, 3))
/// @export
#[extendr]
fn to_ancestry(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let ancestry = vconvert::to_ancestry(&v_usize);

    RMatrix::new_matrix(k, 3, |r, c| ancestry[r][c] as i32)
}

/// Convert an ancestry matrix to a phylo2vec vector
///
/// @param matrix Ancestry representation (shape: (n_leaves - 1, 3))
/// @return phylo2vec vector representation
/// @export
#[extendr]
fn from_ancestry(matrix: RMatrix<i32>) -> Vec<i32> {
    let data = matrix.data();
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let ancestry: Vec<Vec<usize>> = (0..nrows)
        .map(|r| (0..ncols).map(|c| data[c * nrows + r] as usize).collect())
        .collect();

    let ancestry_fixed: Vec<[usize; 3]> = ancestry
        .iter()
        .map(|row| {
            if row.len() == 3 {
                [row[0], row[1], row[2]]
            } else {
                panic!("Expected each row of ancestry matrix to have exactly 3 elements");
            }
        })
        .collect();

    let v = vconvert::from_ancestry(&ancestry_fixed);
    as_i32(v)
}

/// Get pairs from a phylo2vec vector
///
/// @param vector phylo2vec vector representation of a tree topology
/// @return Pairs representation (shape: (n_leaves - 1, 2))
/// @export
#[extendr]
fn to_pairs(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let pairs = vconvert::to_pairs(&v_usize);
    RMatrix::new_matrix(k, 2, |r, c| match c {
        0 => pairs[r].0 as i32,
        1 => pairs[r].1 as i32,
        _ => unreachable!(),
    })
}

/// Convert a pairs matrix to a phylo2vec vector
///
/// @param pairs Pairs representation (shape: (n_leaves - 1, 2))
/// @return phylo2vec vector representation
/// @export
#[extendr]
fn from_pairs(pairs: RMatrix<i32>) -> Vec<i32> {
    let data = pairs.data();
    let nrows = pairs.nrows();
    let pairs_usize: Vec<(usize, usize)> = (0..nrows)
        .map(|r| (data[r] as usize, data[nrows + r] as usize))
        .collect();
    let v = vconvert::from_pairs(&pairs_usize);
    as_i32(v)
}

/// Get the edge list of a phylo2vec vector
///
/// @param vector phylo2vec vector representation of a tree topology
/// @return Edge list representation (shape: (2*(n_leaves - 1), 2))
/// @export
#[extendr]
fn to_edges(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let edges = vconvert::to_edges(&v_usize);
    RMatrix::new_matrix(2 * k, 2, |r, c| match c {
        0 => edges[r].0 as i32,
        1 => edges[r].1 as i32,
        _ => unreachable!(),
    })
}

/// Convert an edge list to a phylo2vec vector
///
/// @param edges Edge list representation (shape: (2*(n_leaves - 1), 2))
/// @return phylo2vec vector representation
/// @export
#[extendr]
fn from_edges(edges: RMatrix<i32>) -> Vec<i32> {
    let data = edges.data();
    let nrows = edges.nrows();
    let edges_usize: Vec<(usize, usize)> = (0..nrows)
        .map(|r| (data[r] as usize, data[nrows + r] as usize))
        .collect();
    let v = vconvert::from_edges(&edges_usize);
    as_i32(v)
}

/// Add a leaf to a phylo2vec vector
///
/// @param vector phylo2vec vector representation of a tree topology
/// @param leaf The leaf to add (0-indexed)
/// @param branch The branch to attach the new leaf to (0-indexed)
/// @return New vector with the added leaf
/// @export
#[extendr]
fn add_leaf(vector: Vec<i32>, leaf: i32, branch: i32) -> Vec<i32> {
    let mut v_usize: Vec<usize> = as_usize(vector);
    let new_v = vops::add_leaf(&mut v_usize, leaf as usize, branch as usize);
    as_i32(new_v)
}

/// Remove a leaf from a phylo2vec vector
///
/// @param vector phylo2vec vector representation of a tree topology
/// @param leaf The leaf to remove (0-indexed)
/// @return A list with two elements:
/// - `v`: New vector with the removed leaf
/// - `branch`: The branch the removed leaf was attached to (0-indexed)
/// @export
#[extendr]
fn remove_leaf(vector: Vec<i32>, leaf: i32) -> Robj {
    let mut v_usize: Vec<usize> = as_usize(vector);
    let (new_v, branch) = vops::remove_leaf(&mut v_usize, leaf as usize);
    list!(v = as_i32(new_v), branch = branch as i32).into()
}

/// Get the first recent common ancestor between two nodes in a phylo2vec tree
/// node1 and node2 can be leaf nodes (0 to n_leaves) or internal nodes (n_leaves to 2*(n_leaves-1)).
/// Similar to ape's `getMRCA` function in R (for leaf nodes)
/// and ETE's `get_common_ancestor` in Python (for all nodes), but for phylo2vec vectors.
///
/// @param tree A phylo2vec tree
/// @param node1 The first node (0-indexed)
/// @param node2 The second node (0-indexed)
/// @return The common ancestor node (0-indexed)
/// @export
#[extendr]
fn get_common_ancestor(tree: RTree, node1: i32, node2: i32) -> Result<i32, Error> {
    let node1 = node1 as usize;
    let node2 = node2 as usize;
    let mrca = match tree {
        RTree::Matrix(m) => {
            let matrix = convert_from_rmatrix(&m).unwrap();
            let (v, _) = mbase::parse_matrix(&matrix.view());
            vops::get_common_ancestor(&v, node1, node2)
        }
        RTree::Vector(v) => {
            let v_usize: Vec<usize> = as_usize(v);
            vops::get_common_ancestor(&v_usize, node1, node2)
        }
    };
    mrca.map(|node| node as i32)
        .map_err(|e| Error::OutOfRange(e.into()))
}

/// Get the depth of a node in a phylo2vec tree
/// Depth is the distance from root to that node. Root has depth 0.
/// For vectors, this is the topological depth. For matrices, this is the depth with branch lengths.
///
/// @param tree A phylo2vec tree
/// @param node The node to get the depth of (0-indexed)
/// @return The depth of the node
/// @export
#[extendr]
fn get_node_depth(tree: RTree, node: i32) -> Result<f64, Error> {
    let node_usize = node as usize;
    let depth = match tree {
        RTree::Matrix(m) => {
            let matrix = convert_from_rmatrix(&m).unwrap();
            mops::get_node_depth(&matrix.view(), node_usize)
        }
        RTree::Vector(v) => {
            let v_usize: Vec<usize> = as_usize(v);
            vops::get_node_depth(&v_usize, node_usize)
        }
    };
    depth.map_err(|e| Error::OutOfRange(e.into()))
}

/// Get the depths of all nodes in a phylo2vec tree
/// Depth is the distance from root to each node. Root has depth 0.
///
/// @param tree A phylo2vec tree
/// @return A vector of depths for all nodes (length: 2*n_leaves - 1)
/// @export
#[extendr]
fn get_node_depths(tree: RTree) -> Vec<f64> {
    match tree {
        RTree::Matrix(m) => {
            let matrix = convert_from_rmatrix(&m).unwrap();
            mops::get_node_depths(&matrix.view())
        }
        RTree::Vector(v) => {
            let v_usize: Vec<usize> = as_usize(v);
            vops::get_node_depths(&v_usize)
        }
    }
}

/// Produce an ordered version (i.e., birth-death process version)
/// of a phylo2vec vector using the Queue Shuffle algorithm.
///
/// Queue Shuffle ensures that the output tree is ordered,
/// while also ensuring a smooth path through the space of orderings
///
/// for more details, see https://doi.org/10.1093/gbe/evad213
///
/// @param vector phylo2vec vector representation of a tree topology
/// @param shuffle_cherries If true, the algorithm will randomly shuffle the order of cherries (i.e., pairs of leaves)
/// @return A list with two elements:
/// - `v`: The ordered phylo2vec vector
/// - `mapping`: A mapping of the original labels to the new labels
/// @export
#[extendr]
fn queue_shuffle(vector: Vec<i32>, shuffle_cherries: bool) -> List {
    let v_usize: Vec<usize> = as_usize(vector);
    let (v_new, label_mapping) = vops::queue_shuffle(&v_usize, shuffle_cherries);
    list!(v = as_i32(v_new), mapping = as_i32(label_mapping))
}

/// Check if a newick string has branch lengths
///
/// @param newick Newick representation of a tree
/// @return TRUE if the newick has branch lengths, FALSE otherwise
/// @export
#[extendr]
fn has_branch_lengths(newick: &str) -> bool {
    newick::has_branch_lengths(newick)
}

/// Find the number of leaves in a Newick string
///
/// @param newick Newick representation of a tree
/// @return Number of leaves
/// @export
#[extendr]
fn find_num_leaves(newick: &str) -> usize {
    newick::find_num_leaves(newick)
}

/// Create an integer-taxon label mapping (label_mapping)
/// from a string-based newick (where leaves are strings)
/// and produce a mapped integer-based newick (where leaves are integers).
///
/// Note 1: this does not check for the validity of the Newick string.
///
/// Note 2: the parent nodes are removed from the output,
/// but the branch lengths/annotations are kept.
///
/// @param newick Newick representation of a tree with string labels
/// @return A list with two elements:
/// - `newick`: Newick with integer labels
/// - `mapping`: A vector where the index corresponds to the integer label
/// @export
#[extendr]
fn create_label_mapping(newick: &str) -> List {
    let (nw_int, label_mapping) = newick::create_label_mapping(newick);

    let mut label_mapping_list: Vec<String> = Vec::new();

    for i in 0..label_mapping.len() {
        label_mapping_list.push(label_mapping[&i].clone());
    }

    list!(newick = nw_int, mapping = label_mapping_list)
}

/// Apply an integer-taxon label mapping (label_mapping)
/// to an integer-based newick (where leaves are integers)
/// and produce a mapped Newick (where leaves are strings (taxa))
///
/// For more details, see `create_label_mapping`.
///
/// Note 1: this does not check for the validity of the Newick string.
///
/// Note 2: the parent nodes are removed from the output,
/// but the branch lengths/annotations are kept.
///
/// @param newick Newick representation of a tree with integer labels
/// @param label_mapping_list A vector where the index corresponds to the integer label
/// @return Newick with string labels
/// @export
#[extendr]
fn apply_label_mapping(newick: &str, label_mapping_list: Vec<String>) -> String {
    let label_mapping_hash = label_mapping_list
        .iter()
        .enumerate()
        .map(|(i, label)| (i, label.clone()))
        .collect::<HashMap<usize, String>>();

    // Not great, but cannot figure out how to map_err to an rextendr error
    newick::apply_label_mapping(newick, &label_mapping_hash).unwrap()
}

/// Remove parent labels from the Newick string
///
/// @param newick Newick representation of a tree with branch lengths
/// @return Newick string without branch lengths
/// @export
#[extendr]
fn remove_branch_lengths(newick: &str) -> String {
    newick::remove_branch_lengths(newick)
}

/// Remove parent labels from the Newick string
///
/// @param newick Newick representation of a tree with parent labels
/// @return Newick string without parent labels
/// @export
#[extendr]
fn remove_parent_labels(newick: &str) -> String {
    newick::remove_parent_labels(newick)
}

/// Get the (topological) cophenetic distance matrix of a phylo2vec tree
///
/// The cophenetic distance between two leaves is the distance from each leaf to their most recent common ancestor.
/// For phylo2vec vectors, this is the topological distance. For phylo2vec matrices, this is the distance with branch lengths.
///
/// @param tree A phylo2vec tree
/// @param unrooted If true, the distance is calculated as the distance between each leaf and their most recent common ancestor, multiplied by 2. If false, the distance is calculated as the distance from each leaf to their most recent common ancestor.
/// @return Cophenetic distance matrix (shape: (n_leaves, n_leaves))
/// @export
#[extendr]
fn cophenetic_distances(tree: RTree, #[default = "FALSE"] unrooted: bool) -> RMatrix<f64> {
    let coph_rs = match tree {
        RTree::Matrix(m) => {
            let matrix = convert_from_rmatrix(&m).unwrap();
            mgraph::cophenetic_distances(&matrix.view(), unrooted)
        }
        RTree::Vector(v) => {
            let v_usize: Vec<usize> = as_usize(v);
            vgraph::cophenetic_distances(&v_usize, unrooted)
        }
    };
    let n_leaves = coph_rs.shape()[0];
    let mut coph_r = RMatrix::new_matrix(n_leaves, n_leaves, |r, c| coph_rs[[r, c]]);
    let dimnames = (0..n_leaves).map(|x| x as i32).collect::<Vec<i32>>();
    coph_r.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    coph_r
}

/// Get a precursor of the precision matrix of a phylo2vec tree
/// The precision matrix is the inverse of the variance-covariance matrix.
/// The precision matrix can be obtained using Schur's complement on this precursor.
/// Output shape: (2 * (n_leaves - 1), 2 * (n_leaves- 1)]
///
/// @param tree A phylo2vec tree
/// @return A precursor of the precision matrix (shape: (2 * (n_leaves - 1), 2 * (n_leaves- 1)))
/// @export
#[extendr]
fn pre_precision(tree: RTree) -> RMatrix<f64> {
    let pre_precision_rs = match tree {
        RTree::Matrix(m) => {
            let matrix = convert_from_rmatrix(&m).unwrap();
            mgraph::pre_precision(&matrix.view())
        }
        RTree::Vector(v) => {
            let v_usize: Vec<usize> = as_usize(v);
            vgraph::pre_precision(&v_usize)
        }
    };
    let n = pre_precision_rs.shape()[0];
    let mut preprecision_r = RMatrix::new_matrix(n, n, |r, c| pre_precision_rs[[r, c]]);
    let dimnames = (0..n).map(|x| x as i32).collect::<Vec<i32>>();
    preprecision_r.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    preprecision_r
}

/// Get the variance-covariance matrix of a phylo2vec tree
///
/// @param tree A phylo2vec tree
/// @return Variance-covariance matrix (shape: (n_leaves, n_leaves))
/// @export
#[extendr]
fn vcovp(tree: RTree) -> RMatrix<f64> {
    let vcv_rs = match tree {
        RTree::Matrix(m) => {
            let matrix = convert_from_rmatrix(&m).unwrap();
            mgraph::vcv(&matrix.view())
        }
        RTree::Vector(v) => {
            let v_usize: Vec<usize> = as_usize(v);
            vgraph::vcv(&v_usize)
        }
    };
    let n_leaves = vcv_rs.shape()[0];
    let mut vcv_r = RMatrix::new_matrix(n_leaves, n_leaves, |r, c| vcv_rs[[r, c]]);
    let dimnames = (0..n_leaves).map(|x| x as i32).collect::<Vec<i32>>();
    vcv_r.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    vcv_r
}

// Get the oriented incidence matrix of a phylo2vec vector in dense format
#[extendr]
fn incidence_dense(input_vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize = as_usize(input_vector);
    let k = v_usize.len();
    let dense_rs = vgraph::Incidence::new(&v_usize).to_dense();
    let mut dense_r = RMatrix::new_matrix(2 * k + 1, 2 * k, |r, c| dense_rs[[r, c]] as i32);
    let colnames = (0..2 * k).map(|x| x as i32).collect::<Vec<i32>>();
    let mut rownames = colnames.clone();
    rownames.push(2 * k as i32);
    dense_r.set_dimnames(List::from_values(vec![rownames, colnames]));

    dense_r
}

// Get the incidence matrix of a phylo2vec vector in COO format
#[extendr]
fn incidence_coo(input_vector: Vec<i32>) -> List {
    let v_usize = as_usize(input_vector);
    let (data, rows, cols) = vgraph::Incidence::new(&v_usize).to_coo();
    list!(
        data = data.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
        rows = as_i32(rows),
        cols = as_i32(cols)
    )
}

// Get the incidence matrix of a phylo2vec vector in CSC format
#[extendr]
fn incidence_csc(input_vector: Vec<i32>) -> List {
    let v_usize = as_usize(input_vector);
    let (data, indices, indptr) = vgraph::Incidence::new(&v_usize).to_csc();
    list!(
        data = data.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
        indices = as_i32(indices),
        indptr = as_i32(indptr)
    )
}

// Get the incidence matrix of a phylo2vec vector in CSR format
#[extendr]
fn incidence_csr(input_vector: Vec<i32>) -> List {
    let v_usize = as_usize(input_vector);
    let (data, indices, indptr) = vgraph::Incidence::new(&v_usize).to_csr();
    list!(
        data = data.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
        indices = as_i32(indices),
        indptr = as_i32(indptr)
    )
}

/// Compute the Robinson-Foulds distance between two trees.
///
/// RF distance counts the number of bipartitions (splits) that differ
/// between two tree topologies. Lower values indicate more similar trees.
///
/// @param v1 First tree as phylo2vec vector
/// @param v2 Second tree as phylo2vec vector
/// @param normalize If TRUE, return normalized distance in range `[0.0, 1.0]`
/// @return RF distance (numeric)
/// @export
#[extendr]
fn robinson_foulds(v1: Vec<i32>, v2: Vec<i32>, #[default = "FALSE"] normalize: bool) -> f64 {
    let v1_usize = as_usize(v1);
    let v2_usize = as_usize(v2);
    vdist::robinson_foulds(&v1_usize, &v2_usize, normalize)
}

/// Compute the Sackin index of a tree.
///
/// The Sackin index is a measure of tree imbalance, defined as the sum of the depths of all leaves in the tree.
/// Higher values indicate more imbalanced trees, while lower values indicate more balanced trees.
///
/// @param vector phylo2vec vector representation of a tree topology
/// @return Sackin index (numeric)
/// @export
#[extendr]
fn sackin(vector: Vec<i32>) -> i32 {
    let v_usize = as_usize(vector);
    vbalance::sackin(&v_usize) as i32
}

/// Compute the variance of leaf depths in a tree.
///
/// Higher values indicate more imbalanced trees, while lower values indicate more balanced trees.
///
/// @param vector phylo2vec vector representation of a tree topology
/// @return Variance of leaf depths (numeric)
/// @export
#[extendr]
fn leaf_depth_variance(vector: Vec<i32>) -> f64 {
    let v_usize = as_usize(vector);
    vbalance::leaf_depth_variance(&v_usize)
}

/// Compute the B2 index of a tree from Shao and Sokal (1990).
///
/// The B2 index is a measure of tree balance based on the probabilities of random walks from the root to each leaf.
/// For a binary tree, the B2 index can be calculated as the sum of the depth of each leaf multiplied by 2 raised to the power of negative depth of that leaf.
/// Higher values indicate more balanced trees, while lower values indicate more imbalanced trees.
/// For more details, see https://doi.org/10.1007/s00285-021-01662-7.
///
/// @param vector phylo2vec vector representation of a tree topology
/// @return B2 index (numeric)
/// @export
#[extendr]
fn b2(vector: Vec<i32>) -> f64 {
    let v_usize = as_usize(vector);
    vbalance::b2(&v_usize)
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod phylo2vec;
    fn add_leaf;
    fn b2;
    fn apply_label_mapping;
    fn check_m;
    fn check_v;
    fn cophenetic_distances;
    fn create_label_mapping;
    fn find_num_leaves;
    fn from_ancestry;
    fn from_edges;
    fn from_pairs;
    fn get_common_ancestor;
    fn get_node_depth;
    fn get_node_depths;
    fn has_branch_lengths;
    fn incidence_coo;
    fn incidence_csc;
    fn incidence_csr;
    fn incidence_dense;
    fn leaf_depth_variance;
    fn queue_shuffle;
    fn remove_branch_lengths;
    fn remove_parent_labels;
    fn pre_precision;
    fn remove_leaf;
    fn robinson_foulds;
    fn sackin;
    fn sample_matrix;
    fn sample_vector;
    fn to_ancestry;
    fn to_edges;
    fn to_newick;
    fn to_pairs;
    fn to_matrix;
    fn to_vector;
    fn vcovp;
}
