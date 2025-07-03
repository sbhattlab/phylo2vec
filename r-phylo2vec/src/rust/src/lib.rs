use std::collections::HashMap;
use std::result::Result;

use extendr_api::prelude::*;
use ndarray::{Array2, ArrayView2};

use phylo2vec::matrix::base as mbase;
use phylo2vec::matrix::convert as mconvert;
use phylo2vec::matrix::graph as mgraph;
use phylo2vec::newick;
use phylo2vec::vector::base as vbase;
use phylo2vec::vector::convert as vconvert;
use phylo2vec::vector::graph as vgraph;
use phylo2vec::vector::ops as vops;

fn as_usize(v: Vec<i32>) -> Vec<usize> {
    v.iter().map(|&x| x as usize).collect()
}

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

/// Sample a random tree topology via Phylo2Vec
/// @export
#[extendr]
fn sample_vector(n_leaves: usize, ordered: bool) -> Vec<i32> {
    let v = vbase::sample_vector(n_leaves, ordered);
    as_i32(v)
}

/// Sample a random tree with branch lengths via Phylo2Vec
/// @export
#[extendr]
fn sample_matrix(n_leaves: usize, ordered: bool) -> RMatrix<f64> {
    let matrix = mbase::sample_matrix(n_leaves, ordered);
    let shape = matrix.shape();
    RMatrix::new_matrix(shape[0], shape[1], |r, c| matrix[[r, c]])
}

/// Recover a rooted tree (in Newick format) from a Phylo2Vec vector
/// @export
#[extendr]
fn to_newick_from_vector(vector: Vec<i32>) -> String {
    let v_usize = as_usize(vector);
    vconvert::to_newick(&v_usize)
}

/// Recover a rooted tree (in Newick format) from a Phylo2Vec matrix
/// @export
#[extendr]
fn to_newick_from_matrix(matrix: RMatrix<f64>) -> String {
    let matrix = convert_from_rmatrix(&matrix).unwrap();
    mconvert::to_newick(&matrix.view())
}

/// Convert a newick string to a Phylo2Vec vector
/// @export
#[extendr]
fn to_vector(newick: &str) -> Vec<i32> {
    let v = vconvert::from_newick(newick);
    as_i32(v)
}

/// Convert a newick string to a Phylo2Vec vector
/// @export
#[extendr]
fn to_matrix(newick: &str) -> RMatrix<f64> {
    let matrix = mconvert::from_newick(newick);
    let shape = matrix.shape();
    RMatrix::new_matrix(shape[0], shape[1], |r, c| matrix[[r, c]])
}

/// Validate a Phylo2Vec vector
/// @export
#[extendr]
fn check_v(vector: Vec<i32>) {
    let v_usize = as_usize(vector);
    vbase::check_v(&v_usize);
}

/// Validate a Phylo2Vec vector
/// @export
#[extendr]
fn check_m(vector: RMatrix<f64>) {
    let matrix = convert_from_rmatrix(&vector).unwrap();

    mbase::check_m(&matrix.view());
}

/// Get the ancestry matrix of a Phylo2Vec vector
/// @export
#[extendr]
fn to_ancestry(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let ancestry = vconvert::to_ancestry(&v_usize);

    RMatrix::new_matrix(k, 3, |r, c| ancestry[r][c] as i32)
}

/// Convert an ancestry matrix to a Phylo2Vec vector
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

/// Get pairs from a Phylo2Vec vector
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

/// Convert a pairs matrix to a Phylo2Vec vector
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

/// Get the edge list of a Phylo2Vec vector
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

/// Convert an edge list to a Phylo2Vec vector
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

/// Add a leaf to a Phylo2Vec vector
/// @export
#[extendr]
fn add_leaf(vector: Vec<i32>, leaf: i32, branch: i32) -> Vec<i32> {
    let mut v_usize: Vec<usize> = as_usize(vector);
    let new_v = vops::add_leaf(&mut v_usize, leaf as usize, branch as usize);
    as_i32(new_v)
}

/// Remove a leaf from a Phylo2Vec vector
/// @export
#[extendr]
fn remove_leaf(vector: Vec<i32>, leaf: i32) -> Robj {
    let mut v_usize: Vec<usize> = as_usize(vector);
    let (new_v, branch) = vops::remove_leaf(&mut v_usize, leaf as usize);
    list!(v = as_i32(new_v), branch = branch as i32).into()
}

// Get the first recent common ancestor between two nodes in a Phylo2Vec tree
// node1 and node2 can be leaf nodes (0 to n_leaves) or internal nodes (n_leaves to 2*(n_leaves-1)).
// Similar to ape's `getMRCA` function in R (for leaf nodes)
// and ETE's `get_common_ancestor` in Python (for all nodes), but for Phylo2Vec vectors.

/// @export
#[extendr]
fn get_common_ancestor(vector: Vec<i32>, node1: i32, node2: i32) -> i32 {
    let v_usize: Vec<usize> = as_usize(vector);
    let common_ancestor = vops::get_common_ancestor(&v_usize, node1 as usize, node2 as usize);
    common_ancestor as i32
}

/// Produce an ordered version (i.e., birth-death process version)
/// of a Phylo2Vec vector using the Queue Shuffle algorithm.
///
/// Queue Shuffle ensures that the output tree is ordered,
/// while also ensuring a smooth path through the space of orderings
///
/// for more details, see https://doi.org/10.1093/gbe/evad213
///
/// @param vector A Phylo2Vec vector (i.e., a vector of integers)
/// @param shuffle_cherries If true, the algorithm will shuffle cherries (i.e., pairs of leaves)
/// @return A list with two elements:
/// - `v`: The ordered Phylo2Vec vector
/// - `mapping`: A mapping of the original labels to the new labels
/// @export
#[extendr]
fn queue_shuffle(vector: Vec<i32>, shuffle_cherries: bool) -> List {
    let v_usize: Vec<usize> = as_usize(vector);
    let (v_new, label_mapping) = vops::queue_shuffle(&v_usize, shuffle_cherries);
    list!(v = as_i32(v_new), mapping = as_i32(label_mapping))
}

/// Check if a newick string has branch lengths
/// @export
#[extendr]
fn has_branch_lengths(newick: &str) -> bool {
    newick::has_branch_lengths(newick)
}

/// Create an integer-taxon label mapping (label_mapping)
/// from a string-based newick (where leaves are strings)
/// and produce a mapped integer-based newick (where leaves are integers).
///
/// Note 1: this does not check for the validity of the Newick string.
///
/// Note 2: the parent nodes are removed from the output,
/// but the branch lengths/annotations are kept.
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

/// Get the topological cophenetic distance matrix of a Phylo2Vec vector
/// @export
#[extendr]
fn cophenetic_from_vector(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let coph_rs = vgraph::cophenetic_distances(&v_usize);
    let mut coph_r = RMatrix::new_matrix(k + 1, k + 1, |r, c| coph_rs[[r, c]] as i32);
    let dimnames = (0..=k).map(|x| x as i32).collect::<Vec<i32>>();
    coph_r.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    coph_r
}

/// Get the cophenetic distance matrix of a Phylo2Vec matrix
/// @export
#[extendr]
fn cophenetic_from_matrix(matrix: ArrayView2<f64>) -> RMatrix<f64> {
    let coph_rs = mgraph::cophenetic_distances(&matrix.view());
    let n_leaves = coph_rs.shape()[0];
    let mut coph_r = RMatrix::new_matrix(n_leaves, n_leaves, |r, c| coph_rs[[r, c]]);
    let dimnames = (0..n_leaves).map(|x| x as i32).collect::<Vec<i32>>();
    coph_r.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    coph_r
}

/// Get the precision matrix of a Phylo2Vec vector
/// @export
#[extendr]
fn pre_precision_from_vector(vector: Vec<i32>) -> RMatrix<f64> {
    let v_usize: Vec<usize> = as_usize(vector);
    let pre_precision_rs = vgraph::pre_precision(&v_usize);
    let n = pre_precision_rs.shape()[0];
    let mut preprecision_r = RMatrix::new_matrix(n, n, |r, c| pre_precision_rs[[r, c]]);
    let dimnames = (0..n).map(|x| x as i32).collect::<Vec<i32>>();
    preprecision_r.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    preprecision_r
}

/// Get the precision matrix of a Phylo2Vec matrix
/// @export
#[extendr]
fn pre_precision_from_matrix(matrix: RMatrix<f64>) -> RMatrix<f64> {
    let matrix_rs = convert_from_rmatrix(&matrix).unwrap();
    let pre_precision_rs = mgraph::pre_precision(&matrix_rs.view());
    let n = pre_precision_rs.shape()[0];
    let mut pre_precision_r = RMatrix::new_matrix(n, n, |r, c| pre_precision_rs[[r, c]]);
    let dimnames = (0..n).map(|x| x as i32).collect::<Vec<i32>>();
    pre_precision_r.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    pre_precision_r
}

/// Get the variance-covariance matrix of a Phylo2Vec vector
/// @export
#[extendr]
fn vcov_from_vector(vector: Vec<i32>) -> RMatrix<f64> {
    let v_usize: Vec<usize> = as_usize(vector);
    let vcv_rs = vgraph::vcv(&v_usize);
    let n_leaves = vcv_rs.shape()[0];
    let mut vcv_r = RMatrix::new_matrix(n_leaves, n_leaves, |r, c| vcv_rs[[r, c]]);
    let dimnames = (0..n_leaves).map(|x| x as i32).collect::<Vec<i32>>();
    vcv_r.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    vcv_r
}

/// Get the variance-covariance matrix of a Phylo2Vec matrix
/// @export
#[extendr]
fn vcov_from_matrix(matrix: RMatrix<f64>) -> RMatrix<f64> {
    let matrix_rs = convert_from_rmatrix(&matrix).unwrap();
    let vcv_rs = mgraph::vcv(&matrix_rs.view());
    let n_leaves = vcv_rs.shape()[0];
    let mut vcv_matrix = RMatrix::new_matrix(n_leaves, n_leaves, |r, c| vcv_rs[[r, c]]);
    let dimnames = (0..n_leaves).map(|x| x as i32).collect::<Vec<i32>>();
    vcv_matrix.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    vcv_matrix
}

/// Get the incidence matrix of a Phylo2Vec vector in dense format
/// @export
#[extendr]
fn incidence_dense(input_vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize = as_usize(input_vector);
    let dense_rs = vgraph::Incidence::new(&v_usize).to_dense();
    let n_leaves = dense_rs.shape()[0];
    let mut dense_r = RMatrix::new_matrix(n_leaves, n_leaves, |r, c| dense_rs[[r, c]] as i32);
    let dimnames = (0..n_leaves).map(|x| x as i32).collect::<Vec<i32>>();
    dense_r.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    dense_r
}

/// Get the incidence matrix of a Phylo2Vec vector in COUP format
/// @export
#[extendr]
fn incidence_coo(input_vector: Vec<i32>) -> RMatrix<i32> {
    let k = input_vector.len();
    let v_usize = as_usize(input_vector);
    let (data, rows, cols) = vgraph::Incidence::new(&v_usize).to_coo();
    RMatrix::new_matrix(4 * k, 3, |r, c| match c {
        0 => data[r] as i32,
        1 => rows[r] as i32,
        2 => cols[r] as i32,
        _ => unreachable!(),
    })
}

/// Get the incidence matrix of a Phylo2Vec vector in CSR format
/// @export
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

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod phylo2vec;
    fn add_leaf;
    fn apply_label_mapping;
    fn check_m;
    fn check_v;
    fn cophenetic_from_matrix;
    fn cophenetic_from_vector;
    fn create_label_mapping;
    fn from_ancestry;
    fn from_edges;
    fn from_pairs;
    fn get_common_ancestor;
    fn has_branch_lengths;
    fn incidence_coo;
    fn incidence_csr;
    fn incidence_dense;
    fn pre_precision_from_matrix;
    fn pre_precision_from_vector;
    fn queue_shuffle;
    fn remove_leaf;
    fn sample_matrix;
    fn sample_vector;
    fn to_ancestry;
    fn to_edges;
    fn to_newick_from_matrix;
    fn to_newick_from_vector;
    fn to_pairs;
    fn to_matrix;
    fn to_vector;
    fn vcov_from_matrix;
    fn vcov_from_vector;
}
