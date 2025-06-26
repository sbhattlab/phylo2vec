use extendr_api::prelude::*;
use ndarray::{Array2, Axis};
use std::collections::HashMap;
use std::result::Result;

use phylo2vec::tree_vec::ops;
use phylo2vec::utils;

fn as_usize(v: Vec<i32>) -> Vec<usize> {
    v.iter().map(|&x| x as usize).collect()
}

fn as_i32(v: Vec<usize>) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

// Convert R matrix to Rust Vec<Vec<f32>>
fn convert_from_rmatrix(matrix: &Robj) -> Result<Array2<f32>, &'static str> {
    let data = matrix.as_real_slice().ok_or("Expected numeric matrix")?;
    let dims = matrix.dim().ok_or("Matrix is missing dimensions")?;

    let (nrows, ncols) = (dims[0].inner() as usize, dims[1].inner() as usize);

    if data.len() != nrows * ncols {
        return Err("Matrix dimensions do not match data length");
    }
    // Create a 2D array with the specified dimensions
    let mut array = Array2::<f32>::zeros((nrows, ncols));
    for i in 0..nrows {
        for j in 0..ncols {
            array[[i, j]] = data[j * nrows + i] as f32; // Convert to f32
        }
    }

    Ok(array)

    // Convert column-major to row-major Vec<Vec<f32>>
    // Ok((0..nrows)
    //     .map(|row| {
    //         (0..ncols)
    //             .map(|col| data[col * nrows + row] as f32)
    //             .collect()
    //     })
    //     .collect())
}

/// Sample a random tree topology via Phylo2Vec
/// @export
#[extendr]
fn sample_vector(n_leaves: usize, ordered: bool) -> Vec<i32> {
    let v = utils::sample_vector(n_leaves, ordered);
    as_i32(v)
}

/// Sample a random tree with branch lengths via Phylo2Vec
/// @export
#[extendr]
fn sample_matrix(n_leaves: usize, ordered: bool) -> RMatrix<f64> {
    let matrix = utils::sample_matrix(n_leaves, ordered);
    let shape = matrix.shape();
    RMatrix::new_matrix(shape[0], shape[1], |r, c| matrix[[r, c]] as f64)
}

/// Recover a rooted tree (in Newick format) from a Phylo2Vec vector
/// @export
#[extendr]
fn to_newick_from_vector(vector: Vec<i32>) -> String {
    let v_usize = as_usize(vector);
    ops::to_newick_from_vector(&v_usize)
}

/// Recover a rooted tree (in Newick format) from a Phylo2Vec matrix
/// @export
#[extendr]
fn to_newick_from_matrix(matrix: RMatrix<f64>) -> String {
    let matrix = convert_from_rmatrix(&matrix).unwrap();
    ops::to_newick_from_matrix(&matrix.view())
}

/// Convert a newick string to a Phylo2Vec vector
/// @export
#[extendr]
fn to_vector(newick: &str) -> Vec<i32> {
    let v = ops::to_vector(newick);
    as_i32(v)
}

/// Convert a newick string to a Phylo2Vec vector
/// @export
#[extendr]
fn to_matrix(newick: &str) -> RMatrix<f64> {
    let matrix = ops::matrix::to_matrix(newick);
    let shape = matrix.shape();
    RMatrix::new_matrix(shape[0], shape[1], |r, c| matrix[[r, c]] as f64)
}

/// Validate a Phylo2Vec vector
/// @export
#[extendr]
fn check_v(vector: Vec<i32>) {
    let v_usize = as_usize(vector);
    utils::check_v(&v_usize);
}

/// Validate a Phylo2Vec vector
/// @export
#[extendr]
fn check_m(vector: RMatrix<f64>) {
    let matrix = convert_from_rmatrix(&vector).unwrap();

    utils::check_m(&matrix.view());
}

/// Get the ancestry matrix of a Phylo2Vec vector
/// @export
#[extendr]
fn to_ancestry(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let ancestry = ops::get_ancestry(&v_usize);

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

    let v = ops::from_ancestry(&ancestry_fixed);
    as_i32(v)
}

/// Get pairs from a Phylo2Vec vector
/// @export
#[extendr]
fn to_pairs(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let pairs = ops::get_pairs(&v_usize);
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
    let v = ops::from_pairs(&pairs_usize);
    as_i32(v)
}

/// Get the edge list of a Phylo2Vec vector
/// @export
#[extendr]
fn to_edges(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let edges = ops::get_edges(&v_usize);
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
    let v = ops::from_edges(&edges_usize);
    as_i32(v)
}

/// Add a leaf to a Phylo2Vec vector
/// @export
#[extendr]
fn add_leaf(vector: Vec<i32>, leaf: i32, branch: i32) -> Vec<i32> {
    let mut v_usize: Vec<usize> = as_usize(vector);
    let new_v = ops::add_leaf(&mut v_usize, leaf as usize, branch as usize);
    as_i32(new_v)
}

/// Remove a leaf from a Phylo2Vec vector
/// @export
#[extendr]
fn remove_leaf(vector: Vec<i32>, leaf: i32) -> Robj {
    let mut v_usize: Vec<usize> = as_usize(vector);
    let (new_v, branch) = ops::remove_leaf(&mut v_usize, leaf as usize);
    list!(v = as_i32(new_v), branch = branch as i32).into()
}

/// Get the topological cophenetic distance matrix of a Phylo2Vec vector
/// @export
#[extendr]
fn cophenetic_from_vector(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let coph_usize = ops::cophenetic_distances(&v_usize);
    let mut coph = RMatrix::new_matrix(k + 1, k + 1, |r, c| coph_usize[[r, c]] as i32);
    let dimnames = (0..k + 1).map(|x| x as i32).collect::<Vec<i32>>();
    coph.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    coph
}

/// Get the cophenetic distance matrix of a Phylo2Vec matrix
/// @export
#[extendr]
fn cophenetic_from_matrix(matrix: RMatrix<f64>) -> RMatrix<f64> {
    let matrix_f32 = convert_from_rmatrix(&matrix).unwrap();
    let k = matrix_f32.shape()[0];
    let coph_f32 = ops::matrix::cophenetic_distances_with_bls(&matrix_f32.view());
    let mut coph = RMatrix::new_matrix(k + 1, k + 1, |r, c| coph_f32[[r, c]] as f64);
    let dimnames = (0..k + 1).map(|x| x as i32).collect::<Vec<i32>>();
    coph.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    coph
}

/// Check if a newick string has branch lengths
/// @export
#[extendr]
fn has_branch_lengths(newick: &str) -> bool {
    ops::newick::has_branch_lengths(newick)
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
    let (nw_int, label_mapping) = ops::newick::create_label_mapping(newick);

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
    ops::newick::apply_label_mapping(newick, &label_mapping_hash).unwrap()
}

// Get the first recent common ancestor between two nodes in a Phylo2Vec tree
// node1 and node2 can be leaf nodes (0 to n_leaves) or internal nodes (n_leaves to 2*(n_leaves-1)).
// Similar to ape's `getMRCA` function in R (for leaf nodes)
// and ETE's `get_common_ancestor` in Python (for all nodes), but for Phylo2Vec vectors.

/// @export
#[extendr]
fn get_common_ancestor(vector: Vec<i32>, node1: i32, node2: i32) -> i32 {
    let v_usize: Vec<usize> = as_usize(vector);
    let common_ancestor =
        ops::vector::get_common_ancestor(&v_usize, node1 as usize, node2 as usize);
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
    let (v_new, label_mapping) = ops::vector::queue_shuffle(&v_usize, shuffle_cherries);
    list!(v = as_i32(v_new), mapping = as_i32(label_mapping))
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
}
