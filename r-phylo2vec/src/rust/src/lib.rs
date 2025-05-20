use extendr_api::prelude::*;
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
fn convert_from_rmatrix(matrix: &Robj) -> Result<Vec<Vec<f32>>, &'static str> {
    let data = matrix.as_real_slice().ok_or("Expected numeric matrix")?;
    let dims = matrix.dim().ok_or("Matrix is missing dimensions")?;

    let (nrows, ncols) = (dims[0].inner() as usize, dims[1].inner() as usize);

    // Convert column-major to row-major Vec<Vec<f32>>
    Ok((0..nrows)
        .map(|row| {
            (0..ncols)
                .map(|col| data[col * nrows + row] as f32)
                .collect()
        })
        .collect())
}

/// Sample a random tree via Phylo2Vec
/// @export
#[extendr]
fn sample_vector(n_leaves: usize, ordered: bool) -> Vec<i32> {
    let v = utils::sample_vector(n_leaves, ordered);
    as_i32(v)
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
fn to_newick_from_matrix(matrix: Robj) -> String {
    let matrix = convert_from_rmatrix(&matrix).unwrap();
    ops::to_newick_from_matrix(&matrix)
}

/// Convert a newick string to a Phylo2Vec vector
/// @export
#[extendr]
fn to_vector(newick: &str) -> Vec<i32> {
    let v = ops::to_vector(newick);
    as_i32(v)
}

/// Validate a Phylo2Vec vector
/// @export
#[extendr]
fn check_v(vector: Vec<i32>) {
    let v_usize = as_usize(vector);
    utils::check_v(&v_usize);
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
fn cophenetic_distances(vector: Vec<i32>) -> RMatrix<i32> {
    let v_usize: Vec<usize> = as_usize(vector);
    let k = v_usize.len();
    let distances = ops::cophenetic_distances(&v_usize);
    let mut coph = RMatrix::new_matrix(k + 1, k + 1, |r, c| distances[r][c] as i32);
    let dimnames = (0..k + 1).map(|x| x as i32).collect::<Vec<i32>>();
    coph.set_dimnames(List::from_values(vec![dimnames.clone(), dimnames]));

    coph
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod phylo2vec;
    fn add_leaf;
    fn check_v;
    fn cophenetic_distances;
    fn from_ancestry;
    fn from_edges;
    fn from_pairs;
    fn remove_leaf;
    fn sample_vector;
    fn to_ancestry;
    fn to_edges;
    fn to_newick_from_matrix;
    fn to_newick_from_vector;
    fn to_pairs;
    fn to_vector;
}
