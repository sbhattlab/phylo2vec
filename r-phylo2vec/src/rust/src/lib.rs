use extendr_api::prelude::*;
use std::result::Result;

use phylo2vec::tree_vec::ops;
use phylo2vec::utils;

/// Sample a random tree via Phylo2Vec
/// @export
#[extendr]
fn sample(n_leaves: usize, ordered: bool) -> Vec<i32> {
    let v = utils::sample_vector(n_leaves, ordered);
    v.iter().map(|&x| x as i32).collect()
}

/// Recover a rooted tree (in Newick format) from a Phylo2Vec vector
/// @export
#[extendr]
fn to_newick_from_vector(input_integers: Vec<i32>) -> String {
    let input_vector: Vec<usize> = input_integers.iter().map(|&x| x as usize).collect();
    ops::to_newick_from_vector(&input_vector)
}

/// Recover a rooted tree (in Newick format) from a Phylo2Vec matrix
/// @export
#[extendr]
fn to_newick_from_matrix(input_integers: Robj) -> String {
    let matrix = convert_from_rmatrix(&input_integers).unwrap();
    ops::to_newick_from_matrix(&matrix)
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

/// Convert a newick string to a Phylo2Vec vector
/// @export
#[extendr]
fn to_vector(newick: &str) -> Vec<i32> {
    let v = ops::to_vector(newick);
    v.iter().map(|&x| x as i32).collect()
}

/// Validate a Phylo2Vec vector
/// @export
#[extendr]
fn check_v(input_integers: Vec<i32>) {
    let input_vector: Vec<usize> = input_integers.iter().map(|&x| x as usize).collect();
    utils::check_v(&input_vector);
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod phylo2vec;
    fn sample;
    fn to_newick_from_vector;
    fn to_newick_from_matrix;
    fn to_vector;
    fn check_v;
}
