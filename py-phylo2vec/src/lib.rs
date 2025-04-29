use pyo3::prelude::*;

use phylo2vec::tree_vec::ops;
use phylo2vec::utils;

#[pyfunction]
fn to_newick_from_vector(input_vector: Vec<usize>) -> PyResult<String> {
    let newick = ops::to_newick_from_vector(&input_vector);
    Ok(newick)
}

#[pyfunction]
fn to_newick_from_matrix(input_matrix: Vec<Vec<f32>>) -> PyResult<String> {
    let newick = ops::to_newick_from_matrix(&input_matrix);
    Ok(newick)
}

#[pyfunction]
fn to_vector(newick: &str) -> Vec<usize> {
    ops::to_vector(newick)
}

#[pyfunction]
fn to_matrix(newick: &str) -> Vec<Vec<f32>> {
    ops::matrix::to_matrix(newick)
}

#[pyfunction]
fn get_ancestry(input_vector: Vec<usize>) -> Vec<[usize; 3]> {
    let ancestry: Vec<[usize; 3]> = ops::get_ancestry(&input_vector);

    ancestry
}

#[pyfunction]
fn build_newick(input_ancestry: Vec<[usize; 3]>) -> String {
    let newick_string: String = ops::newick::build_newick(&input_ancestry);
    newick_string
}

#[pyfunction]
fn sample_vector(n_leaves: usize, ordered: bool) -> Vec<usize> {
    utils::sample_vector(n_leaves, ordered)
}

#[pyfunction]
fn cophenetic_distances(input_vector: Vec<usize>, unrooted: bool) -> Vec<Vec<usize>> {
    ops::vector::cophenetic_distances(&input_vector, unrooted)
}

#[pyfunction]
fn sample_matrix(n_leaves: usize, ordered: bool) -> Vec<Vec<f32>> {
    utils::sample_matrix(n_leaves, ordered)
}

#[pyfunction]
fn check_v(input_vector: Vec<usize>) {
    utils::check_v(&input_vector);
}

#[pyfunction]
fn add_leaf(mut input_vector: Vec<usize>, leaf: usize, branch: usize) -> Vec<usize> {
    ops::add_leaf(&mut input_vector, leaf, branch)
}

#[pyfunction]
fn remove_leaf(mut input_vector: Vec<usize>, leaf: usize) -> (Vec<usize>, usize) {
    ops::remove_leaf(&mut input_vector, leaf)
}

/// This module is exposed to Python.
/// The line below raises an issue in DeepSource stating that this function's cyclomatic complexity is higher than threshold
/// the analyzer does not understand that this is an API exposure function, hence the comment above to skip over this occurrence.
#[pymodule]
fn _phylo2vec_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(to_newick_from_vector, m)?)?;
    m.add_function(wrap_pyfunction!(to_vector, m)?)?;
    m.add_function(wrap_pyfunction!(to_newick_from_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(build_newick, m)?)?;
    m.add_function(wrap_pyfunction!(get_ancestry, m)?)?;
    m.add_function(wrap_pyfunction!(sample_vector, m)?)?;
    m.add_function(wrap_pyfunction!(sample_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(cophenetic_distances, m)?)?;
    m.add_function(wrap_pyfunction!(check_v, m)?)?;
    m.add_function(wrap_pyfunction!(add_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(remove_leaf, m)?)?;
    // Metadata about the package bindings
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
