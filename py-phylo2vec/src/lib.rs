use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use phylo2vec::matrix::base as mbase;
use phylo2vec::matrix::convert as mconvert;
use phylo2vec::matrix::graph as mgraph;
use phylo2vec::newick;
use phylo2vec::vector::base as vbase;
use phylo2vec::vector::convert as vconvert;
use phylo2vec::vector::graph as vgraph;
use phylo2vec::vector::ops as vops;

#[pyfunction]
fn to_newick_from_vector(input_vector: Vec<usize>) -> PyResult<String> {
    let newick = vconvert::to_newick(&input_vector);
    Ok(newick)
}

#[pyfunction]
fn to_newick_from_matrix(input_matrix: PyReadonlyArray2<f32>) -> PyResult<String> {
    let arr = input_matrix.as_array();
    let newick = mconvert::to_newick(&arr);
    Ok(newick)
}

#[pyfunction]
fn to_vector(newick: &str) -> Vec<usize> {
    vconvert::from_newick(newick)
}

#[pyfunction]
fn to_matrix<'py>(py: Python<'py>, newick: &str) -> Bound<'py, PyArray2<f32>> {
    mconvert::from_newick(newick).into_pyarray(py)
}

#[pyfunction]
fn get_ancestry(input_vector: Vec<usize>) -> Vec<[usize; 3]> {
    let ancestry: Vec<[usize; 3]> = vconvert::to_ancestry(&input_vector);

    ancestry
}

#[pyfunction]
fn from_ancestry(input_ancestry: Vec<[usize; 3]>) -> Vec<usize> {
    let vector: Vec<usize> = vconvert::from_ancestry(&input_ancestry);
    vector
}

#[pyfunction]
fn get_pairs(input_vector: Vec<usize>) -> Vec<(usize, usize)> {
    vconvert::to_pairs(&input_vector)
}

#[pyfunction]
fn from_pairs(input_pairs: Vec<(usize, usize)>) -> Vec<usize> {
    vconvert::from_pairs(&input_pairs)
}

#[pyfunction]
fn get_edges(input_vector: Vec<usize>) -> Vec<(usize, usize)> {
    vconvert::to_edges(&input_vector)
}

#[pyfunction]
fn from_edges(input_edges: Vec<(usize, usize)>) -> Vec<usize> {
    vconvert::from_edges(&input_edges)
}

#[pyfunction]
fn build_newick(input_pairs: Vec<(usize, usize)>) -> String {
    let newick_string: String = vconvert::build_newick(&input_pairs);
    newick_string
}

#[pyfunction]
fn cophenetic_distances(py: Python<'_>, input_vector: Vec<usize>) -> Bound<'_, PyArray2<f32>> {
    vgraph::cophenetic_distances(&input_vector).into_pyarray(py)
}

#[pyfunction]
fn cophenetic_distances_with_bls<'py>(
    py: Python<'py>,
    input_matrix: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let m = input_matrix.as_array();
    mgraph::cophenetic_distances(&m).into_pyarray(py)
}

#[pyfunction]
fn sample_vector(n_leaves: usize, ordered: bool) -> Vec<usize> {
    vbase::sample_vector(n_leaves, ordered)
}

#[pyfunction]
fn sample_matrix(py: Python<'_>, n_leaves: usize, ordered: bool) -> Bound<'_, PyArray2<f32>> {
    mbase::sample_matrix(n_leaves, ordered).into_pyarray(py)
}

#[pyfunction]
fn check_v(input_vector: Vec<usize>) {
    vbase::check_v(&input_vector);
}

#[pyfunction]
fn check_m(input_matrix: PyReadonlyArray2<f32>) {
    let m = input_matrix.as_array();
    mbase::check_m(&m);
}

#[pyfunction]
fn find_num_leaves(newick: &str) -> usize {
    newick::find_num_leaves(newick)
}

#[pyfunction]
fn has_branch_lengths(newick: &str) -> bool {
    newick::has_branch_lengths(newick)
}

#[pyfunction]
fn has_parents(newick: &str) -> bool {
    newick::has_parents(newick)
}

#[pyfunction]
fn remove_branch_lengths(newick: &str) -> PyResult<String> {
    Ok(newick::remove_branch_lengths(newick))
}

#[pyfunction]
fn remove_parent_labels(newick: &str) -> PyResult<String> {
    Ok(newick::remove_parent_labels(newick))
}

#[pyfunction]
fn add_leaf(mut input_vector: Vec<usize>, leaf: usize, branch: usize) -> Vec<usize> {
    vops::add_leaf(&mut input_vector, leaf, branch)
}

#[pyfunction]
fn remove_leaf(mut input_vector: Vec<usize>, leaf: usize) -> (Vec<usize>, usize) {
    vops::remove_leaf(&mut input_vector, leaf)
}

#[pyfunction]
fn apply_label_mapping(newick: String, label_mapping: HashMap<usize, String>) -> PyResult<String> {
    let result = newick::apply_label_mapping(&newick, &label_mapping);

    // Map the potential NewickError to a PyValueErr
    // https://pyo3.rs/v0.22.3/function/error-handling#foreign-rust-error-types
    let newick_int = result.map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Label mapping failed: {}", e))
    })?;

    Ok(newick_int)
}

#[pyfunction]
fn create_label_mapping(newick: String) -> PyResult<(String, HashMap<usize, String>)> {
    Ok(newick::create_label_mapping(&newick))
}

#[pyfunction]
fn get_common_ancestor(v: Vec<usize>, node1: usize, node2: usize) -> usize {
    vops::get_common_ancestor(&v, node1, node2)
}

#[pyfunction]
fn queue_shuffle(v: Vec<usize>, shuffle_cherries: bool) -> (Vec<usize>, Vec<usize>) {
    vops::queue_shuffle(&v, shuffle_cherries)
}

// skipcq: RS-R1000
/// This module is exposed to Python.
/// The line below raises an issue in DeepSource stating that this function's cyclomatic complexity is higher than threshold
/// the analyzer does not understand that this is an API exposure function, hence the comment above to skip over this occurrence.
#[pymodule]
fn _phylo2vec_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(apply_label_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(build_newick, m)?)?;
    m.add_function(wrap_pyfunction!(check_m, m)?)?;
    m.add_function(wrap_pyfunction!(check_v, m)?)?;
    m.add_function(wrap_pyfunction!(cophenetic_distances, m)?)?;
    m.add_function(wrap_pyfunction!(cophenetic_distances_with_bls, m)?)?;
    m.add_function(wrap_pyfunction!(create_label_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(find_num_leaves, m)?)?;
    m.add_function(wrap_pyfunction!(from_ancestry, m)?)?;
    m.add_function(wrap_pyfunction!(from_edges, m)?)?;
    m.add_function(wrap_pyfunction!(from_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(get_ancestry, m)?)?;
    m.add_function(wrap_pyfunction!(get_common_ancestor, m)?)?;
    m.add_function(wrap_pyfunction!(get_edges, m)?)?;
    m.add_function(wrap_pyfunction!(get_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(has_branch_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(has_parents, m)?)?;
    m.add_function(wrap_pyfunction!(queue_shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(remove_branch_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(remove_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(remove_parent_labels, m)?)?;
    m.add_function(wrap_pyfunction!(sample_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(sample_vector, m)?)?;
    m.add_function(wrap_pyfunction!(to_newick_from_vector, m)?)?;
    m.add_function(wrap_pyfunction!(to_newick_from_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(to_vector, m)?)?;
    // Metadata about the package bindings
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
