use pyo3::prelude::*;

use phylo2vec::tree_vec::ops;
use phylo2vec::utils;

/// This function takes a Python list and converts it to a Rust vector.
#[pyfunction]
fn to_newick(input_vector: Vec<usize>) -> PyResult<String> {
    let newick = ops::to_newick(&input_vector);
    Ok(newick)
}

#[pyfunction]
fn to_vector(newick: &str) -> Vec<usize> {
    let v = ops::to_vector(&newick);
    v
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
fn sample(n_leaves: usize, ordered: bool) -> Vec<usize> {
    let v = utils::sample(n_leaves, ordered);
    v
}

#[pyfunction]
fn check_v(input_vector: Vec<usize>) {
    utils::check_v(&input_vector);
}

/// This module is exposed to Python.
#[pymodule]
fn _phylo2vec_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(to_newick, m)?)?;
    m.add_function(wrap_pyfunction!(to_vector, m)?)?;
    m.add_function(wrap_pyfunction!(build_newick, m)?)?;
    m.add_function(wrap_pyfunction!(get_ancestry, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(check_v, m)?)?;
    // Metadata about the package bindings
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
