use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::*;

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

#[derive(FromPyObject)]
enum PyTree<'py> {
    Matrix(PyReadonlyArray2<'py, f64>),
    Vector(Vec<usize>),
}

#[pyfunction]
fn sample_vector(n_leaves: isize, ordered: bool) -> PyResult<Vec<usize>> {
    if n_leaves < 2 {
        Err(PyValueError::new_err("n_leaves must be at least 2"))
    } else {
        Ok(vbase::sample_vector(n_leaves as usize, ordered))
    }
}

#[pyfunction]
fn sample_matrix(
    py: Python<'_>,
    n_leaves: isize,
    ordered: bool,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    if n_leaves < 2 {
        Err(PyValueError::new_err("n_leaves must be at least 2"))
    } else {
        Ok(mbase::sample_matrix(n_leaves as usize, ordered).into_pyarray(py))
    }
}

#[pyfunction]
fn check_v(input_vector: Vec<usize>) -> PyResult<()> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        vbase::check_v(&input_vector);
    }));

    match result {
        Ok(_) => Ok(()),
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "Rust panic occurred, but couldn't extract message".to_string()
            };

            Err(PyAssertionError::new_err(msg))
        }
    }
}

#[pyfunction]
fn check_m(input_matrix: PyReadonlyArray2<f64>) -> PyResult<()> {
    let m = input_matrix.as_array();

    let result = catch_unwind(AssertUnwindSafe(|| {
        mbase::check_m(&m);
    }));

    match result {
        Ok(_) => Ok(()),
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "Rust panic occurred, but couldn't extract message".to_string()
            };

            Err(PyAssertionError::new_err(msg))
        }
    }
}

#[pyfunction]
fn to_newick(tree: PyTree) -> String {
    match tree {
        PyTree::Matrix(m) => mconvert::to_newick(&m.as_array()),
        PyTree::Vector(v) => vconvert::to_newick(&v),
    }
}

#[pyfunction]
fn to_vector(newick: &str) -> Vec<usize> {
    vconvert::from_newick(newick)
}

#[pyfunction]
fn to_matrix<'py>(py: Python<'py>, newick: &str) -> Bound<'py, PyArray2<f64>> {
    mconvert::from_newick(newick).into_pyarray(py)
}

#[pyfunction]
fn get_ancestry(input_vector: Vec<usize>) -> Vec<[usize; 3]> {
    vconvert::to_ancestry(&input_vector)
}

#[pyfunction]
fn from_ancestry(input_ancestry: Vec<[usize; 3]>) -> Vec<usize> {
    vconvert::from_ancestry(&input_ancestry)
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
fn cophenetic_distances<'py>(
    py: Python<'py>,
    tree: PyTree<'py>,
    unrooted: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match tree {
        PyTree::Matrix(m) => {
            Ok(mgraph::cophenetic_distances(&m.as_array(), unrooted).into_pyarray(py))
        }
        PyTree::Vector(v) => Ok(vgraph::cophenetic_distances(&v, unrooted).into_pyarray(py)),
    }
}

#[pyfunction]
fn pre_precision<'py>(py: Python<'py>, tree: PyTree<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match tree {
        PyTree::Matrix(m) => Ok(mgraph::pre_precision(&m.as_array()).into_pyarray(py)),
        PyTree::Vector(v) => Ok(vgraph::pre_precision(&v).into_pyarray(py)),
    }
}

#[pyfunction]
fn vcv<'py>(py: Python<'py>, tree: PyTree<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match tree {
        PyTree::Matrix(m) => Ok(mgraph::vcv(&m.as_array()).into_pyarray(py)),
        PyTree::Vector(v) => Ok(vgraph::vcv(&v).into_pyarray(py)),
    }
}

#[pyfunction]
fn incidence_dense(py: Python<'_>, input_vector: Vec<usize>) -> PyResult<Bound<'_, PyArray2<i8>>> {
    Ok(vgraph::Incidence::new(&input_vector)
        .to_dense()
        .into_pyarray(py))
}

#[pyfunction]
fn incidence_coo(input_vector: Vec<usize>) -> PyResult<(Vec<i8>, Vec<usize>, Vec<usize>)> {
    Ok(vgraph::Incidence::new(&input_vector).to_coo())
}

#[pyfunction]
fn incidence_csc(input_vector: Vec<usize>) -> PyResult<(Vec<i8>, Vec<usize>, Vec<usize>)> {
    Ok(vgraph::Incidence::new(&input_vector).to_csc())
}

#[pyfunction]
fn incidence_csr(input_vector: Vec<usize>) -> PyResult<(Vec<i8>, Vec<usize>, Vec<usize>)> {
    Ok(vgraph::Incidence::new(&input_vector).to_csr())
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
fn get_common_ancestor(tree: PyTree, node1: isize, node2: isize) -> PyResult<usize> {
    if node1 < 0 || node2 < 0 {
        return Err(PyValueError::new_err("nodes must be non-negative"));
    }
    let node1 = node1 as usize;
    let node2 = node2 as usize;
    let v = match tree {
        PyTree::Matrix(m) => {
            let (v, _) = mbase::parse_matrix(&m.as_array());
            v
        }
        PyTree::Vector(v) => v,
    };

    vops::get_common_ancestor(&v, node1, node2).map_err(PyValueError::new_err)
}

#[pyfunction]
fn get_node_depth(tree: PyTree, node: isize) -> PyResult<f64> {
    if node < 0 {
        return Err(PyValueError::new_err("node must be non-negative"));
    }
    let node = node as usize;
    let depth = match tree {
        PyTree::Matrix(m) => mops::get_node_depth(&m.as_array(), node),
        PyTree::Vector(v) => vops::get_node_depth(&v, node),
    };

    depth.map_err(PyValueError::new_err)
}

#[pyfunction]
fn get_node_depths(tree: PyTree) -> Vec<f64> {
    match tree {
        PyTree::Matrix(m) => mops::get_node_depths(&m.as_array()),
        PyTree::Vector(v) => vops::get_node_depths(&v),
    }
}

#[pyfunction]
fn queue_shuffle(v: Vec<usize>, shuffle_cherries: bool) -> (Vec<usize>, Vec<usize>) {
    vops::queue_shuffle(&v, shuffle_cherries)
}

#[pyfunction]
fn robinson_foulds(tree1: PyTree, tree2: PyTree, normalize: bool) -> f64 {
    let (v1, v2) = match (tree1, tree2) {
        (PyTree::Matrix(m1), PyTree::Matrix(m2)) => (
            mbase::parse_matrix(&m1.as_array()).0,
            mbase::parse_matrix(&m2.as_array()).0,
        ),
        (PyTree::Vector(v1), PyTree::Vector(v2)) => (v1, v2),
        (PyTree::Matrix(m), PyTree::Vector(v)) | (PyTree::Vector(v), PyTree::Matrix(m)) => {
            let v_from_m = mbase::parse_matrix(&m.as_array()).0;
            (v_from_m, v)
        }
    };

    vdist::robinson_foulds(&v1, &v2, normalize)
}

#[pyfunction]
fn sackin(v: Vec<usize>) -> usize {
    vbalance::sackin(&v)
}

#[pyfunction]
fn leaf_depth_variance(v: Vec<usize>) -> f64 {
    vbalance::leaf_depth_variance(&v)
}

#[pyfunction]
fn b2(v: Vec<usize>) -> f64 {
    vbalance::b2(&v)
}

// skipcq: RS-R1000
/// This module is exposed to Python.
/// The line below raises an issue in DeepSource stating that this function's cyclomatic complexity is higher than threshold
/// the analyzer does not understand that this is an API exposure function, hence the comment above to skip over this occurrence.
#[pymodule]
fn _phylo2vec_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(apply_label_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(b2, m)?)?;
    m.add_function(wrap_pyfunction!(build_newick, m)?)?;
    m.add_function(wrap_pyfunction!(check_m, m)?)?;
    m.add_function(wrap_pyfunction!(check_v, m)?)?;
    m.add_function(wrap_pyfunction!(cophenetic_distances, m)?)?;
    m.add_function(wrap_pyfunction!(create_label_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(find_num_leaves, m)?)?;
    m.add_function(wrap_pyfunction!(from_ancestry, m)?)?;
    m.add_function(wrap_pyfunction!(from_edges, m)?)?;
    m.add_function(wrap_pyfunction!(from_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(get_ancestry, m)?)?;
    m.add_function(wrap_pyfunction!(get_common_ancestor, m)?)?;
    m.add_function(wrap_pyfunction!(get_edges, m)?)?;
    m.add_function(wrap_pyfunction!(get_node_depth, m)?)?;
    m.add_function(wrap_pyfunction!(get_node_depths, m)?)?;
    m.add_function(wrap_pyfunction!(get_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(has_branch_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(has_parents, m)?)?;
    m.add_function(wrap_pyfunction!(incidence_coo, m)?)?;
    m.add_function(wrap_pyfunction!(incidence_csc, m)?)?;
    m.add_function(wrap_pyfunction!(incidence_csr, m)?)?;
    m.add_function(wrap_pyfunction!(incidence_dense, m)?)?;
    m.add_function(wrap_pyfunction!(leaf_depth_variance, m)?)?;
    m.add_function(wrap_pyfunction!(pre_precision, m)?)?;
    m.add_function(wrap_pyfunction!(queue_shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(remove_branch_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(remove_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(remove_parent_labels, m)?)?;
    m.add_function(wrap_pyfunction!(robinson_foulds, m)?)?;
    m.add_function(wrap_pyfunction!(sackin, m)?)?;
    m.add_function(wrap_pyfunction!(sample_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(sample_vector, m)?)?;
    m.add_function(wrap_pyfunction!(to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(to_newick, m)?)?;
    m.add_function(wrap_pyfunction!(to_vector, m)?)?;
    m.add_function(wrap_pyfunction!(vcv, m)?)?;
    // Metadata about the package bindings
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
