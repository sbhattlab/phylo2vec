use pyo3::prelude::*;

use phylo2vec::tree_vec::ops;
use phylo2vec::utils;

/// This function takes a Python list and converts it to a Rust vector.
#[pyfunction]
fn to_newick(input_vector: Vec<usize>) -> PyResult<String> {
    // TODO: Take in numpy array instead of just simple list?
    let newick = ops::to_newick(&input_vector);
    Ok(newick)
}

#[pyfunction]
fn get_ancestry(input_vector: Vec<usize>) -> Vec<[usize; 3]> {
    let ancestry: Vec<[usize; 3]> = ops::get_ancestry(&input_vector);

    ancestry
}

#[pyfunction]
fn get_pairs(input_vector: Vec<usize>) -> Vec<(usize, usize)> {
    let pairs: Vec<(usize, usize)> = ops::get_pairs(&input_vector);
    pairs
}

#[pyfunction]
fn get_pairs_avl(input_vector: Vec<usize>) -> Vec<(usize, usize)> {
    let pairs: Vec<(usize, usize)> = ops::get_pairs_avl(&input_vector);
    pairs
}

#[pyfunction]
fn build_newick(input_ancestry: Vec<[usize; 3]>) -> String {
    let newick_string: String = ops::vector::build_newick(&input_ancestry);
    newick_string
}

#[pyfunction]
#[pyo3(signature = (n_leaves, ordered=false, /), text_signature = "(n_leaves, ordered=False, /)")]
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
    m.add_function(wrap_pyfunction!(build_newick, m)?)?;
    m.add_function(wrap_pyfunction!(get_ancestry, m)?)?;
    m.add_function(wrap_pyfunction!(get_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(get_pairs_avl, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(check_v, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_newick() {
        let v = vec![0, 0, 1];
        let newick = ops::to_newick(&v);
        assert_eq!(newick, "((0,2)5,(1,3)4)6;");
    }
}
