use pyo3::prelude::*;

pub use::phylo2vec::to_newick as _to_newick;
// pub use crate::core::to_vector as _to_vector;

/// This function takes a Python list and converts it to a Rust vector.
#[pyfunction]
fn to_newick(input_vector: Vec<usize>) -> PyResult<String> {
    // TODO: Take in numpy array instead of just simple list?
    let newick = _to_newick::to_newick(input_vector);
    Ok(newick)
}

#[pyfunction]
fn get_ancestry(input_vector: Vec<usize>) -> Vec<(usize, usize, usize)> {
    let ancestry: Vec<(usize, usize, usize)> = _to_newick::_get_ancestry(&input_vector);

    ancestry
}

#[pyfunction]
fn get_pairs(input_vector: Vec<usize>) -> Vec<(usize, usize)> {
    let pairs: Vec<(usize, usize)> = _to_newick::_get_pairs(&input_vector);
    pairs
}

#[pyfunction]
fn build_newick(input_ancestry: Vec<(usize, usize, usize)>) -> String {
    let newick_string: String = _to_newick::_build_newick(&input_ancestry);
    newick_string
}


/// This module is exposed to Python.
#[pymodule]
fn _phylo2vec_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(to_newick, m)?)?;
    m.add_function(wrap_pyfunction!(build_newick, m)?)?;
    m.add_function(wrap_pyfunction!(get_ancestry, m)?)?;
    m.add_function(wrap_pyfunction!(get_pairs, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    // use super::*;

    // #[test]
    // fn test_to_newick() {
    //     let v = vec![0, 0, 1];
    //     let newick = _to_newick::to_newick(&v);
    //     assert_eq!(newick, "((0,2)5,(1,3)4)6;");
    // }

    // #[test]
    // fn test_to_vector() {
    //     let newick: &str = "((0,2)5,(1,3)4)6;";
    //     let output_v: Vec<usize> = _to_vector::to_vector(&newick);
    //     assert_eq!(output_v, vec![0, 0, 1]);
    // }
}
