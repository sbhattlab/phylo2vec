use crate::tree_vec::types::Ancestry;

// The recursive function that builds the Newick string
fn _build_newick_recursive_inner(p: usize, ancestry: &Ancestry) -> String {
    let leaf_max = ancestry.len();

    // Extract the children (c1, c2) and ignore the parent from the ancestry tuple
    let [c1, c2, _] = ancestry[p - leaf_max - 1];

    // Recursive calls for left and right children, checking if they are leaves or internal nodes
    let left = if c1 > leaf_max {
        _build_newick_recursive_inner(c1, ancestry)
    } else {
        c1.to_string() // It's a leaf node, just convert to string
    };

    let right = if c2 > leaf_max {
        _build_newick_recursive_inner(c2, ancestry)
    } else {
        c2.to_string() // It's a leaf node, just convert to string
    };

    // Create the Newick string in the form (left, right)p
    format!("({},{}){}", left, right, p)
}

/// Build newick string from the ancestry matrix
pub fn build_newick(ancestry: &Ancestry) -> String {
    // Get the root node, which is the parent value of the last ancestry element
    let root = ancestry.last().unwrap()[2];

    // Build the Newick string starting from the root, and append a semicolon
    format!("{};", _build_newick_recursive_inner(root, ancestry))
}
