use crate::tree_vec::types::Ancestry;

mod newick_patterns;

pub use newick_patterns::NewickPatterns;

fn _get_cherries_recursive_inner(ancestry: &mut Ancestry, newick: &str, has_parents: bool) {
    let mut open_idx: usize = 0;

    for (i, ch) in newick.chars().enumerate() {
        if ch == '(' {
            open_idx = i + 1;
        } else if ch == ')' {
            let pairs: Vec<usize> = newick[open_idx..i]
                .split(',')
                .map(|x: &str| x.parse::<usize>().unwrap())
                .collect();
            let c1 = pairs[0];
            let c2 = pairs[1];
            let parent: usize;
            let new_newick: String;
            match has_parents {
                // For case that the newick string has parents
                true => {
                    parent = newick[i + 1..]
                        .split(',')
                        .next()
                        .unwrap_or("")
                        .split(')')
                        .next()
                        .unwrap_or("")
                        .parse::<usize>()
                        .unwrap();
                    new_newick = format!("{}{}", &newick[..open_idx - 1], &newick[i + 1..]);
                }
                // For case that the newick string does not have parents
                false => {
                    parent = std::cmp::max(c1, c2);
                    new_newick = newick.replace(
                        &newick[open_idx - 1..i + 1],
                        &std::cmp::min(c1, c2).to_string(),
                    );
                }
            }

            ancestry.push([c1, c2, parent]);

            return _get_cherries_recursive_inner(ancestry, &new_newick, has_parents);
        }
    }

    // Extract the children
}

pub fn get_cherries(newick: &str) -> Ancestry {
    let mut ancestry: Ancestry = Vec::new();
    _get_cherries_recursive_inner(&mut ancestry, &newick[..newick.len() - 1], true);
    ancestry
}

pub fn get_cherries_no_parents(newick: &str) -> Ancestry {
    let mut ancestry: Ancestry = Vec::new();
    _get_cherries_recursive_inner(&mut ancestry, &newick[..newick.len() - 1], false);
    ancestry
}

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

/// Remove parent labels from the Newick string
///
/// # Example
///
/// ```
/// use phylo2vec::tree_vec::ops::newick::remove_parent_labels;
///
/// let newick = "(((0,(3,5)6)8,2)9,(1,4)7)10;";
/// let result = remove_parent_labels(newick);
/// assert_eq!(result, "(((0,(3,5)),2),(1,4));");
/// ```
pub fn remove_parent_labels(newick: &str) -> String {
    let newick_patterns = NewickPatterns::new();
    return newick_patterns.parents.replace_all(newick, ")").to_string();
}

/// Check if the Newick string has parent labels
///
/// # Example
///
/// ```
/// use phylo2vec::tree_vec::ops::newick::has_parents;
///
/// let newick = "(((0,(3,5)6)8,2)9,(1,4)7)10;";
/// let result = has_parents(newick);
/// assert_eq!(result, true);
///
/// let newick_no_parents = "(((0,(3,5)),2),(1,4));";
/// let result_no_parents = has_parents(newick_no_parents);
/// assert_eq!(result_no_parents, false);
/// ```
pub fn has_parents(newick: &str) -> bool {
    let newick_patterns = NewickPatterns::new();
    return newick_patterns.parents.is_match(newick);
}

/// Find the number of leaves in the Newick string
///
/// # Example
///
/// ```
/// use phylo2vec::tree_vec::ops::newick::find_num_leaves;
///
/// let newick = "(((0,(3,5)6)8,2)9,(1,4)7)10;";
/// let result = find_num_leaves(newick);
/// assert_eq!(result, 6);
/// ```
pub fn find_num_leaves(newick: &str) -> usize {
    let newick_patterns = NewickPatterns::new();
    let result: Vec<usize> = newick_patterns
        .pairs
        .captures_iter(newick)
        .map(|caps| {
            let (_, [_, node]) = caps.extract();
            node.parse::<usize>().unwrap()
        })
        .collect();

    return result.len();
}

/// Build newick string from the ancestry matrix
pub fn build_newick(ancestry: &Ancestry) -> String {
    // Get the root node, which is the parent value of the last ancestry element
    let root = ancestry.last().unwrap()[2];

    // Build the Newick string starting from the root, and append a semicolon
    format!("{};", _build_newick_recursive_inner(root, ancestry))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_vec::ops::to_newick;
    use crate::utils::sample;
    use rstest::*;

    #[rstest]
    #[case("(((0,(3,5)6)8,2)9,(1,4)7)10;", "(((0,(3,5)),2),(1,4));")]
    #[case("(0,(1,(2,(3,(4,5)6)7)8)9)10;", "(0,(1,(2,(3,(4,5)))));")]
    #[case("((0,2)5,(1,3)4)6;", "((0,2),(1,3));")]
    fn test_remove_parent_labels(#[case] newick: &str, #[case] expected: &str) {
        let result = remove_parent_labels(&newick);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_has_parents(#[case] n_leaves: usize) {
        let v = sample(n_leaves, false);
        let newick = to_newick(&v);
        // Check if the newick string has parents
        let result = has_parents(&newick);
        assert_eq!(result, true);

        // Check if the newick string does not have parents
        let result_no_parents = has_parents(&remove_parent_labels(&newick));
        assert_eq!(result_no_parents, false);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_find_num_leaves(#[case] n_leaves: usize) {
        let v = sample(n_leaves, false);
        let newick = to_newick(&v);
        // Check if the newick string has parents
        let result = find_num_leaves(&newick);
        assert_eq!(result, n_leaves);
    }
}
