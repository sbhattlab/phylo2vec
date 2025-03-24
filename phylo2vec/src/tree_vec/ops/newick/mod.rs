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
}

fn _get_cherries_recursive_inner_with_bls(
    ancestry: &mut Ancestry,
    bls: &mut Vec<[f32; 2]>,
    newick: &str,
    newick_has_parents: bool,
) {
    let mut open_idx: usize = 0;

    for (i, ch) in newick.chars().enumerate() {
        if ch == '(' {
            open_idx = i + 1;
        } else if ch == ')' {
            let parts: Vec<&str> = newick[open_idx..i].split(',').collect();
            let part = parts[0];
            let c1: usize;
            let c2;
            let mut bl1 = 0.0;
            let mut bl2 = 0.0;
            if part.contains(':') {
                // have to account for base case in which newick string is just a leaf
                let (child1_str, bl1_str) = part.split_once(':').unwrap();
                let (child2_str, bl2_str) = parts[1].split_once(':').unwrap();

                // Parse the children (c1, c2)
                c1 = child1_str.parse::<usize>().unwrap();
                c2 = child2_str.parse::<usize>().unwrap();

                // Parse the branch lengths (bl1, bl2)
                bl1 = bl1_str.parse::<f32>().unwrap_or(0.0);
                bl2 = bl2_str.parse::<f32>().unwrap_or(0.0);
            } else {
                c1 = parts[0].parse::<usize>().unwrap();
                c2 = parts[1].parse::<usize>().unwrap();
            }

            // The parent node (if present)
            let parent: usize;
            // let _blp: f32;
            let new_newick: String;

            if newick_has_parents {
                let parent_pair = newick[i + 1..]
                    .split(',')
                    .next()
                    .unwrap_or("")
                    .split(')')
                    .next()
                    .unwrap_or("")
                    .to_string();

                // Attempt to split by ":" to extract the branch length if it exists
                if parent_pair.is_empty() {
                    continue;
                } else {
                    let (parent_str, blp_str) = parent_pair.split_once(':').unwrap();
                    eprint!("parent_str: {}, blp_str: {}", parent_str, blp_str);
                    parent = match parent_str.parse::<usize>() {
                        Ok(parent_value) => parent_value, // Successfully parsed the parent node
                        Err(_) => std::cmp::max(c1, c2),  // Fallback value if parsing fails
                    };
                }
                new_newick = format!("{}{}", &newick[..open_idx - 1], &newick[i + 1..]);
            }
            // If the newick string does not have parents
            else {
                parent = std::cmp::max(c1, c2);
                new_newick = newick.replace(
                    &newick[open_idx - 1..=i],
                    &std::cmp::min(c1, c2).to_string(),
                );
            }

            // Append to ancestry (nodes)
            ancestry.push([c1, c2, parent]);

            // Append the branch lengths
            bls.push([bl1, bl2]);

            // Recursively process the next part of the newick string
            return _get_cherries_recursive_inner_with_bls(
                ancestry,
                bls,
                &new_newick,
                newick_has_parents,
            );
        }
    }
}

pub fn get_cherries(newick: &str) -> Ancestry {
    if newick.is_empty() {
        return Vec::new(); // Return empty ancestry and branch length vectors
    }
    let mut ancestry: Ancestry = Vec::new();
    _get_cherries_recursive_inner(&mut ancestry, &newick[..newick.len() - 1], true);
    ancestry
}

pub fn get_cherries_with_bls(newick: &str) -> (Ancestry, Vec<[f32; 2]>) {
    if newick.is_empty() {
        return (Vec::new(), Vec::new()); // Return empty ancestry and branch length vectors
    }
    let mut ancestry: Ancestry = Vec::new();
    let mut bls: Vec<[f32; 2]> = Vec::new();
    _get_cherries_recursive_inner_with_bls(
        &mut ancestry,
        &mut bls,
        &newick[..newick.len() - 1],
        true,
    );
    (ancestry, bls)
}

pub fn get_cherries_no_parents(newick: &str) -> Ancestry {
    if newick.is_empty() {
        return Vec::new(); // Return empty ancestry and branch length vectors
    }
    let mut ancestry: Ancestry = Vec::new();
    _get_cherries_recursive_inner(&mut ancestry, &newick[..newick.len() - 1], false);
    ancestry
}

pub fn get_cherries_no_parents_with_bls(newick: &str) -> (Ancestry, Vec<[f32; 2]>) {
    if newick.is_empty() {
        return (Vec::new(), Vec::new()); // Return empty ancestry and branch length vectors
    }
    let mut ancestry: Ancestry = Vec::new();
    let mut bls: Vec<[f32; 2]> = Vec::new();
    _get_cherries_recursive_inner_with_bls(
        &mut ancestry,
        &mut bls,
        &newick[..newick.len() - 1],
        false,
    );
    (ancestry, bls)
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
    use crate::utils::sample_vector;
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
        let v = sample_vector(n_leaves, false);
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
        let v = sample_vector(n_leaves, false);
        let newick = to_newick(&v);
        // Check if the newick string has parents
        let result = find_num_leaves(&newick);
        assert_eq!(result, n_leaves);
    }

    #[rstest]
    #[case("((1:0.5,2:0.7)1:0.9,3:0.8)2:0.8;", vec![[1, 2, 1], [1, 3, 2]], vec![[0.5, 0.7], [0.9, 0.8]])]
    #[case("(1:0.5,2:0.7);", vec![[1, 2, 2]], vec![[0.5, 0.7]] )]
    fn test_get_cherries_with_bls(
        #[case] newick: &str,
        #[case] expected_ancestry: Vec<[usize; 3]>,
        #[case] expected_bls: Vec<[f32; 2]>,
    ) {
        let ancestry: Ancestry;
        let bls: Vec<[f32; 2]>;
        if has_parents(newick) {
            (ancestry, bls) = get_cherries_with_bls(newick);
        } else {
            (ancestry, bls) = get_cherries_no_parents_with_bls(newick);
        }

        // Verify the ancestry
        assert_eq!(ancestry, expected_ancestry); // Ensure ancestry matches the expected

        // Verify the branch lengths
        assert_eq!(bls.len(), expected_bls.len()); // Ensure the number of branch lengths is correct
        assert_eq!(bls, expected_bls); // Ensure branch lengths match the expected
    }
}
