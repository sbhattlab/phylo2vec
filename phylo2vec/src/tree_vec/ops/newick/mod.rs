use std::collections::HashMap;
use thiserror;

use crate::tree_vec::types::{Ancestry, Pairs};

mod newick_patterns;

pub use newick_patterns::NewickPatterns;

#[derive(Debug, thiserror::Error)]
pub enum NewickError {
    // For problematic int parsing in the Newick string
    #[error("ParseIntError: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),
    // For problematic float parsing in the Newick string
    #[error("ParseFloatError: {0}")]
    ParseFloatError(#[from] std::num::ParseFloatError),
    // For problematic stack popping in get_cherries
    #[error("Stack underflow error encountered")]
    StackUnderflow,
}

fn node_substr(s: &str, start: usize) -> (&str, usize) {
    let substr: &str = &s[start..];
    let mut end: usize = start;

    // Find the next comma, closing parenthesis, or semicolon
    for (i, c) in substr.char_indices() {
        if c == ',' || c == ')' || c == ';' {
            end = start + i;
            break;
        }
    }

    let node = &s[start..end];

    (node, end)
}

/// Extract the cherries from a Newick string with branch lengths
///
/// The cherries are processed in order of appearance in the string,
/// going as deep as possible in the tree structure
///
/// # Arguments
///
/// * `newick` - A string representing a phylogenetic tree in Newick format.
///   Leaves are noted as integers (0, 1, 2, ...) according to
///   the Phylo2Vec convention.
///
/// # Returns
///
/// A vector of triplets representing the cherries in the tree.
/// Throws a `NewickError` if the Newick string is invalid.
///
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::newick::get_cherries;
///
/// let newick = "((0,2)5,(1,3)4)6;";
///
/// let cherries = get_cherries(newick).expect("Oops");
///
/// assert_eq!(cherries, vec![[0, 2, 5], [1, 3, 4], [5, 4, 6]]);
/// ```
///
pub fn get_cherries(newick: &str) -> Result<Ancestry, NewickError> {
    if newick.is_empty() {
        return Ok(Vec::new());
    }
    let mut ancestry: Ancestry = Vec::new();
    let mut stack: Vec<usize> = Vec::new();

    let newick_bytes = newick.as_bytes();

    let mut i: usize = 0;
    while i < newick.len() {
        let c: char = newick_bytes[i] as char;

        if c == ')' {
            // Pop the children nodes from the stack
            let c2: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;
            let c1: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;

            // Get the parent node after ")"
            let (p, end) = node_substr(newick, i + 1);

            if p.is_empty() {
                // Case 1: No parent node
                let mut c_ordered = [c1, c2];
                c_ordered.sort();

                // Add the triplet (c1, c2, max(c1, c2))
                ancestry.push([c1, c2, c_ordered[1]]);
                // Push min(c1, c2) to the stack
                // to represent this internal node going forward
                stack.push(c_ordered[0]);
            } else {
                // Case 2: Parent node present

                // Add the triplet (c1, c2, p)
                let p_int = p.parse::<usize>().map_err(NewickError::ParseIntError)?;
                ancestry.push([c1, c2, p_int]);
                // Push the parent node to the stack
                stack.push(p_int);
                // Advance the index to the end of the parent node
                i = end - 1;
            }
        } else if c.is_ascii_digit() {
            // Get the next node and push it to the stack
            let (node, end) = node_substr(newick, i);
            stack.push(node.parse::<usize>().map_err(NewickError::ParseIntError)?);
            // Advance the index to the end of the node
            i = end - 1;
        }

        i += 1;
    }

    Ok(ancestry)
}

/// Extract the cherries from a Newick string with branch lengths
///
/// # Arguments
///
/// * `newick` - A string representing a phylogenetic tree in Newick format.
///   Leaves are noted as integers (0, 1, 2, ...) according to
///   the Phylo2Vec convention.
///
/// # Returns
///
/// A vector of triplets representing the cherries in the tree.
/// Throws a `NewickError` if the Newick string is invalid.
///
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::newick::get_cherries_with_bls;
///
/// // Without explicit parent nodes
/// let newick = "((0:0.1,2:0.2):0.3,(1:0.5,3:0.7):0.4);";
///
/// let (cherries, bls) = get_cherries_with_bls(newick).expect("Oops");
///
/// assert_eq!(cherries, vec![[0, 2, 2], [1, 3, 3], [0, 1, 1]]);
/// assert_eq!(bls, vec![[0.1, 0.2], [0.5, 0.7], [0.3, 0.4]]);
/// ```
///
pub fn get_cherries_with_bls(newick: &str) -> Result<(Ancestry, Vec<[f32; 2]>), NewickError> {
    if newick.is_empty() {
        return Ok((Vec::new(), Vec::new())); // Return empty ancestry and branch length vectors
    }
    let mut ancestry: Ancestry = Vec::new();
    let mut bls: Vec<[f32; 2]> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut bl_stack: Vec<f32> = Vec::new();

    let mut i: usize = 0;

    let newick_bytes = newick.as_bytes();

    while i < newick.len() {
        let c: char = newick_bytes[i] as char;

        if c == ')' {
            // Pop the children nodes from the stack
            let c2: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;
            let c1: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;

            // Pop the BLs from the BL stack
            let bl2: f32 = bl_stack.pop().ok_or(NewickError::StackUnderflow)?;
            let bl1: f32 = bl_stack.pop().ok_or(NewickError::StackUnderflow)?;
            bls.push([bl1, bl2]);

            let (annotation, end) = node_substr(newick, i + 1);
            i = end - 1;

            match annotation.split_once(':') {
                Some(("", blp)) => {
                    // Case 1: No parent node
                    let mut c_ordered = [c1, c2];
                    c_ordered.sort();

                    // Add the triplet (c1, c2, max(c1, c2))
                    ancestry.push([c1, c2, c_ordered[1]]);
                    // Push min(c1, c2) to the stack
                    // to represent this internal node going forward
                    stack.push(c_ordered[0]);
                    // Push the parent BL to the BL stack
                    bl_stack.push(blp.parse::<f32>().map_err(NewickError::ParseFloatError)?);
                }
                Some((p, blp)) => {
                    // Case 2: Parent node present

                    // Add the triplet (c1, c2, p)
                    let p_int = p.parse::<usize>().map_err(NewickError::ParseIntError)?;
                    ancestry.push([c1, c2, p_int]);
                    // Push the parent node to the stack
                    stack.push(p_int);
                    // Push the parent BL to the BL stack
                    bl_stack.push(blp.parse::<f32>().map_err(NewickError::ParseFloatError)?);
                }
                None => {
                    if end == newick.len() - 1 {
                        // Case 3: Reached the root
                        if annotation.is_empty() {
                            // Case 3.1: No parent node --> store the max leaf
                            let mut c_ordered = [c1, c2];
                            c_ordered.sort();

                            ancestry.push([c1, c2, c_ordered[1]]);
                        } else {
                            // Case 3.2: Parent node present
                            // Convert to int and push to ancestry
                            let root: usize = annotation
                                .parse::<usize>()
                                .map_err(NewickError::ParseIntError)?;
                            ancestry.push([c1, c2, root]);
                        }
                        // We reached the root, so we can break
                        break;
                    } else {
                        // Case 4: Unknown (missing annotation?)
                        panic!("Missing annotation in the Newick string");
                    }
                }
            }
        } else if c.is_ascii_digit() {
            // Get the next annotated node
            let (annotated_node, end) = node_substr(newick, i);
            i = end - 1;

            // Split the node from its branch length
            let (node, bln) = annotated_node.split_once(':').unwrap();

            stack.push(node.parse::<usize>().map_err(NewickError::ParseIntError)?);
            bl_stack.push(bln.parse::<f32>().map_err(NewickError::ParseFloatError)?);
        }

        i += 1;
    }

    Ok((ancestry, bls))
}

/// Prepare the cache for building the Newick string
fn prepare_cache(pairs: &Pairs) -> Vec<String> {
    let num_leaves = pairs.len() + 1;

    let mut cache: Vec<String> = vec![String::new(); num_leaves];

    // c1 will always be preceded by a left paren: (c1,c2)p
    // So we add a left paren to the cache to avoid insert operations
    for &(c1, _) in pairs.iter() {
        cache[c1].push('(');
    }

    // Add all leaf labels to the cache
    for (i, s) in cache.iter_mut().enumerate() {
        s.push_str(&i.to_string());
    }

    cache
}

/// Build newick string from a vector of pairs
pub fn build_newick(pairs: &Pairs) -> String {
    let num_leaves = pairs.len() + 1;

    let mut cache: Vec<String> = prepare_cache(pairs);

    for (i, &(c1, c2)) in pairs.iter().enumerate() {
        // std::mem::take helps with efficient swapping of values like std::move in C++
        let s2 = std::mem::take(&mut cache[c2]);

        // Parent node (not needed in theory, but left for legacy reasons)
        let sp = (num_leaves + i).to_string();

        // sub-newick structure: (c1,c2)p
        cache[c1].push(',');
        cache[c1].push_str(&s2);
        cache[c1].push(')');
        cache[c1].push_str(&sp);
    }

    format!("{};", cache[0])
}

/// Build newick string from the ancestry matrix and branch lengths
pub fn build_newick_with_bls(pairs: &Pairs, branch_lengths: &[[f32; 2]]) -> String {
    let num_leaves = pairs.len() + 1;

    let mut cache = prepare_cache(pairs);

    for (i, (&(c1, c2), &[bl1, bl2])) in pairs.iter().zip(branch_lengths.iter()).enumerate() {
        let s2 = std::mem::take(&mut cache[c2]);
        let sp = (num_leaves + i).to_string();
        let sb1 = bl1.to_string();
        let sb2 = bl2.to_string();

        cache[c1].push(':');
        cache[c1].push_str(&sb1);
        cache[c1].push(',');
        cache[c1].push_str(&s2);
        cache[c1].push(':');
        cache[c1].push_str(&sb2);
        cache[c1].push(')');
        cache[c1].push_str(&sp);
    }

    format!("{};", cache[0])
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
    newick_patterns.parents.replace_all(newick, ")").to_string()
}

/// Remove branch length annotations from the Newick string
///
/// # Example
///
/// ```
/// use phylo2vec::tree_vec::ops::newick::remove_branch_lengths;
///
/// let newick = "((0:0.0194831,2:0.924941)5:0.18209481,(1:1,3:3)4:4)6;";
/// let result = remove_branch_lengths(newick);
/// assert_eq!(result, "((0,2)5,(1,3)4)6;");
/// ```
pub fn remove_branch_lengths(newick: &str) -> String {
    let newick_patterns = NewickPatterns::new();
    newick_patterns
        .branch_lengths
        .replace_all(newick, "")
        .to_string()
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
    newick_patterns.parents.is_match(newick)
}

/// Check if the Newick string has branch lengths
///
/// # Example
///
/// ```
/// use phylo2vec::tree_vec::ops::newick::has_branch_lengths;
/// let newick = "(((0:0.1,(3:0.1,5:0.1)6:0.1)8:0.1,2:0.1)9:0.1,(1:0.1,4:0.1)7:0.1)10;";
/// let result = has_branch_lengths(newick);
/// assert!(result);
///
/// let newick_no_bls = "(((0,(3,5)6)8,2)9,(1,4)7)10;";
/// let result_no_bls = has_branch_lengths(newick_no_bls);
/// assert!(!result_no_bls);
/// ```
pub fn has_branch_lengths(newick: &str) -> bool {
    let newick_patterns = NewickPatterns::new();
    newick_patterns.branch_lengths.is_match(newick)
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

    result.len()
}

/// Create an integer-taxon label mapping (label_mapping)
/// from a string-based newick (where leaves are strings)
/// and produce a mapped integer-based newick (where leaves are integers).
///
/// Note 1: this does not check for the validity of the Newick string.
///
/// Note 2: the parent nodes are removed from the output,
/// but the branch lengths/annotations are kept.
///
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::newick::create_label_mapping;
///
/// let nw_str = "(((aaaaaaaaae,(aaaaaaaaaf,aaaaaaaaag)),aaaaaaaaaa),(aaaaaaaaab,(aaaaaaaaac,aaaaaaaaad)));";
/// let (nw_int, label_mapping) = create_label_mapping(&nw_str);
/// assert_eq!(nw_int, "(((0,(1,2)),3),(4,(5,6)));");
/// assert_eq!(label_mapping.get(&0), Some(&"aaaaaaaaae".to_string()));
///
/// // With branch lengths and parents
/// let nw_str2 = "(((aaaaaaaaae:2.5,(aaaaaaaaaf:1,aaaaaaaaag:1)p0:0.1)p1:405,aaaaaaaaaa:1)p2:1,(aaaaaaaaab:1,(aaaaaaaaac:1,aaaaaaaaad:1)p3:1)p4:1)p5;";
/// let (nw_int2, label_mapping2) = create_label_mapping(&nw_str2);
/// assert_eq!(nw_int2, "(((0:2.5,(1:1,2:1):0.1):405,3:1):1,(4:1,(5:1,6:1):1):1);");
/// assert_eq!(label_mapping2.get(&1), Some(&"aaaaaaaaaf".to_string()));
/// ```
pub fn create_label_mapping(newick: &str) -> (String, HashMap<usize, String>) {
    if newick.is_empty() {
        return (String::new(), HashMap::<usize, String>::new());
    }

    let mut newick_int = String::new();
    // Label mapping
    // Key: leaf index (integer)
    // Value: leaf label (string)
    let mut label_mapping: HashMap<usize, String> = HashMap::new();
    let mut next_leaf: usize = 0;

    let newick_bytes = newick.as_bytes();

    let mut i: usize = 0;

    while i < newick.len() {
        // Next character
        let c: char = newick_bytes[i] as char;

        if c != '(' && c != ',' && c != ';' && c != ')' {
            // Get the next node
            let (leaf, end) = node_substr(newick, i);
            i = end - 1;

            // Push the next leaf to the integer Newick
            newick_int.push_str(&next_leaf.to_string());

            match leaf.split_once(':') {
                Some((n, bl)) => {
                    // Add the node to the label mapping
                    label_mapping.insert(next_leaf, n.to_string());
                    // Push the branch length to the integer Newick
                    newick_int.push(':');
                    newick_int.push_str(bl);
                }
                None => {
                    // No branch length => add the node as it is to the label mapping
                    label_mapping.insert(next_leaf, leaf.to_string());
                }
            }

            // Increment the next leaf index
            next_leaf += 1;
        } else {
            newick_int.push(c);

            // Process internal/parent nodes
            if c == ')' {
                i += 1;
                let (parent, end) = node_substr(newick, i);
                i = end - 1;

                if let Some((_, bl)) = parent.split_once(':') {
                    // Push the parent branch length to the integer Newick
                    // Note: the parent node itself is not pushed
                    newick_int.push(':');
                    newick_int.push_str(bl);
                }
            }
        }

        i += 1;
    }

    (newick_int, label_mapping)
}

/// Apply an integer-taxon label mapping (label_mapping)
/// to an integer-based newick (where leaves are integers)
/// and produce a mapped Newick (where leaves are strings (taxa))
///
/// For more details, see `create_label_mapping`.
///
/// Note 1: this does not check for the validity of the Newick string.
///
/// Note 2: the parent nodes are removed from the output,
/// but the branch lengths/annotations are kept.
///
/// # Example
/// ```
/// use std::collections::HashMap;
/// use phylo2vec::tree_vec::ops::newick::apply_label_mapping;
///
/// let nw_int = "(((0:1,(1:1,2:1):1):1,3:1):1,(4:1,(5:1,6:1):1):1);";
/// let label_mapping = HashMap::<usize, String>::from(
///     [
///         (0, "aaaaaaaaae".into()), (1, "aaaaaaaaaf".into()),
///         (2, "aaaaaaaaag".into()), (3, "aaaaaaaaaa".into()),
///         (4, "aaaaaaaaab".into()), (5, "aaaaaaaaac".into()),
///         (6, "aaaaaaaaad".into())
///     ]
/// );
/// let nw_str = apply_label_mapping(&nw_int, &label_mapping).expect("Oops");
/// assert_eq!(
///     nw_str,
///     "(((aaaaaaaaae:1,(aaaaaaaaaf:1,aaaaaaaaag:1):1)\
///     :1,aaaaaaaaaa:1):1,(aaaaaaaaab:1,(aaaaaaaaac:1,aaaaaaaaad:1)\
///     :1):1);"
/// );
/// ```
pub fn apply_label_mapping(
    newick: &str,
    label_mapping: &HashMap<usize, String>,
) -> Result<String, NewickError> {
    if newick.is_empty() {
        return Ok(String::new());
    }

    let mut newick_str = String::new();
    let newick_bytes = newick.as_bytes();

    let mut i: usize = 0;

    while i < newick.len() {
        let c: char = newick_bytes[i] as char;

        if c != '(' && c != ',' && c != ';' && c != ')' {
            // Get the next node
            let (leaf, end) = node_substr(newick, i);
            i = end - 1;

            match leaf.split_once(':') {
                Some((n, bl)) => {
                    // Add the node to the label mapping
                    let leaf_int = n.parse::<usize>().map_err(NewickError::ParseIntError)?;

                    // Push the mapped leaf to the string Newick
                    newick_str.push_str(&label_mapping[&leaf_int]);

                    // Push the branch length to the string Newick
                    newick_str.push(':');
                    newick_str.push_str(bl);
                }
                None => {
                    let leaf_int = leaf.parse::<usize>().map_err(NewickError::ParseIntError)?;

                    // No branch length => add the mapped node as it is to the string Newick
                    newick_str.push_str(&label_mapping[&leaf_int]);
                }
            }
        } else {
            newick_str.push(c);

            // Process internal/parent nodes
            if c == ')' {
                i += 1;
                let (parent, end) = node_substr(newick, i);
                i = end - 1;

                if let Some((_, bl)) = parent.split_once(':') {
                    // Push the parent branch length to the string Newick
                    // Note: the parent node itself is not pushed
                    newick_str.push(':');
                    newick_str.push_str(bl);
                }
            }
        }

        i += 1;
    }

    Ok(newick_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_vec::ops::{parse_matrix, to_newick_from_matrix, to_newick_from_vector};
    use crate::utils::{sample_matrix, sample_vector};
    use rand::{distributions::Alphanumeric, Rng};
    use rstest::*;

    #[rstest]
    #[case("(((0,(3,5)6)8,2)9,(1,4)7)10;", "(((0,(3,5)),2),(1,4));")]
    #[case("(0,(1,(2,(3,(4,5)6)7)8)9)10;", "(0,(1,(2,(3,(4,5)))));")]
    #[case("((0,2)5,(1,3)4)6;", "((0,2),(1,3));")]
    fn test_remove_parent_labels(#[case] newick: &str, #[case] expected: &str) {
        let result = remove_parent_labels(newick);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(
        "(((0:0.0,(3:0.3,5:0.5)6:0.6)8:0.8,2:0.2)9:0.9,(1:0.1,4:0.4)7:0.7)10;",
        "(((0,(3,5)6)8,2)9,(1,4)7)10;"
    )]
    #[case(
        "(0:0.0,(1:0.00001,(2:2,(3:3,(4:4,5:5.0)6:6)7:7)8:8)9:9)10;",
        "(0,(1,(2,(3,(4,5)6)7)8)9)10;"
    )]
    #[case(
        "((0:0.0194831,2:0.924941)5:0.18209481,(1:1,3:3)4:4)6;",
        "((0,2)5,(1,3)4)6;"
    )]
    #[case("(((2:0.02,1:0.01),0:0.041),3:1.42);", "(((2,1),0),3);")]
    fn test_remove_branch_lengths(#[case] newick: &str, #[case] expected: &str) {
        let result = remove_branch_lengths(newick);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_has_parents(#[case] n_leaves: usize) {
        let v = sample_vector(n_leaves, false);
        let newick = to_newick_from_vector(&v);
        // Check if the newick string has parents
        let result = has_parents(&newick);
        assert!(result); // skipcq: RS-W1024

        // Check if the newick string does not have parents
        let result_no_parents = has_parents(&remove_parent_labels(&newick));
        assert!(!result_no_parents); // skipcq: RS-W1024
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_has_branch_lengths(#[case] n_leaves: usize) {
        let m = sample_matrix(n_leaves, false);
        let newick = to_newick_from_matrix(&m);
        // Check if the newick string has branch lengths
        let result = has_branch_lengths(&newick);
        assert!(result); // skipcq: RS-W1024

        // Check if the newick string does not have branch lengths
        let (v, _) = parse_matrix(&m);
        let result_no_branch_lengths = has_branch_lengths(&to_newick_from_vector(&v));
        assert!(!result_no_branch_lengths); // skipcq: RS-W1024
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_find_num_leaves(#[case] n_leaves: usize) {
        let v = sample_vector(n_leaves, false);
        let newick = to_newick_from_vector(&v);
        // Check if the newick string has parents
        let result = find_num_leaves(&newick);
        assert_eq!(result, n_leaves);
    }

    #[rstest]
    #[case("((1:0.5,2:0.7)4:0.9,3:0.8)5;", vec![[1, 2, 4], [4, 3, 5]], vec![[0.5, 0.7], [0.9, 0.8]])]
    #[case("(1:0.5,2:0.7);", vec![[1, 2, 2]], vec![[0.5, 0.7]] )]
    fn test_get_cherries_with_bls(
        #[case] newick: &str,
        #[case] expected_ancestry: Vec<[usize; 3]>,
        #[case] expected_bls: Vec<[f32; 2]>,
    ) {
        let (ancestry, bls) =
            get_cherries_with_bls(newick).expect("failed to get cherries with branch lengths");

        // Verify the ancestry
        assert_eq!(ancestry, expected_ancestry); // Ensure ancestry matches the expected

        // Verify the branch lengths
        assert_eq!(bls.len(), expected_bls.len()); // Ensure the number of branch lengths is correct
        assert_eq!(bls, expected_bls); // Ensure branch lengths match the expected
    }

    // Generate a random Newick string with random taxon labels
    fn generate_random_string_newick(n_leaves: usize) -> String {
        // Create random taxon labels
        // Alphanumeric: a-z, A-Z and 0-9.
        let mut rng = rand::thread_rng();
        let mut mapping = HashMap::<usize, String>::new();

        for i in 0..n_leaves {
            let taxon: String = (&mut rng)
                .sample_iter(&Alphanumeric)
                .take(10)
                .map(char::from)
                .collect();
            mapping.insert(i, taxon);
        }

        // Generate a random newick
        let v = sample_vector(n_leaves, false);
        let nw1 = remove_parent_labels(&to_newick_from_vector(&v));

        // Map new taxon labels to each leaf
        apply_label_mapping(&nw1, &mapping).expect("Failed to apply label mapping")
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_label_mapping(#[case] n_leaves: usize) {
        let nw_str = generate_random_string_newick(n_leaves);

        // Get the "sorted" version of the integer Newick
        // Ex: nw1 = "(((0,(1,5)),((2,((3,9),7)),(6,8))),4);"
        // Ex: nw3 = "(((0,(1,2)),((3,((4,5),6)),(7,8))),9);"
        let (nw_int, mapping) = create_label_mapping(&nw_str);

        let nw_str2 = apply_label_mapping(&nw_int, &mapping).expect("Oops");

        // Check if nw2 (first Newick with taxon labels) == nw4 (second Newick with taxon labels)
        // Note: asserting the equality of a random integer Newick from phylo2vec with
        // a new integer Newick from create + apply_label_mapping does not work
        // because the order of the leaves is not guaranteed to be the same.
        // Ex: nw1 = "(((0,(1,5)),((2,((3,9),7)),(6,8))),4);"
        // Ex: nw3 = "(((0,(1,2)),((3,((4,5),6)),(7,8))),9);"
        // Same topology, but different leaf placements
        assert_eq!(nw_str, nw_str2);
    }
}
