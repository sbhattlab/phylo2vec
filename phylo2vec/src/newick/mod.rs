use std::collections::HashMap;
use thiserror;

use crate::types::Ancestry;

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
    // For problematic stack popping in parsing functions
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
///   the `phylo2vec` convention.
///
/// # Returns
///
/// A vector of triplets representing the cherries in the tree.
/// Throws a `NewickError` if the Newick string is invalid.
///
/// # Examples
///
/// ```
/// use phylo2vec::newick::parse;
///
/// let newick = "((0,2)5,(1,3)4)6;";
///
/// let cherries = parse(newick).expect("Oops");
///
/// assert_eq!(cherries, vec![[0, 2, 5], [1, 3, 4], [5, 4, 6]]);
/// ```
/// # Errors
/// Returns a `NewickError` if the Newick string is ill-formed:
/// * Ill-formed parentheses
/// * Node could not be parsed as int
pub fn parse(newick: &str) -> Result<Ancestry, NewickError> {
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
                c_ordered.sort_unstable();

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
///   the `phylo2vec` convention.
///
/// # Returns
///
/// A vector of triplets representing the cherries in the tree.
/// Throws a `NewickError` if the Newick string is invalid.
///
/// # Examples
///
/// ```
/// use phylo2vec::newick::parse_with_bls;
///
/// // Without explicit parent nodes
/// let newick = "((0:0.1,2:0.2):0.3,(1:0.5,3:0.7):0.4);";
///
/// let (cherries, bls) = parse_with_bls(newick).expect("Oops");
///
/// assert_eq!(cherries, vec![[0, 2, 2], [1, 3, 3], [0, 1, 1]]);
/// assert_eq!(bls, vec![[0.1, 0.2], [0.5, 0.7], [0.3, 0.4]]);
/// ```
/// # Panics
/// Panics if branch length parsing failed (most likely missing)
///
/// # Errors
/// Returns a `NewickError` if the Newick string is ill-formed:
/// * Ill-formed parentheses
/// * Node could not be parsed as int
/// * Branch length could not be parsed as float
pub fn parse_with_bls(newick: &str) -> Result<(Ancestry, Vec<[f64; 2]>), NewickError> {
    if newick.is_empty() {
        return Ok((Vec::new(), Vec::new())); // Return empty ancestry and branch length vectors
    }
    let mut ancestry: Ancestry = Vec::new();
    let mut bls: Vec<[f64; 2]> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut bl_stack: Vec<f64> = Vec::new();

    let mut i: usize = 0;

    let newick_bytes = newick.as_bytes();

    while i < newick.len() {
        let c: char = newick_bytes[i] as char;

        if c == ')' {
            // Pop the children nodes from the stack
            let c2: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;
            let c1: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;

            // Pop the BLs from the BL stack
            let bl2: f64 = bl_stack.pop().ok_or(NewickError::StackUnderflow)?;
            let bl1: f64 = bl_stack.pop().ok_or(NewickError::StackUnderflow)?;
            bls.push([bl1, bl2]);

            let (annotation, end) = node_substr(newick, i + 1);
            i = end - 1;

            match annotation.split_once(':') {
                Some(("", blp)) => {
                    // Case 1: No parent node
                    let mut c_ordered = [c1, c2];
                    c_ordered.sort_unstable();

                    // Add the triplet (c1, c2, max(c1, c2))
                    ancestry.push([c1, c2, c_ordered[1]]);
                    // Push min(c1, c2) to the stack
                    // to represent this internal node going forward
                    stack.push(c_ordered[0]);
                    // Push the parent BL to the BL stack
                    bl_stack.push(blp.parse::<f64>().map_err(NewickError::ParseFloatError)?);
                }
                Some((p, blp)) => {
                    // Case 2: Parent node present

                    // Add the triplet (c1, c2, p)
                    let p_int = p.parse::<usize>().map_err(NewickError::ParseIntError)?;
                    ancestry.push([c1, c2, p_int]);
                    // Push the parent node to the stack
                    stack.push(p_int);
                    // Push the parent BL to the BL stack
                    bl_stack.push(blp.parse::<f64>().map_err(NewickError::ParseFloatError)?);
                }
                None => {
                    if end == newick.len() - 1 {
                        // Case 3: Reached the root
                        if annotation.is_empty() {
                            // Case 3.1: No parent node --> store the max leaf
                            let mut c_ordered = [c1, c2];
                            c_ordered.sort_unstable();

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
                    }
                    // Case 4: Unknown (missing annotation?)
                    panic!("Missing annotation in the Newick string");
                }
            }
        } else if c.is_ascii_digit() {
            // Get the next annotated node
            let (annotated_node, end) = node_substr(newick, i);
            i = end - 1;

            // Split the node from its branch length
            let (node, bln) = annotated_node.split_once(':').unwrap();

            stack.push(node.parse::<usize>().map_err(NewickError::ParseIntError)?);
            bl_stack.push(bln.parse::<f64>().map_err(NewickError::ParseFloatError)?);
        }

        i += 1;
    }

    Ok((ancestry, bls))
}

/// Remove parent labels from the Newick string
///
/// # Examples
///
/// ```
/// use phylo2vec::newick::remove_parent_labels;
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
/// # Examples
///
/// ```
/// use phylo2vec::newick::remove_branch_lengths;
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
/// # Examples
///
/// ```
/// use phylo2vec::newick::has_parents;
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
/// # Examples
///
/// ```
/// use phylo2vec::newick::has_branch_lengths;
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

/// Check if the Newick string has integer labels
///
/// # Examples
///
/// ```
/// use phylo2vec::newick::has_integer_labels;
/// let nw_str = "(((E:0.1,(A:0.1,C:0.1):0.1):0.1,F:0.1):0.1,(D:0.1,B:0.1):0.1);";
/// let result_str = has_integer_labels(nw_str);
/// assert!(!result_str);
///
/// let nw_mixed = "(((0:0.1,(A:0.1,C:0.1)6:0.1)8:0.1,2:0.1)9:0.1,(1:0.1,4:0.1)7:0.1)10;";
/// let result_mixed = has_integer_labels(nw_mixed);
/// assert!(!result_mixed);
///
/// let nw_int = "(((0:0.1,(3:0.1,5:0.1)6:0.1)8:0.1,2:0.1)9:0.1,(1:0.1,4:0.1)7:0.1)10;";
/// let result_int = has_integer_labels(nw_int);
/// assert!(result_int);
/// ```
pub fn has_integer_labels(newick: &str) -> bool {
    let newick_patterns = NewickPatterns::new();
    let n_left = newick_patterns.left_node.find_iter(newick).count();
    let n_right = newick_patterns.right_node.find_iter(newick).count();
    let n_left_generic = newick_patterns.left_node_generic.find_iter(newick).count();
    let n_right_generic = newick_patterns.right_node_generic.find_iter(newick).count();

    n_left == n_left_generic && n_right == n_right_generic
}

/// Find the number of leaves in the Newick string
///
/// # Examples
///
/// ```
/// use phylo2vec::newick::find_num_leaves;
///
/// let newick = "(((0,(3,5)6)8,2)9,(1,4)7)10;";
/// let result = find_num_leaves(newick);
/// assert_eq!(result, 6);
/// ```
/// # Panics
/// Panics if the Newick string is ill-formed.
pub fn find_num_leaves(newick: &str) -> usize {
    let newick_patterns = NewickPatterns::new();
    let result = newick_patterns
        .pairs
        .captures_iter(newick)
        .map(|caps| {
            let (_, [_, node]) = caps.extract();
            node.parse::<usize>().unwrap()
        })
        .count();

    result
}

/// Create an integer-taxon label mapping (`label_mapping`)
/// from a string-based newick (where leaves are strings)
/// and produce a mapped integer-based newick (where leaves are integers).
///
/// Note 1: this does not check for the validity of the Newick string.
///
/// Note 2: the parent nodes are removed from the output,
/// but the branch lengths/annotations are kept.
///
/// # Examples
///
/// ```
/// use phylo2vec::newick::create_label_mapping;
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
        return (String::default(), HashMap::<usize, String>::new());
    }

    let mut newick_int = String::default();
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

/// Apply an integer-taxon label mapping (`label_mapping`)
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
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use phylo2vec::newick::apply_label_mapping;
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
/// # Errors
/// Throws a `NewickError` if Newick parsing failed.
pub fn apply_label_mapping<S: ::std::hash::BuildHasher>(
    newick: &str,
    label_mapping: &HashMap<usize, String, S>,
) -> Result<String, NewickError> {
    if newick.is_empty() {
        return Ok(String::default());
    }

    let mut newick_str = String::default();
    let newick_bytes = newick.as_bytes();

    let mut i: usize = 0;

    while i < newick.len() {
        let c: char = newick_bytes[i] as char;

        if c != '(' && c != ',' && c != ';' && c != ')' {
            // Get the next node
            let (leaf, end) = node_substr(newick, i);
            i = end - 1;

            if let Some((n, bl)) = leaf.split_once(':') {
                // Add the node to the label mapping
                let leaf_int = n.parse::<usize>().map_err(NewickError::ParseIntError)?;

                // Push the mapped leaf to the string Newick
                newick_str.push_str(&label_mapping[&leaf_int]);

                // Push the branch length to the string Newick
                newick_str.push(':');
                newick_str.push_str(bl);
            } else {
                let leaf_int = leaf.parse::<usize>().map_err(NewickError::ParseIntError)?;

                // No branch length => add the mapped node as it is to the string Newick
                newick_str.push_str(&label_mapping[&leaf_int]);
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
    use crate::matrix::base::{parse_matrix, sample_matrix};
    use crate::matrix::convert::to_newick as to_newick_from_matrix;
    use crate::vector::base::sample_vector;
    use crate::vector::convert::to_newick as to_newick_from_vector;
    use rand::{distr::Alphanumeric, Rng};
    use rstest::*;

    // Test that default() creates the same patterns as new()
    #[rstest]
    #[case("(1,2)3")]
    #[case("((1,2)3,(4,5)6)7")]
    // Case with BLs
    #[case("((1:0.1,2:0.2)3:0.3,(4:0.4,5:0.5)6:0.6)7:0.7")]
    fn test_newick_patterns_default(#[case] newick: &str) {
        let patterns_new = NewickPatterns::new();
        let patterns_default = NewickPatterns::default();

        // Test that default() creates the same patterns as new()
        assert_eq!(
            patterns_new.pairs.is_match(newick),
            patterns_default.pairs.is_match(newick)
        );
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
        let newick = to_newick_from_matrix(&m.view());
        // Check if the newick string has branch lengths
        let result = has_branch_lengths(&newick);
        assert!(result); // skipcq: RS-W1024

        // Check if the newick string does not have branch lengths
        let (v, _) = parse_matrix(&m.view());
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
    #[case("(((2:2e-2,1:1e-2),0:4.1e-2),3:1.42e0);", "(((2,1),0),3);")]
    #[case(
        "((0:1.94831E-06,2:9.25e-1)5:1.8209481e-3,(1:1,3:3)4:4)6;",
        "((0,2)5,(1,3)4)6;"
    )]
    fn test_remove_branch_lengths(#[case] newick: &str, #[case] expected: &str) {
        let result = remove_branch_lengths(newick);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case("((1,2)4,3)5;", vec![[1, 2, 4], [4, 3, 5]])]
    #[case("((1,2)5,(3,4)6)7;", vec![[1, 2, 5], [3, 4, 6], [5, 6, 7]])]
    #[case("(1,2);", vec![[1, 2, 2]] )]
    fn test_parse(#[case] newick: &str, #[case] expected_ancestry: Vec<[usize; 3]>) {
        let ancestry = parse(newick).expect("failed to get cherries");

        // Verify the ancestry
        assert_eq!(ancestry, expected_ancestry);
    }

    #[rstest]
    #[case("((1:0.5,2:0.7)4:0.9,3:0.8)5;", vec![[1, 2, 4], [4, 3, 5]], vec![[0.5, 0.7], [0.9, 0.8]])]
    #[case("((1:1,2:2)5:5,(3:3,4:4)6:6)7;", vec![[1, 2, 5], [3, 4, 6], [5, 6, 7]], vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] )]
    #[case("((1:1,2:2)5:5,(4:4,3:3)6:6)7;", vec![[1, 2, 5], [4, 3, 6], [5, 6, 7]], vec![[1.0, 2.0], [4.0, 3.0], [5.0, 6.0]] )]
    #[case("(1:0.5,2:0.7);", vec![[1, 2, 2]], vec![[0.5, 0.7]] )]
    fn test_parse_with_bls(
        #[case] newick: &str,
        #[case] expected_ancestry: Vec<[usize; 3]>,
        #[case] expected_bls: Vec<[f64; 2]>,
    ) {
        let (ancestry, bls) =
            parse_with_bls(newick).expect("failed to get cherries with branch lengths");

        // Verify the ancestry
        assert_eq!(ancestry, expected_ancestry);

        // Verify the branch lengths
        assert_eq!(bls.len(), expected_bls.len());
        assert_eq!(bls, expected_bls);
    }

    #[rstest]
    // No BLs
    #[case("((1,2)4,3)5;")]
    #[case("((1,2)5,(3,4)6)7;")]
    // Partially missing BLs
    #[case("((1:0.5,2)4,3:0.8)5;")]
    #[case("((1:0.5,2)4:0.9,3:0.8)5;")]
    #[case("((1:,2:0.2)4:0.9,3:0.8)5;")]
    #[should_panic]
    fn test_parse_with_bls_panics(#[case] newick: &str) {
        parse_with_bls(newick).expect("Expected Newick parsing to fail");
    }

    // Generate a random Newick string with random taxon labels
    fn generate_random_string_newick(n_leaves: usize) -> String {
        // Create random taxon labels
        // Alphanumeric: a-z, A-Z and 0-9.
        let mut rng = rand::rng();
        let mut mapping = HashMap::<usize, String>::new();
        let strlen = 10;

        for i in 0..n_leaves {
            let taxon: String = (&mut rng)
                .sample_iter(&Alphanumeric)
                .take(strlen)
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
    #[case(0)]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_label_mapping(#[case] n_leaves: usize) {
        let nw_str = match n_leaves {
            0 => String::default(),
            _ => generate_random_string_newick(n_leaves),
        };

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

    // Generate a random Newick string with random taxon labels
    fn generate_random_string_newick_with_bls(n_leaves: usize) -> String {
        // Create random taxon labels
        // Alphanumeric: a-z, A-Z and 0-9.
        let mut rng = rand::rng();
        let mut mapping = HashMap::<usize, String>::new();
        let strlen = 10;

        for i in 0..n_leaves {
            let taxon: String = (&mut rng)
                .sample_iter(&Alphanumeric)
                .take(strlen)
                .map(char::from)
                .collect();
            mapping.insert(i, taxon);
        }

        // Generate a random newick
        let m = sample_matrix(n_leaves, false);
        let nw1 = remove_parent_labels(&to_newick_from_matrix(&m.view()));

        // Map new taxon labels to each leaf
        apply_label_mapping(&nw1, &mapping).expect("Failed to apply label mapping")
    }

    #[rstest]
    #[case(0)]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_label_mapping_with_bls(#[case] n_leaves: usize) {
        let nw_str = match n_leaves {
            0 => String::default(),
            _ => generate_random_string_newick_with_bls(n_leaves),
        };

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

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_has_integer_labels(#[case] n_leaves: usize) {
        // nw_str should not have integer labels
        let nw_str = generate_random_string_newick(n_leaves);
        assert!(!has_integer_labels(&nw_str));

        // nw_int should have integer labels
        let (nw_int, _) = create_label_mapping(&nw_str);

        assert!(has_integer_labels(&nw_int));
    }
}
