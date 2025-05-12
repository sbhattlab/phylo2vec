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
            i += 1;

            // Pop the children nodes from the stack
            let c2: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;
            let c1: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;

            // Get the parent node after ")"
            let (p, end) = node_substr(newick, i);
            i = end - 1;

            let p_int = p.parse::<usize>().map_err(NewickError::ParseIntError)?;

            // Add the triplet (c1, c2, p)
            ancestry.push([c1, c2, p_int]);

            // Push the parent node to the stack
            stack.push(p_int);
        } else if c.is_ascii_digit() {
            // Get the next node and push it to the stack
            let (node, end) = node_substr(newick, i);
            i = end - 1;

            stack.push(node.parse::<usize>().map_err(NewickError::ParseIntError)?);
        }

        i += 1;
    }

    Ok(ancestry)
}

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
            i += 1;

            // Pop the children nodes from the stack
            let c2: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;
            let c1: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;

            // Pop the BLs from the BL stack
            let bl2: f32 = bl_stack.pop().ok_or(NewickError::StackUnderflow)?;
            let bl1: f32 = bl_stack.pop().ok_or(NewickError::StackUnderflow)?;
            bls.push([bl1, bl2]);

            let (annotated_p, end) = node_substr(newick, i);
            i = end - 1;

            if end == newick.len() - 1 {
                let p = annotated_p.split(':').next().unwrap();
                let p_int: usize = p.parse::<usize>().map_err(NewickError::ParseIntError)?;
                ancestry.push([c1, c2, p_int]);
            } else {
                // Add the triplet (c1, c2, p)
                let (p, blp) = annotated_p.split_once(':').unwrap();

                let p_int = p.parse::<usize>().map_err(NewickError::ParseIntError)?;

                ancestry.push([c1, c2, p_int]);

                // Push the parent node to the stack
                stack.push(p_int);
                // Push the parent BL to the BL stack
                bl_stack.push(blp.parse::<f32>().map_err(NewickError::ParseFloatError)?);
            }
        } else if c.is_ascii_digit() {
            let (annotated_node, end) = node_substr(newick, i);
            i = end - 1;

            let (node, bln) = annotated_node.split_once(':').unwrap();

            stack.push(node.parse::<usize>().map_err(NewickError::ParseIntError)?);
            bl_stack.push(bln.parse::<f32>().map_err(NewickError::ParseFloatError)?);
        }

        i += 1;
    }

    Ok((ancestry, bls))
}

pub fn get_cherries_no_parents(newick: &str) -> Result<Ancestry, NewickError> {
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

            let mut c_ordered = [c1, c2];
            c_ordered.sort();

            // No parent annotation --> store the max leaf
            ancestry.push([c1, c2, c_ordered[1]]);

            // Push the min leaf to the stack to represent this internal node going forward
            stack.push(c_ordered[0]);
        } else if c.is_ascii_digit() {
            // Get the next leaf and push it to the stack
            let (leaf, end) = node_substr(newick, i);
            i = end - 1;

            stack.push(leaf.parse::<usize>().map_err(NewickError::ParseIntError)?);
        }

        i += 1;
    }

    Ok(ancestry)
}
pub fn get_cherries_no_parents_with_bls(
    newick: &str,
) -> Result<(Ancestry, Vec<[f32; 2]>), NewickError> {
    if newick.is_empty() {
        return Ok((Vec::new(), Vec::new())); // Return empty ancestry and branch length vectors
    }
    let mut ancestry: Ancestry = Vec::new();
    let mut bls: Vec<[f32; 2]> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut bl_stack: Vec<f32> = Vec::new();

    let mut i: usize = 0;

    let newick_bytes: &[u8] = newick.as_bytes();

    while i < newick.len() {
        let c: char = newick_bytes[i] as char;

        if c == ')' {
            i += 1;

            // Pop the children nodes from the stack
            let c2: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;
            let c1: usize = stack.pop().ok_or(NewickError::StackUnderflow)?;

            // Pop the BLs from the BL stack
            let bl2: f32 = bl_stack.pop().ok_or(NewickError::StackUnderflow)?;
            let bl1: f32 = bl_stack.pop().ok_or(NewickError::StackUnderflow)?;

            let mut c_ordered = [c1, c2];
            c_ordered.sort();

            // No parent annotation --> store the max leaf
            ancestry.push([c1, c2, c_ordered[1]]);
            bls.push([bl1, bl2]);
            // Find the parental BL
            // Ex: ":0.2"
            let (annotated_node, end) = node_substr(newick, i);
            i = end - 1;

            if annotated_node.is_empty() && end == newick.len() - 1 {
                // if this is true, we reached the root without a BL
                break;
            }

            // Push the min leaf to the stack
            stack.push(c_ordered[0]);
            // Push the parent BL to the BL stack
            bl_stack.push(
                annotated_node[1..]
                    .parse::<f32>()
                    .map_err(NewickError::ParseFloatError)?,
            );
        } else if c.is_ascii_digit() {
            let (annotated_node, end) = node_substr(newick, i);
            i = end - 1;

            let (node, bln) = annotated_node.split_once(':').unwrap();

            stack.push(node.parse::<usize>().map_err(NewickError::ParseIntError)?);
            bl_stack.push(bln.parse::<f32>().map_err(NewickError::ParseFloatError)?);
        }

        i += 1;
    }

    Ok((ancestry, bls))
}

/// Build newick string from a vector of pairs
pub fn build_newick(pairs: &Pairs) -> String {
    let num_leaves = pairs.len() + 1;

    // Faster than map+collect for some reason
    let mut cache: Vec<String> = (0..num_leaves).map(|i| i.to_string()).collect();

    for (i, &(c1, c2)) in pairs.iter().enumerate() {
        // std::mem::take helps efficient swapping of values like std::move in C++
        let s2 = std::mem::take(&mut cache[c2]);
        // Parent node (not needed in theory, but left for legacy reasons)
        let sp = (num_leaves + i).to_string();

        cache[c1].insert(0, '(');
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

    // Faster than map+collect for some reason
    let mut cache: Vec<String> = Vec::with_capacity(num_leaves);
    for i in 0..num_leaves {
        cache.push(i.to_string());
    }

    for (i, (&(c1, c2), &[bl1, bl2])) in pairs.iter().zip(branch_lengths.iter()).enumerate() {
        let s1 = std::mem::take(&mut cache[c1]);
        let s2 = std::mem::take(&mut cache[c2]);
        let sp = (num_leaves + i).to_string();
        let sb1 = bl1.to_string();
        let sb2 = bl2.to_string();

        let capacity = s1.len() + s2.len() + sp.len() + sb1.len() + sb2.len() + 5;
        let mut sub_newick = String::with_capacity(capacity);
        sub_newick.push('(');
        sub_newick.push_str(&s1);
        sub_newick.push(':');
        sub_newick.push_str(&sb1);
        sub_newick.push(',');
        sub_newick.push_str(&s2);
        sub_newick.push(':');
        sub_newick.push_str(&sb2);
        sub_newick.push(')');
        sub_newick.push_str(&sp);
        cache[c1] = sub_newick;
    }

    cache[0].push(';');
    cache[0].clone()
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
    return newick_patterns
        .branch_lengths
        .replace_all(newick, "")
        .to_string();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree_vec::ops::{parse_matrix, to_newick_from_matrix, to_newick_from_vector};
    use crate::utils::{sample_matrix, sample_vector};
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
            (ancestry, bls) =
                get_cherries_with_bls(newick).expect("failed to get cherries with branch lengths");
        } else {
            (ancestry, bls) = get_cherries_no_parents_with_bls(newick)
                .expect("failed to get cherries with branch lengths (no parents)");
        }

        // Verify the ancestry
        assert_eq!(ancestry, expected_ancestry); // Ensure ancestry matches the expected

        // Verify the branch lengths
        assert_eq!(bls.len(), expected_bls.len()); // Ensure the number of branch lengths is correct
        assert_eq!(bls, expected_bls); // Ensure branch lengths match the expected
    }
}
