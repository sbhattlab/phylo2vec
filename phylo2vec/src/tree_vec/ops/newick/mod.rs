use std::num::IntErrorKind;

use crate::tree_vec::types::{Ancestry, Pairs};

mod newick_patterns;

pub use newick_patterns::NewickPatterns;

fn _stoi_substr(s: &str, start: usize, end: &mut usize) -> Result<usize, IntErrorKind> {
    let s = &s[start..];
    let mut value = 0;
    for (i, c) in s.chars().enumerate() {
        if let Some(digit) = c.to_digit(10) {
            value = value * 10 + digit as usize;
            *end = start + i + 1;
        } else {
            break;
        }
    }
    if *end > start {
        Ok(value)
    } else {
        Err(std::num::IntErrorKind::Empty)
    }
}

fn _get_cherries_inner(ancestry: &mut Ancestry, newick: &str) {
    let mut stack = Vec::new();
    let mut i = 0;

    while i < newick.len() {
        let c = newick.as_bytes()[i] as char;
        if c == ')' {
            i += 1;

            let c2 = stack.pop().unwrap();
            let c1 = stack.pop().unwrap();

            let mut end = 0;
            let p = _stoi_substr(newick, i, &mut end).expect("Bad input");
            i = end - 1;

            ancestry.push([c1, c2, p]);
            stack.push(p);
        } else if c.is_ascii_digit() {
            let mut end = 0;
            let node = _stoi_substr(newick, i, &mut end).expect("Bad input");
            stack.push(node);
            i = end - 1;
        }
        i += 1;
    }
}

fn _get_cherries_no_parents_inner(ancestry: &mut Ancestry, newick: &str) {
    let newick_length = newick.len();
    let mut stack = Vec::with_capacity(newick_length);
    let mut i = 0;

    while i < newick_length {
        let c = newick.as_bytes()[i] as char;

        if c == ')' {
            let c2 = stack.pop().unwrap();
            let c1 = stack.pop().unwrap();

            let c_max = std::cmp::max(c1, c2);
            ancestry.push([c1, c2, c_max]);

            let c_min = std::cmp::min(c1, c2);
            stack.push(c_min);
        } else if c.is_ascii_digit() {
            let mut end = 0;
            let leaf = _stoi_substr(newick, i, &mut end).expect("Bad input");
            stack.push(leaf);
            i = end - 1;
        }
        i += 1;
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
    _get_cherries_inner(&mut ancestry, &newick[..newick.len() - 1]);
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
    _get_cherries_no_parents_inner(&mut ancestry, &newick[..newick.len() - 1]);
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

/// Build newick string from a vector of pairs
pub fn build_newick(pairs: &Pairs) -> String {
    let num_leaves = pairs.len() + 1;

    // Faster than map+collect for some reason
    let mut cache: Vec<String> = Vec::with_capacity(num_leaves);
    for i in 0..num_leaves {
        cache.push(i.to_string());
    }

    for (i, &(c1, c2)) in pairs.iter().enumerate() {
        // std::mem::take helps efficient swapping of values like std::move in C++
        let s1 = std::mem::take(&mut cache[c1]);
        let s2 = std::mem::take(&mut cache[c2]);
        // Parent node (not needed in theory, but left for legacy reasons)
        let sp = (num_leaves + i).to_string();

        // https://github.com/hoodie/concatenation_benchmarks-rs
        // 3 = 2 parentheses + 1 comma
        let capacity = s1.len() + s2.len() + sp.len() + 3;
        let mut sub_newick = String::with_capacity(capacity);
        sub_newick.push('(');
        sub_newick.push_str(&s1);
        sub_newick.push(',');
        sub_newick.push_str(&s2);
        sub_newick.push(')');
        sub_newick.push_str(&sp);

        cache[c1] = sub_newick;
    }

    cache[0].push(';');
    cache[0].clone()
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
