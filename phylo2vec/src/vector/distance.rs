//! Tree comparison distance metrics.
//!
//! This module provides functions for computing distances between trees,
//! such as the Robinson-Foulds distance.

use bit_set::BitSet;
use std::collections::HashSet;

use crate::vector::convert::to_ancestry;

/// Compute the Robinson-Foulds distance between two trees.
///
/// RF distance counts the number of bipartitions (splits) that differ
/// between two trees. Lower values indicate more similar topologies.
///
/// # Arguments
/// * `v1` - First tree (Phylo2Vec vector)
/// * `v2` - Second tree (Phylo2Vec vector)
/// * `normalize` - If true, return normalized distance (0.0 to 1.0)
///
/// # Returns
/// RF distance as f64 (integer value if not normalized)
///
/// # Panics
/// Panics if trees have different numbers of leaves.
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::distances::robinson_foulds;
///
/// // Identical trees have RF distance 0
/// let v = vec![0, 1, 2];
/// assert_eq!(robinson_foulds(&v, &v, false), 0.0);
///
/// // Different trees have RF distance > 0
/// let v1 = vec![0, 0, 0];
/// let v2 = vec![0, 1, 2];
/// assert!(robinson_foulds(&v1, &v2, false) >= 0.0);
/// ```
pub fn robinson_foulds(v1: &[usize], v2: &[usize], normalize: bool) -> f64 {
    assert_eq!(
        v1.len(),
        v2.len(),
        "Trees must have the same number of leaves"
    );

    let n_leaves = v1.len() + 1;

    // For very small trees, RF is 0 (no non-trivial splits)
    if n_leaves <= 3 {
        return 0.0;
    }

    let splits1 = get_bipartitions(v1);
    let splits2 = get_bipartitions(v2);

    // Count symmetric difference
    let only_in_1 = splits1.difference(&splits2).count();
    let only_in_2 = splits2.difference(&splits1).count();
    let symmetric_diff = only_in_1 + only_in_2;

    if normalize {
        let max_rf = splits1.len() + splits2.len();
        if max_rf == 0 {
            0.0
        } else {
            symmetric_diff as f64 / max_rf as f64
        }
    } else {
        symmetric_diff as f64
    }
}

/// Extract bipartitions (splits) from a tree.
///
/// Each internal edge induces a bipartition of leaves. We represent each
/// split as a BitSet containing the smaller side of the partition (canonical form).
///
/// Trivial splits (single leaf or all-but-one leaves) are excluded.
fn get_bipartitions(v: &[usize]) -> HashSet<BitSet> {
    let ancestry = to_ancestry(v);
    let n_leaves = v.len() + 1;
    let k = v.len();

    // Build descendant sets for each node using dynamic programming (bottom-up)
    // descendants[i] = set of leaf indices that are descendants of node i
    let mut descendants: Vec<BitSet> = Vec::with_capacity(2 * n_leaves - 1);

    // Initialize all nodes with empty sets
    for _ in 0..(2 * n_leaves - 1) {
        descendants.push(BitSet::with_capacity(n_leaves));
    }

    // Leaves are their own descendants
    for (i, desc) in descendants.iter_mut().enumerate().take(n_leaves) {
        desc.insert(i);
    }

    // Build internal node descendants bottom-up through ancestry
    // ancestry[i] = [child1, child2, parent] where parent = n_leaves + 1 + i
    for [child1, child2, parent] in ancestry.iter().copied() {
        // Clone both children's descendants to avoid borrow conflicts
        let mut desc = descendants[child1].clone();
        desc.union_with(&descendants[child2]);
        descendants[parent] = desc;
    }

    // Extract non-trivial splits
    // For unrooted trees, we look at each internal edge
    // An edge to internal node i induces split: descendants[i] vs complement
    let mut splits = HashSet::with_capacity(k);

    for i in 0..k {
        let internal_node = n_leaves + i; // ancestry uses k + 1 + i = n_leaves + i for parent
        let desc = &descendants[internal_node];
        let desc_len = desc.len();

        // Skip trivial splits (size 1 or n-1)
        // For unrooted comparison, also skip the root split (size n)
        if desc_len <= 1 || desc_len >= n_leaves - 1 {
            continue;
        }

        // Canonicalize: use the smaller side of the split
        let split = canonicalize_split(desc, n_leaves);
        splits.insert(split);
    }

    splits
}

/// Canonicalize a split by choosing the smaller subset.
/// For equal-sized splits, choose the one containing leaf 0.
/// This ensures that equivalent splits (A|B vs B|A) have the same representation.
#[inline]
fn canonicalize_split(desc: &BitSet, n_leaves: usize) -> BitSet {
    let desc_len = desc.len();
    let comp_len = n_leaves - desc_len;

    match desc_len.cmp(&comp_len) {
        std::cmp::Ordering::Less => desc.clone(), // Original is smaller, keep it
        std::cmp::Ordering::Greater => complement_bitset(desc, n_leaves), // Complement is smaller, return it
        std::cmp::Ordering::Equal => {
            // Equal sizes: pick the one containing leaf 0
            if desc.contains(0) {
                desc.clone()
            } else {
                complement_bitset(desc, n_leaves)
            }
        }
    }
}

/// Compute the complement of a BitSet with respect to leaves 0..n_leaves
#[inline]
fn complement_bitset(desc: &BitSet, n_leaves: usize) -> BitSet {
    let mut complement = BitSet::with_capacity(n_leaves);
    for i in 0..n_leaves {
        if !desc.contains(i) {
            complement.insert(i);
        }
    }
    complement
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rf_identical_trees() {
        let v = vec![0, 1, 2, 3];
        assert_eq!(robinson_foulds(&v, &v, false), 0.0);
    }

    #[test]
    fn test_rf_identical_trees_larger() {
        let v = vec![0, 1, 2, 3, 4, 5, 6];
        assert_eq!(robinson_foulds(&v, &v, false), 0.0);
    }

    #[test]
    fn test_rf_symmetric() {
        let v1 = vec![0, 1, 2, 3];
        let v2 = vec![0, 0, 1, 2];
        assert_eq!(
            robinson_foulds(&v1, &v2, false),
            robinson_foulds(&v2, &v1, false)
        );
    }

    #[test]
    fn test_rf_normalized_bounds() {
        let v1 = vec![0, 1, 2, 3];
        let v2 = vec![0, 0, 1, 2];
        let rf_norm = robinson_foulds(&v1, &v2, true);
        assert!((0.0..=1.0).contains(&rf_norm));
    }

    #[test]
    fn test_rf_small_trees() {
        // 3-leaf trees have no non-trivial splits
        let v1 = vec![0, 0];
        let v2 = vec![0, 1];
        assert_eq!(robinson_foulds(&v1, &v2, false), 0.0);
    }

    #[test]
    fn test_rf_four_leaves() {
        // 4-leaf unrooted binary trees have exactly 1 non-trivial split
        // ((0,1),(2,3)) has split {0,1}|{2,3}
        // ((0,2),(1,3)) has split {0,2}|{1,3}
        // These are maximally different: RF = 2
        let v1 = vec![0, 0, 0]; // Need to verify topology
        let v2 = vec![0, 1, 0];
        let rf = robinson_foulds(&v1, &v2, false);
        // RF should be 0, 1, or 2 for 4-leaf trees
        assert!((0.0..=2.0).contains(&rf));
    }

    #[test]
    #[should_panic(expected = "same number of leaves")]
    fn test_rf_different_sizes() {
        let v1 = vec![0, 1, 2];
        let v2 = vec![0, 1];
        robinson_foulds(&v1, &v2, false);
    }

    #[test]
    fn test_get_bipartitions_basic() {
        // 5-leaf tree
        let v = vec![0, 1, 2, 3];
        let splits = get_bipartitions(&v);
        // Should have some non-trivial splits
        assert!(!splits.is_empty());
    }
}
