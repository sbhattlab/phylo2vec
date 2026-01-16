/// Operations on Phylo2Vec vectors
use std::cmp::Ordering;
use std::collections::HashSet;

use crate::types::{Ancestry, Pairs};
use crate::vector::convert::{
    build_vector, from_pairs, order_cherries, order_cherries_no_parents, to_ancestry, to_pairs,
};

/// Adds a new leaf to the tree
///
/// # Arguments
/// * `tree` - The tree to add the leaf to
/// * `leaf` - Index of the new leaf to add
/// * `branch` - Index of the branch to attach the leaf to
///
/// # Result
/// Modifies the tree structure by adding the new leaf and updating indices
pub fn add_leaf(v: &mut Vec<usize>, leaf: usize, branch: usize) -> Vec<usize> {
    v.push(branch);

    let mut ancestry_add = to_ancestry(v);

    let mut found_first_leaf = false;
    for row in ancestry_add.iter_mut() {
        for val in row.iter_mut() {
            if !found_first_leaf && *val == v.len() {
                // Find the indices of the first leaf
                // and then set the value to the new leaf
                *val = leaf;
                found_first_leaf = true;
            } else if *val >= leaf {
                *val += 1;
            }
        }
    }

    // ancestry_add[leaf_coords][leaf_col] = leaf as isize;
    // let ancestry_add_ref = &mut ancestry_add;
    order_cherries(&mut ancestry_add);
    order_cherries_no_parents(&mut ancestry_add);
    build_vector(&ancestry_add)
}

/// Removes a leaf from the tree
///
/// # Arguments
/// * `tree` - The tree to remove the leaf from
/// * `leaf` - Index of the leaf to remove
///
/// # Returns
/// The index of the sister node of the removed leaf
///
/// # Side effects
/// Modifies the tree structure by removing the leaf and updating indices
pub fn remove_leaf(v: &mut [usize], leaf: usize) -> (Vec<usize>, usize) {
    let ancestry = to_ancestry(v);
    let leaf_coords = argeq(&ancestry, leaf);
    let leaf_row = leaf_coords.0;
    let leaf_col = leaf_coords.1;

    // Find the parent of the leaf to remove
    let parent = ancestry[leaf_row][2];
    let sister = ancestry[leaf_row][1 - leaf_col];
    let num_cherries = ancestry.len();

    let mut ancestry_rm = Vec::with_capacity(num_cherries - 1);

    for r in 0..num_cherries - 1 {
        let mut new_row = if r < leaf_row {
            ancestry[r]
        } else {
            ancestry[r + 1]
        };

        for node in new_row.iter_mut() {
            if *node == parent {
                *node = sister;
            }

            // Subtract 1 for leaves > "leaf"
            // (so that the vector is still valid)
            if *node > leaf {
                *node -= 1;
                if *node >= parent {
                    *node -= 1;
                }
            }
        }

        ancestry_rm.push(new_row);
    }

    order_cherries(&mut ancestry_rm);
    order_cherries_no_parents(&mut ancestry_rm);
    let new_vec = build_vector(&ancestry_rm);

    (new_vec, sister)
}

pub fn argeq(ancestry: &Ancestry, node: usize) -> (usize, usize) {
    for (r, a_r) in ancestry.iter().enumerate() {
        for (c, a_rc) in a_r.iter().enumerate() {
            if *a_rc == node {
                return (r, c);
            }
        }
    }
    panic!("Node not found in ancestry");
}

// Get the ancestry path of a tree node
fn get_ancestry_path_of_node(pairs: &Pairs, node: usize) -> Vec<usize> {
    let mut path = Vec::new();

    let mut current_node = node;
    let mut to_take = 0;

    if current_node > pairs.len() {
        // Skip the first `to_take` pairs (which are under `node``)
        to_take = node - pairs.len() - 1;
        current_node = pairs[node - pairs.len() - 1].0;
    }

    for (i, pair) in pairs.iter().enumerate().skip(to_take) {
        if pair.0 == current_node || pair.1 == current_node {
            path.push(pairs.len() + i + 1);
            current_node = pair.0;
        }
    }

    path
}

// Find the minimum common ancestor between two ancestry paths
fn min_common_ancestor(path1: &[usize], path2: &[usize]) -> usize {
    let mut i = 0;
    let mut j = 0;

    while i < path1.len() && j < path2.len() {
        match path1[i].cmp(&path2[j]) {
            Ordering::Equal => return path1[i],
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
        }
    }

    // Return the last element if no common value is found
    path1[path1.len() - 1]
}

/// Get the most recent common ancestor (MRCA) of two nodes in a Phylo2Vec vector.
/// node1 and node2 can be leaf nodes (0 to n_leaves) or internal nodes (n_leaves to 2*(n_leaves-1)).
///
/// Similar to `get_common_ancestor` in ETE (Python).
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::ops::get_common_ancestor;
///
/// let v = vec![0, 1, 2, 3, 4];
/// let mrca = get_common_ancestor(&v, 2, 3);
/// // Newick string of v = (0,(1,(2,(3,(4,5)6)7)8)9)10;
/// // So 2 and 3 share the MRCA 8.
/// assert_eq!(mrca, 8);
/// ```
pub fn get_common_ancestor(v: &[usize], node1: usize, node2: usize) -> usize {
    let node_max = 2 * v.len();

    if node1 > node_max || node2 > node_max {
        panic!("Node indices must be within the range of the Phylo2Vec vector. Max = {node_max}, got node1 = {node1} and node2 = {node2}");
    }

    if node1 == node2 {
        return node1;
    }

    let pairs = to_pairs(v);

    let path1 = get_ancestry_path_of_node(&pairs, node1);
    let path2 = get_ancestry_path_of_node(&pairs, node2);

    min_common_ancestor(&path1, &path2)
}

/// Generic function to calculate the depths of all nodes in a Phylo2Vec tree.
///
/// The depth of a node is the length of the path from the root to that node
/// (i.e., distance from root). This follows the BEAST/ETE convention.
///
/// The root has depth 0, and depths increase as you move toward the leaves.
///
/// When `bls` is `None`, all branch lengths are assumed to be 1.0 (topological depth).
///
/// # Returns
/// A vector of depths for all nodes (length = 2 * n_leaves - 1).
/// Index i contains the depth of node i.
pub fn _get_node_depths(v: &[usize], bls: Option<&Vec<[f64; 2]>>) -> Vec<f64> {
    let k = v.len();
    let n_leaves = k + 1;
    let n_nodes = 2 * n_leaves - 1;

    // Special case: single leaf tree (k=0)
    if k == 0 {
        return vec![0.0];
    }

    let ancestry = to_ancestry(v);

    // depths[node] = distance from root to node
    let mut depths = vec![0.0; n_nodes];

    // Root is the last parent in ancestry, has depth 0
    let root = ancestry[k - 1][2];
    depths[root] = 0.0;

    // Traverse ancestry in reverse order (top-down from root)
    // ancestry[i] = [c1, c2, parent], bls[i] = [bl_to_c1, bl_to_c2]
    for i in (0..k).rev() {
        let [c1, c2, parent] = ancestry[i];
        let [bl1, bl2] = match bls {
            Some(b) => b[i],
            None => [1.0, 1.0],
        };
        depths[c1] = depths[parent] + bl1;
        depths[c2] = depths[parent] + bl2;
    }

    depths
}

/// Generic function to calculate the depth of a single node in a Phylo2Vec tree.
///
/// The depth of a node is the length of the path from the root to that node
/// (i.e., distance from root). This follows the BEAST/ETE convention.
///
/// The root has depth 0, and depths increase as you move toward the leaves.
///
/// When `bls` is `None`, all branch lengths are assumed to be 1.0 (topological depth).
pub fn _get_node_depth(v: &[usize], bls: Option<&Vec<[f64; 2]>>, node: usize) -> f64 {
    let k = v.len();
    let n_leaves = k + 1;
    let n_nodes = 2 * n_leaves - 1;

    if node >= n_nodes {
        panic!(
            "Node index out of bounds. Max node = {}, got node = {}",
            n_nodes - 1,
            node
        );
    }

    _get_node_depths(v, bls)[node]
}

/// Get the depths of all nodes in a Phylo2Vec vector (topological).
///
/// The depth of a node is the length of the path from the root to that node.
/// The root has depth 0, and depths increase as you move toward the leaves.
///
/// For vectors, topological depth is returned (all branch lengths = 1).
///
/// # Returns
/// A vector of depths for all nodes (length = 2 * n_leaves - 1).
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::ops::get_node_depths;
///
/// // Tree: (0,(1,(2,3)4)5)6
/// let v = vec![0, 1, 2];
/// let depths = get_node_depths(&v);
/// // Root (node 6) has depth 0
/// assert_eq!(depths[6], 0.0);
/// // Node 5 is 1 edge from root
/// assert_eq!(depths[5], 1.0);
/// // Node 4 is 2 edges from root
/// assert_eq!(depths[4], 2.0);
/// // Leaf 0 is 1 edge from root
/// assert_eq!(depths[0], 1.0);
/// // Leaves 2 and 3 are 3 edges from root
/// assert_eq!(depths[2], 3.0);
/// assert_eq!(depths[3], 3.0);
/// ```
pub fn get_node_depths(v: &[usize]) -> Vec<f64> {
    _get_node_depths(v, None)
}

/// Get the depth of a node in a Phylo2Vec vector (topological).
///
/// The depth of a node is the length of the path from the root to that node.
/// The root has depth 0.
///
/// For vectors, topological depth is returned (all branch lengths = 1).
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::ops::get_node_depth;
///
/// // Tree: (0,(1,(2,3)4)5)6
/// let v = vec![0, 1, 2];
/// // Root (node 6) has depth 0
/// assert_eq!(get_node_depth(&v, 6), 0.0);
/// // Node 5 is 1 edge from root
/// assert_eq!(get_node_depth(&v, 5), 1.0);
/// // Node 4 is 2 edges from root
/// assert_eq!(get_node_depth(&v, 4), 2.0);
/// // Leaf 0 is 1 edge from root
/// assert_eq!(get_node_depth(&v, 0), 1.0);
/// ```
pub fn get_node_depth(v: &[usize], node: usize) -> f64 {
    _get_node_depth(v, None, node)
}

/// Produce an ordered version (i.e., birth-death process version)
/// of a Phylo2Vec vector using the Queue Shuffle algorithm.
///
/// Queue Shuffle ensures that the output tree is ordered,
/// while also ensuring a smooth path through the space of orderings
///
/// for more details, see <https://doi.org/10.1093/gbe/evad213>
///
/// Illustration of the algorithm with a simple case:
///                  ////-3
///            ////6|
///      ////7|      \\\\-2
///     |      \\\\-1
///   -8|
///     |
///     |      ////-4
///      \\\\5|
///            \\\\-0
///
///   The ancestry array of this tree is:
///   [[0, 4, 5]
///   [2, 3, 6]
///   [1, 6, 7]
///   [5, 7, 8]]
///
///   Unrolled from the bottom right, it becomes:
///   8 7 5 6 1 3 2 4 0
///
///   We encode the nodes as follows:
///   Start by encoding the first two non-root nodes as 0, 1
///   For the next pairs:
///    * The left member takes the label was the previous parent node
///    * The right member increments the previous right member by 1
///
///   8 7 5 6 1 3 2 4 0
///     0 1 0 2
///
///   (previous parent node = 7, encoded as 0)
///
///   then
///
///   8 7 5 6 1 3 2 4 0
///     0 1 0 2 1 3
///
///   (previous parent node = 5, encoded as 1)
///
///   then
///
///   8 7 5 6 1 3 2 4 0
///     0 1 0 2 1 3 0 4
///
///   (previous parent node = 6, encoded as 0)
///
///   The created sequence, viewed two by two
///   constitutes the reversed pairs of the Phylo2Vec vector:
///   ((0, 1), (0, 2), (1, 3), (0, 4))
///
///   Note that the full algorithm also features a queue of internal nodes
///   which could switch the processing order of rows in the ancestry array.
///
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::ops::queue_shuffle;
///
/// // Tree with 7 leaves
/// let v = vec![0, 0, 0, 2, 5, 3];
/// let (v_qs, label_mapping) = queue_shuffle(&v, false);
/// assert_eq!(v_qs, vec![0, 1, 1, 2, 3, 2]);
/// // Suppose that you originally have a list of string labels for taxa
/// // Initially, represent the n-th taxon by an integer leaf n
/// // In this scenario,
/// // Leaves 0, 1, 2, 3 remain unchanged
/// // The 5th taxon is now represented by 6
/// // The 6th taxon is now represented by 4
/// // The 7th taxon is now represented by 5
/// assert_eq!(label_mapping, vec![1, 5, 6, 4, 0, 2, 3]);
/// ```
pub fn queue_shuffle(v: &[usize], shuffle_cherries: bool) -> (Vec<usize>, Vec<usize>) {
    let ancestry = to_ancestry(v);

    let k = v.len();
    let n_leaves = k + 1;

    // Queue of internal nodes
    let mut queue = vec![2 * k];
    let mut j = 0;

    // Output pairs
    let mut new_pairs: Pairs = Vec::new();

    // Output mapping of leaves to their new labels (indices)
    let mut label_mapping: Vec<usize> = (0..n_leaves).collect();

    // Node code tracks the `code` of internal nodes
    let mut node_code = Vec::new();

    while new_pairs.len() < k {
        // Next row in the ancestry
        let [c2, c1, _] = ancestry[queue[j] - n_leaves];

        // `shuffle_cherries` allows to randomly permutate the order of children
        // useful in optimisation algorithms
        let child_order = if shuffle_cherries && rand::random() {
            [c2, c1]
        } else {
            [c1, c2]
        };

        let next_leaf = j + 1;

        // If the node code is empty, we are at the root
        // Otherwise, we take the 2nd previous node code
        let new_pair = if node_code.is_empty() {
            (0, 1)
        } else {
            (node_code[j - 1], next_leaf)
        };

        // Push the new pair to the output
        new_pairs.push(new_pair);

        // Process internal nodes
        for (i, &c) in child_order.iter().enumerate() {
            if c >= n_leaves {
                // Add internal nodes to the queue
                queue.push(c);
                // Encode internal nodes in the node code
                if i == 0 {
                    node_code.push(new_pair.0)
                } else {
                    node_code.push(new_pair.1);
                }
            }
        }

        // Process leaf nodes --> update the label mapping
        if child_order[1] < n_leaves {
            label_mapping[new_pair.1] = c2;
        }
        if child_order[0] < n_leaves {
            label_mapping[new_pair.0] = c1;
        }

        j += 1;
    }

    new_pairs.reverse();

    let v_qs = from_pairs(&new_pairs);

    // Check that the label mapping is unique
    {
        let unique: HashSet<_> = label_mapping.iter().copied().collect();
        assert_eq!(
            unique.len(),
            label_mapping.len(),
            "label_mapping elements must be unique"
        );
    }

    (v_qs, label_mapping)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    /// Test the addition of a new leaf to the tree
    ///
    /// Tests are using 6 leaf tree with different leaf and branch indices
    #[rstest]
    #[case(vec![0, 1, 2, 5, 4, 2], 5, 3, vec![0, 1, 2, 5, 3, 4, 2])]
    #[case(vec![0, 1, 2, 5, 4, 2], 7, 0, vec![0, 1, 2, 5, 4, 2, 0])]
    #[case(vec![0, 1, 2, 5, 4, 2], 7, 2, vec![0, 1, 2, 5, 4, 2, 2])]
    fn test_add_leaf(
        #[case] mut v: Vec<usize>,
        #[case] leaf: usize,
        #[case] branch: usize,
        #[case] expected: Vec<usize>,
    ) {
        let new_vec = add_leaf(&mut v, leaf, branch);
        assert_eq!(new_vec, expected);
    }

    /// Test the removal of a leaf from the tree
    ///
    /// Tests are using 6 leaf tree with different leaf and sister branch indices
    #[rstest]
    #[case(vec![0, 1, 2, 5, 4, 2], 5, 4, vec![0, 1, 2, 5, 2])]
    #[case(vec![0, 1, 2, 5, 4, 2], 6, 2, vec![0, 1, 2, 5, 4])]
    #[case(vec![0, 1, 2, 5, 4, 2], 0, 11, vec![0, 1, 4, 3, 1])]
    fn test_remove_leaf(
        #[case] mut v: Vec<usize>,
        #[case] leaf: usize,
        #[case] branch: usize,
        #[case] expected: Vec<usize>,
    ) {
        let (new_vec, sister) = remove_leaf(&mut v, leaf);
        assert_eq!(new_vec, expected);
        assert_eq!(sister, branch);
    }

    #[rstest]
    #[case(vec![[0, 1, 4], [4, 2, 5], [5, 3, 6]], 0, (0, 0))]
    #[case(vec![[0, 1, 4], [4, 2, 5], [5, 3, 6]], 1, (0, 1))]
    #[case(vec![[0, 1, 4], [4, 2, 5], [5, 3, 6]], 2, (1, 1))]
    #[case(vec![[0, 1, 4], [4, 2, 5], [5, 3, 6]], 4, (0, 2))]
    #[case(vec![[0, 1, 4], [4, 2, 5], [5, 3, 6]], 3, (2, 1))]
    fn test_argeq(
        #[case] ancestry: Ancestry,
        #[case] leaf: usize,
        #[case] expected_coords: (usize, usize),
    ) {
        let coords = argeq(&ancestry, leaf);
        assert_eq!(coords, expected_coords);
    }

    #[rstest]
    #[case(vec![[0, 1, 2]], 3)]
    #[case(vec![[0, 1, 4], [4, 2, 5], [5, 3, 6]], 7)]
    #[should_panic]
    fn test_argeq_panic(#[case] ancestry: Ancestry, #[case] leaf: usize) {
        argeq(&ancestry, leaf);
    }

    #[rstest]
    // (0,(1,(2,(3,(4,5)))));
    #[case(vec![0, 1, 2, 3, 4], 0, 1, 10)]
    #[case(vec![0, 1, 2, 3, 4], 4, 5, 6)]
    #[case(vec![0, 1, 2, 3, 4], 2, 3, 8)]
    #[case(vec![0, 1, 2, 3, 4], 2, 7, 8)]
    #[case(vec![0, 1, 2, 3, 4], 6, 1, 9)]
    // (((0,(3,5)6)8,2)9,(1,4)7)10;
    #[case(vec![0, 0, 0, 1, 3], 0, 2, 9)]
    #[case(vec![0, 0, 0, 1, 3], 1, 4, 7)]
    #[case(vec![0, 0, 0, 1, 3], 3, 5, 6)]
    #[case(vec![0, 0, 0, 1, 3], 2, 4, 10)]
    #[case(vec![0, 0, 0, 1, 3], 3, 8, 8)]
    // (0,((4,5)8,(1,(3,(2,6)7)))));
    #[case(vec![0, 1, 2, 5, 4, 2], 2, 6, 7)]
    #[case(vec![0, 1, 2, 5, 4, 2], 4, 5, 8)]
    #[case(vec![0, 1, 2, 5, 4, 2], 1, 3, 10)]
    #[case(vec![0, 1, 2, 5, 4, 2], 1, 4, 11)]
    #[case(vec![0, 1, 2, 5, 4, 2], 0, 2, 12)]
    #[case(vec![0, 1, 2, 5, 4, 2], 0, 1, 12)]
    #[case(vec![0, 1, 2, 5, 4, 2], 7, 4, 11)]
    #[case(vec![0, 2, 4, 0, 4, 0, 9, 14, 16, 3, 19, 18, 12, 24, 16, 22, 6, 4, 9], 17, 35, 35)]
    fn test_get_common_ancestor(
        #[case] v: Vec<usize>,
        #[case] node1: usize,
        #[case] node2: usize,
        #[case] expected_mrca: usize,
    ) {
        let mrca = get_common_ancestor(&v, node1, node2);
        assert_eq!(
            mrca, expected_mrca,
            "Expected mrca of nodes {node1} and {node2} for v = {v:?} to be {expected_mrca}, but got {mrca}"
        );
    }

    #[rstest]
    // Tree: (0,1)2 - simplest tree
    // Depth is distance from root to node
    #[case(vec![0], 2, 0.0)] // Root has depth 0
    #[case(vec![0], 0, 1.0)] // Leaf 0 is 1 edge from root
    #[case(vec![0], 1, 1.0)] // Leaf 1 is 1 edge from root
    // Asymmetric tree: (1,(2,(0,3)4)5)6;
    #[case(vec![0, 0, 0], 6, 0.0)] // Root has depth 0
    #[case(vec![0, 0, 0], 5, 1.0)] // Node 5 is 1 edge from root
    #[case(vec![0, 0, 0], 4, 2.0)] // Node 4 is 2 edges from root
    #[case(vec![0, 0, 0], 3, 3.0)] // Leaf 3 is 3 edges from root
    #[case(vec![0, 0, 0], 2, 2.0)] // Leaf 2 is 2 edges from root
    #[case(vec![0, 0, 0], 0, 3.0)] // Leaf 0 is 3 edges from root
    #[case(vec![0, 0, 0], 1, 1.0)] // Leaf 1 is 1 edge from root
    // Balanced tree: ((0,1)4,(2,3)5)6;
    #[case(vec![0, 0, 1], 6, 0.0)] // Root has depth 0
    #[case(vec![0, 0, 1], 4, 1.0)] // Node 4 is 1 edge from root
    #[case(vec![0, 0, 1], 5, 1.0)] // Node 5 is 1 edge from root
    #[case(vec![0, 0, 1], 0, 2.0)] // All leaves are 2 edges from root
    #[case(vec![0, 0, 1], 1, 2.0)]
    #[case(vec![0, 0, 1], 2, 2.0)]
    #[case(vec![0, 0, 1], 3, 2.0)]
    fn test_get_node_depth(
        #[case] v: Vec<usize>,
        #[case] node: usize,
        #[case] expected_depth: f64,
    ) {
        let depth = get_node_depth(&v, node);
        assert!(
            (depth - expected_depth).abs() < 1e-10,
            "Expected depth {expected_depth} for node {node} in v = {v:?}, got {depth}"
        );
    }

    #[rstest]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_get_node_depth_root_is_zero(#[case] n_leaves: usize) {
        use crate::vector::base::sample_vector;

        let v = sample_vector(n_leaves, false);
        let root = 2 * v.len(); // Root node index
        let depth = get_node_depth(&v, root);
        assert_eq!(depth, 0.0, "Root should have depth 0");
    }

    #[rstest]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_get_node_depths_returns_correct_length(#[case] n_leaves: usize) {
        use crate::vector::base::sample_vector;

        let v = sample_vector(n_leaves, false);
        let depths = get_node_depths(&v);
        let expected_len = 2 * n_leaves - 1;
        assert_eq!(
            depths.len(),
            expected_len,
            "Expected {} depths, got {}",
            expected_len,
            depths.len()
        );
    }

    #[rstest]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    fn test_get_node_depths_root_is_zero(#[case] n_leaves: usize) {
        use crate::vector::base::sample_vector;

        let v = sample_vector(n_leaves, false);
        let root = 2 * v.len();
        let depths = get_node_depths(&v);
        assert_eq!(depths[root], 0.0, "Root should have depth 0");
    }

    #[rstest]
    #[case(vec![0, 1, 2], 7)] // Max node is 6
    #[case(vec![0], 3)] // Max node is 2
    #[should_panic]
    fn test_get_node_depth_out_of_bounds(#[case] v: Vec<usize>, #[case] node: usize) {
        get_node_depth(&v, node);
    }

    #[rstest]
    #[case(vec![0], vec![0], vec![1, 0])]
    #[case(vec![0, 0], vec![0, 1], vec![1, 2, 0])]
    #[case(vec![0, 0, 0], vec![0, 1, 2], vec![1, 2, 3, 0])]
    #[case(vec![0, 2, 1, 3], vec![0, 1, 1, 1], vec![2, 4, 0, 1, 3])]
    #[case(vec![0, 0, 0, 2, 5, 3], vec![0, 1, 1, 2, 3, 2], vec![1, 5, 6, 4, 0, 2, 3])]
    fn test_queue_shuffle(
        #[case] v_input: Vec<usize>,
        #[case] v_expected: Vec<usize>,
        #[case] mapping_expected: Vec<usize>,
    ) {
        let (v_qs, mapping_qs) = queue_shuffle(&v_input, false);
        assert_eq!(
            v_qs, v_expected,
            "Expected queue shuffled vector to be {v_expected:?}, but got {v_qs:?}"
        );

        assert_eq!(
            mapping_qs, mapping_expected,
            "Expected queue shuffled mapping to be {mapping_expected:?}, but got {mapping_qs:?}"
        );
    }
}
