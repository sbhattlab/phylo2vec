/// Functions to convert Phylo2Vec vectors to other tree representations
use std::collections::HashMap;

use crate::newick::{has_parents, parse};
use crate::types::{Ancestry, Pair, Pairs};
use crate::vector::avl::AVLTree;
use crate::vector::base::is_unordered;

/// Retrieve a Phylo2Vec vector from a list of pairs.
/// A pair is a tuple of nodes (c1, c2) where:
///   - c1 denotes the node from where the branch leading to c2 originates
///   - c2 is the leaf node
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::convert::from_pairs;
///
/// let pairs = vec![(1, 4), (0, 3), (0, 2), (0, 1)];
/// let v = from_pairs(&pairs);
/// assert_eq!(v, vec![0, 0, 0, 1]);
/// ```
pub fn from_pairs(pairs: &Pairs) -> Vec<usize> {
    let mut cherries: Ancestry = Vec::with_capacity(pairs.len());
    for &(c1, c2) in pairs.iter() {
        cherries.push([c1, c2, std::cmp::max(c1, c2)]);
    }

    build_vector(&cherries)
}

/// Get all "pairs" from the Phylo2Vec vector
/// using a for loop implementation.
/// Pair = (B, L)
/// B = branch leading to leaf L
/// Best case is O(n) for ordered trees.
/// Average case is O(n^1.5)
/// Worst case is O(n^2) for unordered trees.
fn to_pairs_loop(v: &[usize]) -> Pairs {
    let k: usize = v.len();
    let mut pairs: Pairs = Vec::with_capacity(k);

    // First loop (reverse iteration)
    for i in (0..k).rev() {
        /*
        If v[i] <= i, it's like a birth-death process.
        The next pair to add is (v[i], next_leaf) as the branch leading to v[i]
        gives birth to next_leaf.
        */
        let next_leaf: usize = i + 1;
        let pair: Pair = (v[i], next_leaf);
        if v[i] <= i {
            pairs.push(pair);
        }
    }

    // Second loop
    for (j, &vj) in v.iter().enumerate().skip(1) {
        let next_leaf = j + 1;
        if vj == 2 * j {
            // 2 * j = extra root ==> pairing = (0, next_leaf)
            let pair: Pair = (0, next_leaf);
            pairs.push(pair);
        } else if vj > j {
            /*
            If v[j] > j, it's not the branch leading to v[j] that gives birth,
            but an internal branch. Insert at the calculated index.
            */
            let index: usize = pairs.len() + vj - 2 * j;
            let new_pair: Pair = (pairs[index - 1].0, next_leaf);
            pairs.insert(index, new_pair);
        }
    }

    pairs
}

/// Get all node pairs from the Phylo2Vec vector
/// using an AVL tree implementation.
/// Pair = (B, L)
/// B = branch leading to leaf L
/// Complexity: O(n log n) for all trees
fn to_pairs_avl(v: &[usize]) -> Pairs {
    let avl_tree: AVLTree = AVLTree::with_vector(v);

    avl_tree.inorder_traversal()
}

/// Get the pair of nodes from the Phylo2Vec vector
/// The implementation is determined by the ordering of v.
/// AVL trees are faster for unordered trees.
/// A simple for loop is faster for ordered trees.
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::convert::to_pairs;
///
/// let v = vec![0, 0, 0, 1];
/// let pairs = to_pairs(&v);
/// assert_eq!(pairs, vec![(1, 4), (0, 3), (0, 2), (0, 1)]);
/// ```
pub fn to_pairs(v: &[usize]) -> Pairs {
    if is_unordered(v) {
        to_pairs_avl(v)
    } else {
        to_pairs_loop(v)
    }
}

/// Retrieve a Phylo2Vec vector from an ancestry representation.
/// The ancestry is a list of cherries, where each cherry is represented as
/// a triplet [c1, c2, p], where:
///     - c1 and c2 are the indices of the leaves in the cherry
///     - p is the index of the parent node of the cherry
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::convert::from_ancestry;
/// let ancestry = vec![
///    [2, 3, 4], // Cherry with leaves 0 and 1, parent 4
///    [0, 1, 5], // Cherry with leaves 0 and 2, parent 5
///    [5, 4, 6], // Cherry with nodes 4 and 5, parent 6
/// ];
/// let v = from_ancestry(&ancestry);
/// assert_eq!(v, vec![0, 2, 2]);
/// ```
pub fn from_ancestry(ancestry: &Ancestry) -> Vec<usize> {
    let mut cherries = ancestry.clone();

    let row_idxs: Vec<usize> = (0..ancestry.len()).collect();
    build_cherries(&mut cherries, &row_idxs);

    build_vector(&cherries)
}

/// Get the ancestry of the Phylo2Vec vector
/// v\[i\] = which BRANCH we do the pairing from
///
/// The initial situation looks like this:
///                  R
///                  |
///                  | --> branch 2
///                // \\
///  branch 0 <-- //   \\  --> branch 1
///               0     1
///
/// For v\[1\], we have 3 possible branches to choose from.
/// v\[1\] = 0 or 1 indicates that we branch out from branch 0 or 1, respectively.
/// The new branch yields leaf 2 (like in ordered trees)
///
/// v\[1\] = 2 is somewhat similar: we create a new branch from R that yields leaf 2.
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::convert::to_ancestry;
/// let v = vec![0, 2, 2];
/// let ancestry = to_ancestry(&v);
/// assert_eq!(ancestry, vec![
///   [2, 3, 4], // Cherry with leaves 2 and 3, parent 4
///   [0, 1, 5], // Cherry with leaves 0 and 1, parent 5
///   [5, 4, 6], // Cherry with nodes 4 and 5, parent 6
/// ]);
/// ```
pub fn to_ancestry(v: &[usize]) -> Ancestry {
    let k = v.len();
    let pairs: Pairs = to_pairs(v);

    let mut ancestry: Ancestry = Vec::with_capacity(k);

    // Keep track of child->highest parent relationship
    let mut parents: Vec<usize> = (0..=(2 * k + 1)).collect();

    for (i, &(c1, c2)) in pairs.iter().enumerate() {
        let next_parent = k + 1 + i;

        // Push the current cherry to ancestry
        ancestry.push([parents[c1], parents[c2], next_parent]);

        // Update the parents of current children
        parents[c1] = next_parent;
        parents[c2] = next_parent;
    }

    ancestry
}

/// Retrieve a Phylo2Vec vector from a list of edges.
/// An edge is a tuple of nodes (child, parent).
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::convert::from_edges;
/// let edges = vec![(2, 4), (3, 4), (0, 5), (1, 5), (4, 6), (5, 6)];
/// let v = from_edges(&edges);
/// assert_eq!(v, vec![0, 2, 2]);
/// ```
pub fn from_edges(edges: &[(usize, usize)]) -> Vec<usize> {
    assert!(edges.len() % 2 == 0, "The number of edges must be even");

    let mut ancestry: Ancestry = Vec::with_capacity(edges.len() / 2);

    for i in (0..edges.len()).step_by(2) {
        ancestry.push([edges[i].0, edges[i + 1].0, edges[i].1]);
    }

    order_cherries(&mut ancestry);

    build_vector(&ancestry)
}

/// Convert a Phylo2Vec vector to a list of edges.
/// An edge is a tuple of nodes (child, parent).
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::convert::to_edges;
///
/// let v = vec![0, 2, 2];
/// let edges = to_edges(&v);
/// assert_eq!(edges, vec![(2, 4), (3, 4), (0, 5), (1, 5), (5, 6), (4, 6)]);
/// ```
pub fn to_edges(v: &[usize]) -> Vec<(usize, usize)> {
    let k = v.len();
    let pairs: Pairs = to_pairs(v);

    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(2 * k);

    // Keep track of child->highest parent relationship
    let mut parents: Vec<usize> = (0..=(2 * k + 1)).collect();

    for (i, &(c1, c2)) in pairs.iter().enumerate() {
        let next_parent = k + 1 + i;

        // Push current edges
        edges.push((parents[c1], next_parent));
        edges.push((parents[c2], next_parent));

        // Update the parents of current children
        parents[c1] = next_parent;
        parents[c2] = next_parent;
    }
    edges
}

/// Retrieve a Phylo2Vec vector from a Newick string.
/// The Newick string should only describe a tree topology
/// with or without parental labels.
/// For a tree with n leaves, leaves are labelled 0 to n-1,
/// and internal nodes are labelled n to 2 * (n-1).
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::convert::from_newick;
/// let newick = "((0,1)5,(2,3)4)6;";
/// let v = from_newick(newick);
/// assert_eq!(v, vec![0, 2, 2]);
///
/// let newick_no_parents = "((0,1),(2,3));";
/// let v2 = from_newick(newick_no_parents);
/// assert_eq!(v2, vec![0, 2, 2]);
/// ```
pub fn from_newick(newick: &str) -> Vec<usize> {
    let mut ancestry: Ancestry = parse(newick).expect("failed to get cherries");

    if has_parents(newick) {
        order_cherries(&mut ancestry);
    } else {
        order_cherries_no_parents(&mut ancestry);
    }

    build_vector(&ancestry)
}

/// A Fenwick Tree (Binary Indexed Tree) for efficiently calculating prefix sums
/// and updating values in logarithmic time complexity. See https://en.wikipedia.org/wiki/Fenwick_tree
///
/// Fenwick trees support two primary operations:
/// - **update:** Increment the value at a specific position
/// - **prefix_sum:** Calculate the cumulative sum up to a given position
struct Fenwick {
    n_leaves: usize,  // The number of leaves in the tree
    data: Vec<usize>, // 1-indexed array implicitly representing the tree
}

impl Fenwick {
    fn new(n: usize) -> Self {
        Fenwick {
            n_leaves: n,
            data: vec![0; n + 1],
        }
    }

    // Sum of [1..=i]
    fn prefix_sum(&self, mut i: usize) -> usize {
        let mut sum = 0;
        while i > 0 {
            sum += self.data[i];
            i -= i & i.wrapping_neg(); // i -= i & -i
        }
        sum
    }

    // Add delta at index i (1..=n)
    fn update(&mut self, mut i: usize, delta: usize) {
        while i <= self.n_leaves {
            self.data[i] += delta;
            i += i & i.wrapping_neg(); // i += i & -i
        }
    }
}

/// Utility function to build a Phylo2Vec vector from an intermediate
/// `cherry` representation.
pub fn build_vector(cherries: &Ancestry) -> Vec<usize> {
    let num_cherries = cherries.len();

    let mut v = vec![0; num_cherries];
    let mut bit = Fenwick::new(num_cherries);

    for [c1, c2, c_max] in cherries.iter().copied() {
        let idx = bit.prefix_sum(c_max - 1);

        v[c_max - 1] = if idx == 0 {
            std::cmp::min(c1, c2)
        } else {
            c_max - 1 + idx
        };
        bit.update(c_max, 1);
    }
    v
}

/// Prepare a vector cache to build a Newick string
/// from a vector of pairs.
pub fn prepare_cache(pairs: &Pairs) -> Vec<String> {
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

/// Build a Newick string from a vector of pairs
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

/// Build a Newick string from a Phylo2Vec vector.
/// The Newick string formed by a vector is a tree topology
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::convert::to_newick;
/// let v = vec![0, 2, 2];
/// let newick = to_newick(&v);
/// assert_eq!(newick, "((0,1)5,(2,3)4)6;");
/// ```
pub fn to_newick(v: &[usize]) -> String {
    let pairs: Pairs = to_pairs(v);
    build_newick(&pairs)
}

fn build_cherries(ancestry: &mut Ancestry, row_idxs: &[usize]) -> Vec<usize> {
    let num_cherries = ancestry.len();
    let num_nodes = 2 * num_cherries + 2;
    let mut min_desc = vec![usize::MAX; num_nodes];
    let mut bl_rows_to_swap: Vec<usize> = Vec::with_capacity(num_cherries);

    for (i, cherry) in ancestry.iter_mut().enumerate() {
        let [c1, c2, p] = *cherry;
        // Get the minimum descendant of c1 and c2 (if they exist)
        // min_desc[child_x] doesn't exist, min_desc_x --> child_x
        let min_desc1 = if min_desc[c1] != usize::MAX {
            min_desc[c1]
        } else {
            c1
        };
        let min_desc2 = if min_desc[c2] != usize::MAX {
            min_desc[c2]
        } else {
            c2
        };

        let (desc_min, desc_max) = if min_desc1 < min_desc2 {
            (min_desc1, min_desc2)
        } else {
            // Swap the branch lengths if min_desc1 > min_desc2
            // This is important to make the matrix formulation
            // invariant to cherry permutations
            // Ex: ((1:0.1,2:0.2)3:0.3); and ((2:0.2,1:0.1)3:0.3)
            // should yield the same matrix
            bl_rows_to_swap.push(row_idxs[i]);
            (min_desc2, min_desc1)
        };

        // Allocate the smallest descendant to min_desc[parent}
        min_desc[p] = desc_min;

        // Allocate the largest descendant as the "parent"
        *cherry = [min_desc1, min_desc2, desc_max];
    }

    bl_rows_to_swap
}

/// Order cherries in an ancestry vector.
/// The goal of this function is to find indices for an argsort to sort the cherries
/// We utilise the parent nodes to sort the cherries.
/// Returns two vectors:
/// 1. `row_idxs`: the indices of the sorted cherries
/// 2. `bl_rows_to_swap`: the indices of the cherries that need to swap their branch lengths
///    The latter is important to ensure bijectivity of the matrix object.
pub fn order_cherries(ancestry: &mut Ancestry) -> (Vec<usize>, Vec<usize>) {
    let num_cherries = ancestry.len();

    // Set offset as max(ancestry[, 2]) - 2 * num_cherries + 1
    // offset = 1 for most cases, except add_leaf where it is 2
    // as we update the ancestry with a new leaf
    let offset: usize = ancestry
        .iter()
        .map(|x| x[2])
        .max()
        .expect("No max found. Malformed ancestry?")
        - 2 * num_cherries
        + 1;

    let mut row_idxs: Vec<usize> = (0..num_cherries).collect();

    let mut new_ancestry: Ancestry = ancestry.clone();
    for (i, cherry) in ancestry.iter().enumerate() {
        // Swap rows in the new ancestry according to the parental label (cherry[2])
        let idx = cherry[2] - num_cherries - offset;
        new_ancestry[idx] = *cherry;
        row_idxs[idx] = i;
    }
    *ancestry = new_ancestry;

    let bl_rows_to_swap = build_cherries(ancestry, &row_idxs);

    (row_idxs, bl_rows_to_swap)
}

/// Order cherries in an ancestry vector without parent labels.
/// The goal of this function is to find indices for an argsort to sort the cherries
/// We utilise the fact that the input contains certain local orders, but not the global order.
/// Returns two vectors:
/// 1. `row_idxs`: the indices of the sorted cherries
/// 2. `bl_rows_to_swap`: the indices of the cherries that need to swap their branch lengths
///    The latter is important to ensure bijectivity of the matrix object.
///
/// Example:
/// [[6 7 7]
///  [6 9 9]
///  [0 6 6]
///  [0 3 3]
///  [0 2 2]
///  [0 8 8]
///  [1 5 5]
///  [1 4 4]
///  [0 1 1]]
/// (6, 7) will become before (6, 9). In Newick terms, one could write a partial Newick as such: ((6, 7), 9);
/// In other words, (6, 7) will form a terminal cherry, and that 9 will be paired with the parent of 6 and 7
///
/// In a similar fashion, (0, 6) comes before (0, 3), (0, 2), (0, 8).
/// And (1, 5) comes before (1, 4).
///
///
/// So we apply the following rule for each cherry:
///  * Determine the minimum (c_min) and maximum (c_max) cherry values.
///  * if c_min was visited beforehand:
///    * its "sorting index" will be the minimum of its previous sister node (visited\[c_min\]) and the current sister node (c_max)
///  * otherwise, set it to c_max
///
/// We thus obtain a sorting index list (```leaves```) as such
/// c1, c2, c_max, index
/// 6,   7,     7, 7
/// 6,   9,     9, 7 (will come after 7)
/// 0,   6,     6, 6
/// 0,   3,     3, 3
/// 0,   2,     2, 2
/// 0,   8,     8, 2
/// 1,   5,     5, 5
/// 1,   4,     4, 4
/// 0,   1,     1, 1
///
/// To order the cherries, we sort (in descending order) the input according to the index.
///
/// In this example, we get:
/// 6 7 7
/// 6 9 9
/// 0 6 6
/// 1 5 5
/// 1 4 4
/// 0 3 3
/// 0 2 2
/// 0 8 8
/// 0 1 1
pub fn order_cherries_no_parents(ancestry: &mut Ancestry) -> (Vec<usize>, Vec<usize>) {
    let num_cherries = ancestry.len();
    let mut to_sort: Vec<usize> = Vec::with_capacity(num_cherries);
    let mut visited: HashMap<usize, usize> = HashMap::new();

    let mut bl_rows_to_swap: Vec<usize> = Vec::with_capacity(num_cherries);

    for &[c1, c2, c_max] in ancestry.iter() {
        let c_min = c1.min(c2);
        let sister = match visited.get(&c_min) {
            Some(&existing) if existing < c_max => existing,
            _ => c_max,
        };

        to_sort.push(sister);
        visited.insert(c_min, sister);
    }

    // argsort with descending order
    // Note: using a stable sort is important to keep the order of cherries
    let mut row_idxs: Vec<usize> = (0..num_cherries).collect();
    row_idxs.sort_by_key(|&i| std::cmp::Reverse(to_sort[i]));

    let mut temp = Ancestry::with_capacity(num_cherries);
    for i in &row_idxs {
        // Swap the branch lengths if c1 > c2
        // This is important to make the matrix formulation
        // invariant to cherry permutations
        // Ex: ((1:0.1,2:0.2):0.3); and ((2:0.2,1:0.1):0.3)
        // should yield the same matrix
        let [c1, c2, _] = ancestry[*i];
        if c1 > c2 {
            bl_rows_to_swap.push(*i);
        }
        temp.push(ancestry[*i]);
    }
    *ancestry = temp;

    (row_idxs, bl_rows_to_swap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::base::sample_vector;
    use rstest::*;

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_ancestry(#[case] num_leaves: usize) {
        let v = sample_vector(num_leaves, false);

        let ancestry = to_ancestry(&v);
        let v2 = from_ancestry(&ancestry);

        assert_eq!(v, v2);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_pairs_loop_vs_avl(#[case] num_leaves: usize) {
        let v = sample_vector(num_leaves, false);

        let pairs_loop = to_pairs_loop(&v);
        let pairs_avl = to_pairs_avl(&v);

        assert_eq!(pairs_loop, pairs_avl);

        let v_ordered = sample_vector(num_leaves, true);

        let pairs_loop_ordered = to_pairs_loop(&v_ordered);
        let pairs_avl_ordered = to_pairs_avl(&v_ordered);

        assert_eq!(pairs_loop_ordered, pairs_avl_ordered);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_pairs(#[case] num_leaves: usize) {
        let v = sample_vector(num_leaves, false);

        let pairs = to_pairs(&v);
        let v2 = from_pairs(&pairs);

        assert_eq!(v, v2);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_edges(#[case] num_leaves: usize) {
        let v = sample_vector(num_leaves, false);

        let edges = to_edges(&v);
        let v2 = from_edges(&edges);

        assert_eq!(v, v2);
    }

    #[rstest]
    #[case(10)]
    #[case(100)]
    #[case(1000)]
    fn test_newick(#[case] n_leaves: usize) {
        // Sample a vector
        let v = sample_vector(n_leaves, false);

        // Convert to Newick
        let newick = to_newick(&v);

        // Convert back to vector
        let v2 = from_newick(&newick);

        // Check if the original and converted vectors are equal
        assert_eq!(v, v2);
    }

    /// Test the conversion of a Newick string to a vector
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)6)8,2)9,(1,4)7)10;")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)6)7)8)9)10;")]
    #[case(vec![0, 0, 1], "((0,2)5,(1,3)4)6;")]
    fn test_from_newick(#[case] expected: Vec<usize>, #[case] newick: &str) {
        let vector = from_newick(newick);
        assert_eq!(vector, expected);
    }

    /// Test the conversion of a Newick string without parents to a vector
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)),2),(1,4));")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)))));")]
    #[case(vec![0, 0, 1], "((0,2),(1,3));")]
    fn test_from_newick_no_parents(#[case] expected: Vec<usize>, #[case] newick: &str) {
        let vector = from_newick(newick);
        assert_eq!(vector, expected);
    }

    #[rstest]
    #[case(vec![(2, 4), (1, 3), (0, 1), (0, 2)], vec![String::from("((0"), String::from("(1"), String::from("(2"), String::from("3"), String::from("4")])]
    #[case(vec![(0, 1), (0, 2), (0, 3), (0, 4)], vec![String::from("((((0"), String::from("1"), String::from("2"), String::from("3"), String::from("4")])]
    fn test_prepare_cache(#[case] pairs: Pairs, #[case] expected: Vec<String>) {
        let cache = prepare_cache(&pairs);
        assert_eq!(cache, expected);
    }

    #[rstest]
    #[case(vec![(2, 4), (1, 3), (0, 1), (0, 2)], "((0,(1,3)6)7,(2,4)5)8;")]
    #[case(vec![(0, 1), (0, 2), (0, 3), (0, 4)], "((((0,1)5,2)6,3)7,4)8;")]
    fn test_build_newick(#[case] pairs: Pairs, #[case] expected: &str) {
        let newick = build_newick(&pairs);
        assert_eq!(newick, expected);
    }

    // Test order_cherries
    #[rstest]
    #[case(vec![
        [0, 3, 6],
        [0, 2, 5],
        [0, 1, 4],
    ],
    vec![
       [0, 1, 1],
       [0, 2, 2],
       [0, 3, 3]
    ])]
    #[case(vec![
        [0, 1, 5],
        [0, 3, 4],
        [0, 2, 6],
    ],
    vec![
       [0, 3, 3],
       [0, 1, 1],
       [0, 2, 2]
    ])]
    fn test_order_cherries(#[case] mut cherries: Ancestry, #[case] expected: Ancestry) {
        order_cherries(&mut cherries);
        assert_eq!(cherries, expected);
    }

    // Test order_cherries_no_parents
    #[rstest]
    #[case(vec![
        [6, 7, 7],
        [6, 9, 9],
        [0, 6, 6],
        [0, 3, 3],
        [0, 2, 2],
        [0, 8, 8],
        [1, 5, 5],
        [1, 4, 4],
        [0, 1, 1]
    ],
    vec![
       [6, 7, 7],
       [6, 9, 9],
       [0, 6, 6],
       [1, 5, 5],
       [1, 4, 4],
       [0, 3, 3],
       [0, 2, 2],
       [0, 8, 8],
       [0, 1, 1]
    ])]
    fn test_order_cherries_no_parents(#[case] mut cherries: Ancestry, #[case] expected: Ancestry) {
        order_cherries_no_parents(&mut cherries);
        assert_eq!(cherries, expected);
    }

    #[rstest]
    #[case(vec![
        [0, 3, 3],
        [0, 2, 2],
        [0, 1, 1]
    ],
    vec![0, 0, 0])]
    #[case(vec![
        [2, 3, 3],
        [1, 2, 2],
        [0, 1, 1]
    ],
    vec![0, 1, 2])]
    #[case(vec![
        [6, 7, 7],
        [6, 9, 9],
        [0, 6, 6],
        [1, 5, 5],
        [1, 4, 4],
        [0, 3, 3],
        [0, 2, 2],
        [0, 8, 8],
        [0, 1, 1]
     ],
        vec![0, 0, 0, 1, 1, 0, 6, 13, 9]
    )]
    fn test_build_vector(#[case] cherries: Ancestry, #[case] expected: Vec<usize>) {
        let v = build_vector(&cherries);
        assert_eq!(v, expected);
    }

    #[test]
    fn test_fenwick_prefix_sum_and_update() {
        let mut fenw = Fenwick::new(10);
        // Initially, all prefix sums should be 0.
        for i in 1..=10 {
            assert_eq!(fenw.prefix_sum(i), 0);
        }
        // Update index 1 with +5.
        fenw.update(1, 5);
        assert_eq!(fenw.prefix_sum(1), 5);
        assert_eq!(fenw.prefix_sum(2), 5);

        // Update index 3 with +3.
        fenw.update(3, 3);
        // The prefix sum up to index 3 should now be 5 (from index 1) + 0 (index 2) + 3 = 8.
        assert_eq!(fenw.prefix_sum(3), 8);

        // Update index 10 with +2.
        fenw.update(10, 2);
        // Total prefix sum at index 10 should include all previous updates.
        assert_eq!(fenw.prefix_sum(10), 10);
    }

    #[test]
    fn test_fenwick_multiple_updates() {
        let mut fenw = Fenwick::new(5);
        // Apply a set of updates.
        let updates = vec![(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)];
        for (index, delta) in updates {
            fenw.update(index, delta);
        }
        // Expected cumulative sums: 1, 1+2=3, 1+2+3=6, 1+2+3+4=10, 1+2+3+4+5=15.
        assert_eq!(fenw.prefix_sum(1), 1);
        assert_eq!(fenw.prefix_sum(2), 3);
        assert_eq!(fenw.prefix_sum(3), 6);
        assert_eq!(fenw.prefix_sum(4), 10);
        assert_eq!(fenw.prefix_sum(5), 15);
    }

    #[test]
    fn test_fenwick_updates_and_queries_complex() {
        let mut fenw = Fenwick::new(8);
        // Perform a series of updates.
        fenw.update(2, 3);
        fenw.update(4, 2);
        fenw.update(7, 4);

        // Check prefix sums at different indices:
        // Index 1: 0
        // Index 2: 3
        // Index 3: 3
        // Index 4: 5 (3 from index 2 + 2 from index 4)
        // Index 5 & 6: still 5
        // Index 7: 5 + 4 = 9
        // Index 8: 9
        assert_eq!(fenw.prefix_sum(1), 0);
        assert_eq!(fenw.prefix_sum(2), 3);
        assert_eq!(fenw.prefix_sum(3), 3);
        assert_eq!(fenw.prefix_sum(4), 5);
        assert_eq!(fenw.prefix_sum(5), 5);
        assert_eq!(fenw.prefix_sum(6), 5);
        assert_eq!(fenw.prefix_sum(7), 9);
        assert_eq!(fenw.prefix_sum(8), 9);
    }
}
