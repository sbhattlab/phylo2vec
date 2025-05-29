use crate::tree_vec::ops::avl::AVLTree;
use crate::tree_vec::types::{Ancestry, Pair, Pairs};
use crate::utils::is_unordered;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Get all "pairs" from the Phylo2Vec vector
/// using a for loop implementation.
/// Pair = (B, L)
/// B = branch leading to leaf L
/// Best case is O(n) for ordered trees.
/// Average case is O(n^1.5)
/// Worst case is O(n^2) for unordered trees.
fn get_pairs_loop(v: &[usize]) -> Pairs {
    let num_of_leaves: usize = v.len();
    let mut pairs: Pairs = Vec::with_capacity(num_of_leaves);

    // First loop (reverse iteration)
    for i in (0..num_of_leaves).rev() {
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
    for (j, &vj) in v.iter().enumerate().take(num_of_leaves).skip(1) {
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

pub fn make_tree(v: &[usize]) -> AVLTree {
    let mut avl_tree = AVLTree::new();
    let k = v.len();
    avl_tree.insert(0, (0, 1));

    for (i, &vi) in v.iter().enumerate().take(k).skip(1) {
        let next_leaf = i + 1;
        if vi <= i {
            avl_tree.insert(0, (v[i], next_leaf));
        } else {
            let index = v[i] - next_leaf;
            let pair = AVLTree::lookup(&avl_tree, index);
            avl_tree.insert(index + 1, (pair.0, next_leaf));
        }
    }

    avl_tree
}

/// Get all node pairs from the Phylo2Vec vector
/// using an AVL tree implementation.
/// Pair = (B, L)
/// B = branch leading to leaf L
/// Complexity: O(n log n) for all trees
fn get_pairs_avl(v: &[usize]) -> Pairs {
    let avl_tree: AVLTree = make_tree(v);

    avl_tree.inorder_traversal()
}

pub fn from_pairs(pairs: &Pairs) -> Vec<usize> {
    let mut cherries: Ancestry = Vec::with_capacity(pairs.len());
    for &(c1, c2) in pairs.iter() {
        cherries.push([c1, c2, std::cmp::max(c1, c2)]);
    }

    order_cherries_no_parents(&mut cherries);

    build_vector(&cherries)
}

/// Get the pair of nodes from the Phylo2Vec vector
/// The implementation is determined by the ordering of v.
/// AVL trees are faster for unordered trees.
/// A simple for loop is faster for ordered trees.
///
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::vector::get_pairs;
///
/// let v = vec![0, 0, 0, 1, 3, 3, 1, 4, 4];
/// let pairs = get_pairs(&v);
/// ```
pub fn get_pairs(v: &[usize]) -> Pairs {
    if is_unordered(v) {
        get_pairs_avl(v)
    } else {
        get_pairs_loop(v)
    }
}

/// Get the ancestry of the Phylo2Vec vector
/// v[i] = which BRANCH we do the pairing from
///
/// The initial situation looks like this:
///                  R
///                  |
///                  | --> branch 2
///                // \\
///  branch 0 <-- //   \\  --> branch 1
///               0     1
///
/// For v[1], we have 3 possible branches too choose from.
/// v[1] = 0 or 1 indicates that we branch out from branch 0 or 1, respectively.
/// The new branch yields leaf 2 (like in ordered trees)
///
/// v[1] = 2 is somewhat similar: we create a new branch from R that yields leaf 2
pub fn get_ancestry(v: &[usize]) -> Ancestry {
    let pairs: Pairs = get_pairs(v);
    let k = v.len();

    // Initialize Ancestry with capacity `k`
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

pub fn from_ancestry(ancestry: &Ancestry) -> Vec<usize> {
    let mut ordered_ancestry: Ancestry = ancestry.clone();
    order_cherries(&mut ordered_ancestry);

    build_vector(&ordered_ancestry)
}

pub fn get_edges_from_pairs(pairs: &Pairs) -> Vec<(usize, usize)> {
    let k = pairs.len();

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

pub fn get_edges(v: &[usize]) -> Vec<(usize, usize)> {
    let pairs: Pairs = get_pairs(v);

    get_edges_from_pairs(&pairs)
}

pub fn from_edges(edges: &[(usize, usize)]) -> Vec<usize> {
    assert!(edges.len() % 2 == 0, "The number of edges must be even");

    let mut ancestry: Ancestry = Vec::with_capacity(edges.len() / 2);

    for i in (0..edges.len()).step_by(2) {
        ancestry.push([edges[i].0, edges[i + 1].0, edges[i].1]);
    }

    order_cherries(&mut ancestry);

    build_vector(&ancestry)
}

pub fn find_coords_of_first_leaf(ancestry: &Ancestry, leaf: usize) -> (usize, usize) {
    for (r, a_r) in ancestry.iter().enumerate() {
        for (c, a_rc) in a_r.iter().enumerate() {
            if *a_rc == leaf {
                return (r, c);
            }
        }
    }
    panic!("Leaf not found in ancestry");
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
    let num_nodes = 2 * num_cherries + 2;

    let mut min_desc = vec![usize::MAX; num_nodes];

    let mut row_idxs: Vec<usize> = (0..num_cherries).collect();
    row_idxs.sort_by_key(|&i| ancestry[i][2]);

    // Sort by the parent node (ascending order)
    let mut new_ancestry: Ancestry = Vec::with_capacity(num_cherries);
    for i in &row_idxs {
        new_ancestry.push(ancestry[*i]);
    }
    *ancestry = new_ancestry;

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
///    * its "sorting index" will be the minimum of its previous sister node (visited[c_min]) and the current sister node (c_max)
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
    let mut row_idxs: Vec<usize> = (0..num_cherries).collect();
    row_idxs.sort_by_key(|&i| std::cmp::Reverse(to_sort[i]));

    // Note: using a stable sort is important to keep the order of cherries
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

pub fn build_vector(cherries: &Ancestry) -> Vec<usize> {
    let num_cherries = cherries.len();
    let num_leaves = num_cherries + 1;

    let mut v = vec![0; num_cherries];
    let mut bit = Fenwick::new(num_leaves);

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

/// Generic function to calculate the cophenetic distance of a tree from
/// a Phylo2Vec vector with or without branch lengths.
/// Output is a pairwise distance matrix of dimensions n x n
/// where n = number of leaves.
/// Each distance corresponds to the sum of the branch lengths between two leaves
/// Inspired from the `cophenetic` function in the `ape` package: https://github.com/emmanuelparadis/ape
pub fn _cophenetic_distances(v: &[usize], bls: Option<&Vec<[f32; 2]>>) -> Vec<Vec<f32>> {
    let k = v.len();
    let ancestry = get_ancestry(v);

    let bls = match bls {
        Some(b) => b.to_vec(),
        None => vec![[1.0; 2]; v.len()],
    };

    // Note: unrooted option was removed.
    // Originally implemented to match tr.unroot() in ete3
    // But now prefer to operate such that unrooting
    // preserves total branch lengths (compatible with ape,
    // and ete3, see https://github.com/etetoolkit/ete/pull/344)

    // Dist shape: N_nodes x N_nodes
    let mut dist: Vec<Vec<f32>> = vec![vec![0.0; 2 * k + 1]; 2 * k + 1];
    let mut all_visited: Vec<usize> = Vec::new();

    for i in 0..k {
        let [c1, c2, p] = ancestry[k - i - 1];
        let [bl1, bl2] = bls[k - i - 1];

        if !all_visited.is_empty() {
            for &visited in &all_visited[0..all_visited.len() - 1] {
                let dist1 = dist[p][visited] + bl1;
                let dist2 = dist[p][visited] + bl2;

                dist[c1][visited] = dist1;
                dist[visited][c1] = dist1;
                dist[c2][visited] = dist2;
                dist[visited][c2] = dist2;
            }
        }

        dist[c1][c2] = bl1 + bl2;
        dist[c2][c1] = bl1 + bl2;

        dist[c1][p] = bl1;
        dist[p][c1] = bl1;

        dist[c2][p] = bl2;
        dist[p][c2] = bl2;

        all_visited.push(c1);
        all_visited.push(c2);
        all_visited.push(p);
    }

    let n_leaves = k + 1;
    let mut result: Vec<Vec<f32>> = vec![vec![0.0; n_leaves]; n_leaves];
    for i in 0..n_leaves {
        for j in 0..n_leaves {
            result[i][j] = dist[i][j];
        }
    }

    result
}

/// Get the cophenetic distances from the Phylo2Vec vector
/// Output is a pairwise distance matrix of dimensions n x n
/// where n = number of leaves.
///
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::vector::cophenetic_distances;
///
/// let v = vec![0, 0, 0, 1, 3, 3, 1, 4, 4];
/// let dist = cophenetic_distances(&v);
/// ```
pub fn cophenetic_distances(v: &[usize]) -> Vec<Vec<usize>> {
    let result = _cophenetic_distances(v, None);

    // Convert f32 to usize
    result
        .iter()
        .map(|row| row.iter().map(|&x| x as usize).collect())
        .collect()
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
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::vector::get_mrca;
///
/// let v = vec![0, 1, 2, 3, 4];
/// let mrca = get_mrca(&v, 2, 3);
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

    let pairs = get_pairs(v);

    let path1 = get_ancestry_path_of_node(&pairs, node1);
    let path2 = get_ancestry_path_of_node(&pairs, node2);

    min_common_ancestor(&path1, &path2)
}
