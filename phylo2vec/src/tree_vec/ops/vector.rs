use crate::tree_vec::ops::avl::AVLTree;
use crate::tree_vec::types::{Ancestry, Pair, PairsVec};
use crate::utils::is_unordered;
use core::num;
use std::collections::HashMap;
use std::usize;

/// Get the pair of nodes from the Phylo2Vec vector
/// using a vector data structure and for loops
/// implementation.
///
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::vector::get_pairs;
///
/// let v = vec![0, 0, 0, 1, 3, 3, 1, 4, 4];
/// let pairs = get_pairs(&v);
/// ```
pub fn get_pairs(v: &Vec<usize>) -> PairsVec {
    let num_of_leaves: usize = v.len();
    let mut pairs: PairsVec = Vec::with_capacity(num_of_leaves);

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
    for j in 1..num_of_leaves {
        let next_leaf = j + 1;
        if v[j] == 2 * j {
            // 2 * j = extra root ==> pairing = (0, next_leaf)
            let pair: Pair = (0, next_leaf);
            pairs.push(pair);
        } else if v[j] > j {
            /*
            If v[j] > j, it's not the branch leading to v[j] that gives birth,
            but an internal branch. Insert at the calculated index.
            */
            let index: usize = pairs.len() + v[j] - 2 * j;
            let new_pair: Pair = (pairs[index - 1].0, next_leaf);
            pairs.insert(index, new_pair);
        }
    }

    pairs
}

/// Get the pair of nodes from the Phylo2Vec vector
/// using an AVL tree data structure implementation.
///
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::vector::get_pairs_avl;
///
/// let v = vec![0, 0, 0, 1, 3, 3, 1, 4, 4];
/// let pairs = get_pairs_avl(&v);
/// ```
pub fn get_pairs_avl(v: &Vec<usize>) -> PairsVec {
    // AVL tree implementation of get_pairs
    let k = v.len();
    let mut avl_tree = AVLTree::new();
    avl_tree.insert(0, (0, 1));

    for i in 1..k {
        let next_leaf = i + 1;
        if v[i] <= i {
            avl_tree.insert(0, (v[i], next_leaf));
        } else {
            let index = v[i] - next_leaf;
            let pair = AVLTree::lookup(&avl_tree, index);
            avl_tree.insert(index + 1, (pair.0, next_leaf));
        }
    }

    avl_tree.get_pairs()
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
pub fn get_ancestry(v: &Vec<usize>) -> Ancestry {
    let pairs: PairsVec;

    // Determine the implementation to use
    // based on whether this is an ordered
    // or unordered tree vector
    match is_unordered(&v) {
        true => {
            pairs = get_pairs_avl(&v);
        }
        false => {
            pairs = get_pairs(&v);
        }
    }
    let num_of_leaves = v.len();
    // Initialize Ancestry with capacity `k`
    let mut ancestry: Ancestry = Vec::with_capacity(num_of_leaves);
    // Keep track of child->highest parent relationship
    let mut parents: Vec<usize> = vec![usize::MAX; 2 * num_of_leaves + 1];

    for i in 0..num_of_leaves {
        let (c1, c2) = pairs[i];

        let parent_of_child1 = if parents[c1] != usize::MAX {
            parents[c1]
        } else {
            c1
        };
        let parent_of_child2 = if parents[c2] != usize::MAX {
            parents[c2]
        } else {
            c2
        };

        // Next parent
        let next_parent = num_of_leaves + i + 1;
        ancestry.push([parent_of_child1, parent_of_child2, next_parent]);

        // Update the parents of current children
        parents[c1] = next_parent;
        parents[c2] = next_parent;
    }

    ancestry
}

pub fn find_coords_of_first_leaf(ancestry: &Ancestry, leaf: usize) -> (usize, usize) {
    for r in 0..ancestry.len() {
        for c in 0..3 {
            if ancestry[r][c] == leaf {
                return (r, c);
            }
        }
    }
    panic!("Leaf not found in ancestry");
}

pub fn order_cherries(ancestry: &mut Ancestry) {
    let num_cherries = ancestry.len();
    let num_nodes = 2 * num_cherries + 2;

    let mut min_desc = vec![usize::MAX; num_nodes];

    // Sort by the parent node (ascending order)
    ancestry.sort_by_key(|x| x[2]);

    for i in 0..num_cherries {
        let [c1, c2, p] = ancestry[i];
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

        // Collect the minimum descendant and allocate it to min_desc[parent]
        let desc_min = std::cmp::min(min_desc1, min_desc2);
        min_desc[p] = desc_min;

        // Instead of the parent, we collect the max node
        let desc_max = std::cmp::max(min_desc1, min_desc2);
        ancestry[i] = [min_desc1, min_desc2, desc_max];
    }
}

/// Order cherries in an ancestry vector without parent labels.
/// The goal of this function is to find indices for an argsort to sort the cherries
/// We utilise the fact that the input contains certain local orders, but not the global order.
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
pub fn order_cherries_no_parents(ancestry: &mut Ancestry) {
    let num_cherries = ancestry.len();
    let mut to_sort: Vec<usize> = Vec::with_capacity(num_cherries);
    let mut visited: HashMap<usize, usize> = HashMap::new();

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
    let mut indices: Vec<usize> = (0..num_cherries).collect();
    indices.sort_by_key(|&i| std::cmp::Reverse(to_sort[i]));

    // Note: using a stable sort is important to keep the order of cherries
    let mut temp = Ancestry::with_capacity(num_cherries);
    for i in indices {
        temp.push(ancestry[i]);
    }
    *ancestry = temp;
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
    return v;
}

/// Get the cophenetic distances from the Phylo2Vec vector
/// Output is a pairwise distance matrix of dimensions n x n
///
/// # Example
/// ```
/// use phylo2vec::tree_vec::ops::vector::cophenetic_distances;
///
/// let v = vec![0, 0, 0, 1, 3, 3, 1, 4, 4];
/// let dist = cophenetic_distances(&v, false);
/// ```
pub fn cophenetic_distances(v: &Vec<usize>, unrooted: bool) -> Vec<Vec<usize>> {
    let mut ancestry = get_ancestry(v);

    if unrooted {
        // Base case for unrooted trees: Simply two nodes connected to each other by a single edge
        if v.len() == 1 {
            return vec![vec![0, 1], vec![1, 0]];
        }
        let nrows = ancestry.len();
        let ncols = ancestry[0].len();
        ancestry[nrows - 1][ncols - 1] = ancestry.iter().flatten().max().unwrap() - 1;
    }

    let n_leaves = v.len() + 1;
    let size = 2 * n_leaves - 1;
    let mut dist: Vec<Vec<usize>> = vec![vec![0; size]; size];
    let mut all_visited: Vec<usize> = Vec::new();

    for i in 0..(n_leaves - 1) {
        let [c1, c2, p] = ancestry[n_leaves - i - 2];

        if all_visited.len() >= 1 {
            // Iterate over all_visited except the last element
            for &visited in &all_visited[0..all_visited.len() - 1] {
                let dist_from_visited = dist[p][visited] + 1;
                // c1 to visited
                dist[c1][visited] = dist_from_visited;
                dist[visited][c1] = dist_from_visited;
                // c2 to visited
                dist[c2][visited] = dist_from_visited;
                dist[visited][c2] = dist_from_visited;
            }
        }
        // c1 to c2: path length = 2
        dist[c1][c2] = 2;
        dist[c2][c1] = 2;
        // c1 to parent: path length = 1
        dist[c1][p] = 1;
        dist[p][c1] = 1;
        // c2 to parent: path length = 1
        dist[c2][p] = 1;
        dist[p][c2] = 1;

        all_visited.push(c1);
        all_visited.push(c2);
        all_visited.push(p);
    }

    // Extract the top-left n_leaves x n_leaves submatrix
    let mut result: Vec<Vec<usize>> = vec![vec![0; n_leaves]; n_leaves];
    for i in 0..n_leaves {
        for j in 0..n_leaves {
            result[i][j] = dist[i][j];
        }
    }
    result
}
