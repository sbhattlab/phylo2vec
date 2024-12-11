use crate::tree_vec::ops::avl::AVLTree;
use crate::tree_vec::types::{Ancestry, Pair, PairsVec};
use crate::utils::is_unordered;
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

pub fn order_cherries_no_parents(ancestry: &mut Ancestry) {
    let num_cherries = ancestry.len();

    for i in 0..num_cherries {
        // Find the next index to process:
        // The goal is to find the row with the highest leaf
        // where both leaves were previously un-visited
        // why? If a leaf in a cherry already appeared in the ancestry,
        // it means that leaf was already involved in a shallower cherry
        let mut idx = usize::MAX;

        // Initially, all cherries have not been processed
        let mut unvisited = vec![true; num_cherries + 1];

        // Temporary max leaf
        let mut max_leaf = 0;

        for j in i..num_cherries {
            let [c1, c2, c_max] = ancestry[j];

            if c_max > max_leaf {
                if unvisited[c1] && unvisited[c2] {
                    max_leaf = c_max;
                    idx = j;
                }
            }

            // c1 and c2 have been processed
            unvisited[c1] = false;
            unvisited[c2] = false;
        }

        if idx != i {
            ancestry[i..idx + 1].rotate_right(1);
        }
    }
}

pub fn build_vector(cherries: Ancestry) -> Vec<usize> {
    let num_cherries = cherries.len();
    let num_leaves = num_cherries + 1;

    let mut v = vec![0; num_cherries];
    let mut idxs = vec![0; num_leaves];

    for i in 0..num_cherries {
        let [c1, c2, c_max] = cherries[i];

        let mut idx = 0;

        for j in 1..c_max {
            idx += idxs[j];
        }
        // Reminder: v[i] = j --> branch i yields leaf j
        v[c_max - 1] = if idx == 0 {
            std::cmp::min(c1, c2)
        } else {
            c_max - 1 + idx
        };
        idxs[c_max] = 1;
    }
    return v;
}
