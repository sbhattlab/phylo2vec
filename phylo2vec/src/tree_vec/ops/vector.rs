use crate::tree_vec::ops::avl::{AVLTree, Pair};
use crate::utils::is_unordered;

/// A type alias for the Ancestry type, which is a vector of vectors representing [child1, child2, parent]
pub type Ancestry = Vec<[usize; 3]>;

/// A type alias for the PairsVec type, which is a vector of tuples representing (child1, child2)
pub type PairsVec = Vec<Pair>;

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
    let mut parents: Vec<isize> = vec![-1; 2 * num_of_leaves + 1];

    for i in 0..num_of_leaves {
        let (c1, c2) = pairs[i];

        let parent_of_child1 = if parents[c1] != -1 {
            parents[c1] as usize
        } else {
            c1
        };
        let parent_of_child2 = if parents[c2] != -1 {
            parents[c2] as usize
        } else {
            c2
        };

        // Next parent
        let next_parent = (num_of_leaves + i + 1) as isize;
        ancestry.push([parent_of_child1, parent_of_child2, next_parent as usize]);

        // Update the parents of current children
        parents[c1] = next_parent;
        parents[c2] = next_parent;
    }

    ancestry
}

// The recursive function that builds the Newick string
fn _build_newick_recursive_inner(p: usize, ancestry: &Ancestry) -> String {
    let leaf_max = ancestry.len();

    // Extract the children (c1, c2) and ignore the parent from the ancestry tuple
    let c1 = ancestry[p - leaf_max - 1][0];
    let c2 = ancestry[p - leaf_max - 1][1];

    // Recursive calls for left and right children, checking if they are leaves or internal nodes
    let left = if c1 > leaf_max {
        _build_newick_recursive_inner(c1, ancestry)
    } else {
        c1.to_string() // It's a leaf node, just convert to string
    };

    let right = if c2 > leaf_max {
        _build_newick_recursive_inner(c2, ancestry)
    } else {
        c2.to_string() // It's a leaf node, just convert to string
    };

    // Create the Newick string in the form (left, right)p
    format!("({},{}){}", left, right, p)
}

/// Build newick string from the ancestry matrix
pub fn build_newick(ancestry: &Ancestry) -> String {
    // Get the root node, which is the parent value of the last ancestry element
    let root = ancestry.last().unwrap()[2];

    // Build the Newick string starting from the root, and append a semicolon
    format!("{};", _build_newick_recursive_inner(root, ancestry))
}

/// Recover a rooted tree (in Newick format) from a Phylo2Vec vector
pub fn to_newick(v: &Vec<usize>) -> String {
    let ancestry: Ancestry = get_ancestry(&v);
    build_newick(&ancestry)
}
