use crate::utils::sample;

// Import the types module
pub mod types;

// Import the operations modules
pub mod ops;

/// A vector representation of a phylogenetic tree
///
/// Contains the tree structure, branch lengths, taxa, and rootedness
#[derive(Debug, PartialEq, Clone)]
pub struct TreeVec {
    n_leaf: usize,
    data: Vec<usize>,
    branch_lengths: Option<Vec<(f64, f64)>>,
    taxa: Option<Vec<String>>,
    is_rooted: bool,
}

/// Implementation of the `TreeVec` struct
impl TreeVec {
    /// Creates a new `TreeVec` instance
    ///
    /// # Arguments
    /// * `data` - Vector containing the tree structure
    /// * `branch_lengths` - Optional vector of branch length tuples (start, end)
    /// * `taxa` - Optional vector of taxon names
    ///
    /// # Returns
    /// A new `TreeVec` instance with the specified data and properties
    pub fn new(
        data: Vec<usize>,
        branch_lengths: Option<Vec<(f64, f64)>>,
        taxa: Option<Vec<String>>,
    ) -> Self {
        let n_leaf = data.len();
        TreeVec {
            data: data,
            n_leaf: n_leaf,
            is_rooted: true,
            branch_lengths: branch_lengths,
            taxa: taxa,
        }
    }

    /// Creates a new random tree with specified number of leaves
    ///
    /// # Arguments
    /// * `n_leaves` - Number of leaves in the tree
    /// * `ordering` - Whether to maintain ordered structure
    ///
    /// # Returns
    /// A new randomly generated `TreeVec` instance
    pub fn from_sample(n_leaves: usize, ordering: bool) -> Self {
        let v = sample(n_leaves, ordering);
        TreeVec::new(v, None, None)
    }

    /// Converts the tree to Newick format
    ///
    /// # Returns
    /// A String containing the Newick representation of the tree
    pub fn to_newick(&self) -> String {
        return ops::to_newick(&self.data);
    }

    /// Gets the ancestry matrix representation of the tree
    ///
    /// # Returns
    /// An `Ancestry` type containing parent-child relationships
    pub fn get_ancestry(&self) -> types::Ancestry {
        return ops::get_ancestry(&self.data);
    }

    /// Adds a new leaf to the tree
    ///
    /// # Arguments
    /// * `leaf` - Index of the new leaf to add
    /// * `branch` - Index of the branch to attach the leaf to
    ///
    /// # Side effects
    /// Modifies the tree structure by adding the new leaf and updating indices
    pub fn add_leaf(&mut self, leaf: usize, branch: usize) {
        self.data.push(branch);

        let mut ancestry_add = self.get_ancestry();

        println!("{:?}", ancestry_add);
        let mut found_first_leaf = false;
        for r in 0..ancestry_add.len() {
            for c in 0..3 {
                if !found_first_leaf && ancestry_add[r][c] == self.data.len() {
                    // Find the indices of the first leaf
                    // and then set the value to the new leaf
                    ancestry_add[r][c] = leaf;
                    found_first_leaf = true;
                } else if ancestry_add[r][c] >= leaf {
                    ancestry_add[r][c] += 1;
                }
            }
        }

        // ancestry_add[leaf_coords][leaf_col] = leaf as isize;
        // let ancestry_add_ref = &mut ancestry_add;
        ops::order_cherries(&mut ancestry_add);
        ops::order_cherries_no_parents(&mut ancestry_add);
        self.data = ops::build_vector(ancestry_add);
    }

    /// Removes a leaf from the tree
    ///
    /// # Arguments
    /// * `leaf` - Index of the leaf to remove
    ///
    /// # Returns
    /// The index of the sister node of the removed leaf
    ///
    /// # Side effects
    /// Modifies the tree structure by removing the leaf and updating indices
    pub fn remove_leaf(&mut self, leaf: usize) -> usize {
        let ancestry = self.get_ancestry();
        let leaf_coords = ops::find_coords_of_first_leaf(&ancestry, leaf);
        let leaf_row = leaf_coords.0;
        let leaf_col = leaf_coords.1;

        // Find the parent of the leaf to remove
        let parent = ancestry[leaf_row][2];
        let sister = ancestry[leaf_row][1 - leaf_col];
        let num_cherries = ancestry.len();

        let mut ancestry_rm = Vec::with_capacity(num_cherries - 1);

        for r in 0..num_cherries - 1 {
            let mut new_row = if r < leaf_row {
                ancestry[r].clone()
            } else {
                ancestry[r + 1].clone()
            };

            for c in 0..3 {
                let mut node = new_row[c];

                if node == parent {
                    node = sister;
                }

                // Subtract 1 for leaves > "leaf"
                // (so that the vector is still valid)
                if node > leaf {
                    node -= 1;
                    if node >= parent {
                        node -= 1;
                    }
                }

                new_row[c] = node;
            }

            ancestry_rm.push(new_row);
        }

        ops::order_cherries(&mut ancestry_rm);
        ops::order_cherries_no_parents(&mut ancestry_rm);
        self.data = ops::build_vector(ancestry_rm);

        return sister;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    /// Test the creation of a new tree
    ///
    /// Tests are using 9 leaf tree with no branch lengths and taxa
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3, 3, 1, 4, 4])]
    #[case(vec![0, 0, 0, 3, 2, 9, 4, 1, 12])]
    fn test_new_tree(#[case] v: Vec<usize>) {
        let expected_v = v.iter().map(|x| *x).collect::<Vec<usize>>();
        let tree = TreeVec::new(v, None, None);

        assert_eq!(tree.data, expected_v);
        assert_eq!(tree.n_leaf, 9);
        assert_eq!(tree.is_rooted, true);
        assert_eq!(tree.branch_lengths, None);
        assert_eq!(tree.taxa, None);
    }

    /// Test the creation of a new tree from a sample
    ///
    /// Tests are using 50 leaf tree with ordering and no ordering
    #[rstest]
    #[case(50, true)]
    #[case(50, false)]
    fn test_new_tree_from_sample(#[case] n_leaves: usize, #[case] ordering: bool) {
        let tree = TreeVec::from_sample(n_leaves, ordering);
        assert_eq!(tree.n_leaf, n_leaves - 1);
        assert_eq!(tree.is_rooted, true);
        assert_eq!(tree.branch_lengths, None);
        assert_eq!(tree.taxa, None);
    }

    /// Test the conversion of a tree to Newick format
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)6)8,2)9,(1,4)7)10;")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)6)7)8)9)10;")]
    #[case(vec![0, 0, 1], "((0,2)5,(1,3)4)6;")]
    fn test_to_newick(#[case] v: Vec<usize>, #[case] expected: &str) {
        let tree = TreeVec::new(v, None, None);
        let newick = tree.to_newick();
        assert_eq!(newick, expected);
    }

    /// Test the retrieval of the ancestry matrix
    ///
    /// Tests are using 5 or less leaf tree with different structures
    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], vec![[3, 5, 6],
        [1, 4, 7],
        [0, 6, 8],
        [8, 2, 9],
        [9, 7, 10]])]
    #[case(vec![0, 1, 2, 3], vec![[3, 4, 5],
        [2, 5, 6],
        [1, 6, 7],
        [0, 7, 8]])]
    #[case(vec![0, 0, 1], vec![[1, 3, 4],
        [0, 2, 5],
        [5, 4, 6]])]
    fn test_get_ancestry(#[case] v: Vec<usize>, #[case] expected: types::Ancestry) {
        let tree = TreeVec::new(v, None, None);
        let ancestry = tree.get_ancestry();
        assert_eq!(ancestry, expected);
    }

    /// Test the addition of a new leaf to the tree
    ///
    /// Tests are using 6 leaf tree with different leaf and branch indices
    #[rstest]
    #[case(vec![0, 1, 2, 5, 4, 2], 5, 3, vec![0, 1, 2, 5, 3, 4, 2])]
    #[case(vec![0, 1, 2, 5, 4, 2], 7, 0, vec![0, 1, 2, 5, 4, 2, 0])]
    #[case(vec![0, 1, 2, 5, 4, 2], 7, 2, vec![0, 1, 2, 5, 4, 2, 2])]
    fn test_add_leaf(
        #[case] v: Vec<usize>,
        #[case] leaf: usize,
        #[case] branch: usize,
        #[case] expected: Vec<usize>,
    ) {
        let mut tree = TreeVec::new(v, None, None);
        tree.add_leaf(leaf, branch);
        assert_eq!(tree.data, expected);
    }

    /// Test the removal of a leaf from the tree
    ///
    /// Tests are using 6 leaf tree with different leaf and sister branch indices
    #[rstest]
    #[case(vec![0, 1, 2, 5, 4, 2], 5, 4, vec![0, 1, 2, 5, 2])]
    #[case(vec![0, 1, 2, 5, 4, 2], 6, 2, vec![0, 1, 2, 5, 4])]
    #[case(vec![0, 1, 2, 5, 4, 2], 0, 11, vec![0, 1, 4, 3, 1])]
    fn test_remove_leaf(
        #[case] v: Vec<usize>,
        #[case] leaf: usize,
        #[case] branch: usize,
        #[case] expected: Vec<usize>,
    ) {
        let mut tree = TreeVec::new(v, None, None);
        let sister = tree.remove_leaf(leaf);
        assert_eq!(tree.data, expected);
        assert_eq!(sister, branch);
    }
}
