/// An object implementing a phylogenetic tree in phylo2vec vector format
use crate::types::Ancestry;
use crate::vector::base::sample_vector;
use crate::vector::convert;
use crate::vector::ops;

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
            data,
            n_leaf,
            is_rooted: true,
            branch_lengths,
            taxa,
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
        let v = sample_vector(n_leaves, ordering);
        TreeVec::new(v, None, None)
    }

    /// Converts the tree to Newick format
    ///
    /// # Returns
    /// A String containing the Newick representation of the tree
    pub fn to_newick(&self) -> String {
        convert::to_newick(&self.data)
    }

    /// Gets the ancestry matrix representation of the tree
    ///
    /// # Returns
    /// An `Ancestry` type containing parent-child relationships
    pub fn get_ancestry(&self) -> Ancestry {
        convert::to_ancestry(&self.data)
    }

    pub fn get_edges(&self) -> Vec<(usize, usize)> {
        convert::to_edges(&self.data)
    }

    /// Adds a new leaf to the tree
    ///
    /// # Arguments
    /// * `leaf` - Index of the new leaf to add
    /// * `branch` - Index of the branch to attach the leaf to
    ///
    /// # Result
    /// Modifies the tree structure by adding the new leaf and updating indices
    pub fn add_leaf(&mut self, leaf: usize, branch: usize) {
        let mut vec = self.data.clone();
        self.data = ops::add_leaf(&mut vec, leaf, branch);
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
        let mut vec = self.data.clone();
        let (data, sister_leaf) = ops::remove_leaf(&mut vec, leaf);
        self.data = data;
        sister_leaf
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
        let expected_v = v.to_vec();
        let tree = TreeVec::new(v, None, None);

        assert_eq!(tree.data, expected_v);
        assert_eq!(tree.n_leaf, 9);
        assert!(tree.is_rooted);
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
        assert!(tree.is_rooted);
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
    fn test_get_ancestry(#[case] v: Vec<usize>, #[case] expected: Ancestry) {
        let tree = TreeVec::new(v, None, None);
        let ancestry = tree.get_ancestry();
        assert_eq!(ancestry, expected);
    }

    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], vec![
        (3, 6),
        (5, 6),
        (1, 7),
        (4, 7),
        (0, 8),
        (6, 8),
        (8, 9),
        (2, 9),
        (9, 10),
        (7, 10)])]
    #[case(vec![0, 1, 2, 3], vec![
        (3, 5),
        (4, 5),
        (2, 6),
        (5, 6),
        (1, 7),
        (6, 7),
        (0, 8),
        (7, 8)])]
    #[case(vec![0, 0, 1], vec![
        (1, 4),
        (3, 4),
        (0, 5),
        (2, 5),
        (5, 6),
        (4, 6)])]
    fn test_get_edges(#[case] v: Vec<usize>, #[case] expected: Vec<(usize, usize)>) {
        let tree = TreeVec::new(v, None, None);
        let ancestry = tree.get_edges();
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
