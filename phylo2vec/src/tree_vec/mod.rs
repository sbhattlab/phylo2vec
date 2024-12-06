use crate::utils::sample;

pub mod ops;

#[derive(Debug, PartialEq, Clone)]
pub struct TreeVec {
    n_leaf: usize,
    data: Vec<usize>,
    branch_lengths: Option<Vec<(f64, f64)>>,
    taxa: Option<Vec<String>>,
    is_rooted: bool,
}

impl TreeVec {
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

    pub fn from_sample(n_leaves: usize, ordering: bool) -> Self {
        let v = sample(n_leaves, ordering);
        TreeVec::new(v, None, None)
    }

    pub fn to_newick(&self) -> String {
        return ops::to_newick(&self.data);
    }

    pub fn get_ancestry(&self) -> Vec<[usize; 3]> {
        return ops::get_ancestry(&self.data);
    }

    pub fn add_leaf(leaf: usize, branch: usize) -> Self {
        unimplemented!();
    }

    pub fn remove_leaf(leaf: usize) -> Self {
        unimplemented!();
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

    #[rstest]
    #[case(vec![0, 0, 0, 1, 3], "(((0,(3,5)6)8,2)9,(1,4)7)10;")]
    #[case(vec![0, 1, 2, 3, 4], "(0,(1,(2,(3,(4,5)6)7)8)9)10;")]
    #[case(vec![0, 0, 1], "((0,2)5,(1,3)4)6;")]
    fn test_to_newick(#[case] v: Vec<usize>, #[case] expected: &str) {
        let tree = TreeVec::new(v, None, None);
        let newick = tree.to_newick();
        assert_eq!(newick, expected);
    }

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
    fn test_get_ancestry(#[case] v: Vec<usize>, #[case] expected: Vec<[usize; 3]>) {
        let tree = TreeVec::new(v, None, None);
        let ancestry = tree.get_ancestry();
        assert_eq!(ancestry, expected);
    }
}
