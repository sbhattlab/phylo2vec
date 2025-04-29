/// Represents the regular expressions used to parse Newick trees.
///
/// This is essentially a holder for the various regular expressions
/// used to parse newick trees such as the left node, right node, pairs,
/// branch lengths, and parents.
///
/// # Example
///
/// ```
/// use phylo2vec::tree_vec::ops::newick::NewickPatterns;
///
/// let newick_patterns = NewickPatterns::new();
/// let newick = "(1,2)3";
/// let result = newick_patterns.pairs.is_match(newick);
/// assert_eq!(result, true);
/// ```
#[derive(Debug)]
pub struct NewickPatterns {
    pub left_node: regex::Regex,
    pub right_node: regex::Regex,
    pub pairs: regex::Regex,
    pub branch_lengths: regex::Regex,
    pub parents: regex::Regex,
}

impl NewickPatterns {
    pub fn new() -> Self {
        let _left_node = r"\(\b(\d+)\b";
        let _right_node = r",\b(\d+)\b";
        let _branch_lengths = r":\d+(\.\d+)?";
        let _parents = r"\)(\d+)";
        let _pairs = format!(r"({})|({})", _left_node, _right_node);
        NewickPatterns {
            // Pattern of an integer label on the left of a pair
            left_node: regex::Regex::new(_left_node).unwrap(),
            // Pattern of an integer label on the right of a pair
            right_node: regex::Regex::new(_right_node).unwrap(),
            // Pattern of a pair of integer labels
            pairs: regex::Regex::new(&_pairs).unwrap(),
            // Pattern of a branch length annotation
            branch_lengths: regex::Regex::new(_branch_lengths).unwrap(),
            // Pattern of a parent label
            parents: regex::Regex::new(_parents).unwrap(),
        }
    }
}

impl Default for NewickPatterns {
    fn default() -> Self {
        Self::new()
    }
}
