/// Represents the regular expressions used to parse Newick trees.
///
/// This is essentially a holder for the various regular expressions
/// used to parse newick trees such as the left node, right node, pairs,
/// branch lengths, and parents.
///
/// # Examples
///
/// ```
/// use phylo2vec::newick::NewickPatterns;
///
/// let newick_patterns = NewickPatterns::new();
/// let newick = "(1,2)3";
/// let result = newick_patterns.pairs.is_match(newick);
/// assert_eq!(result, true);
/// ```
#[derive(Debug)]
pub struct NewickPatterns {
    pub left_node: regex::Regex,
    pub left_node_generic: regex::Regex,
    pub right_node: regex::Regex,
    pub right_node_generic: regex::Regex,
    pub pairs: regex::Regex,
    pub branch_lengths: regex::Regex,
    pub parents: regex::Regex,
}

impl NewickPatterns {
    /// Create a new instance of `NewickPatterns`.
    ///
    /// Patterns:
    /// * `left_node`: `\(\b(\d+)\b`
    /// * `right_node`: `,\b(\d+)\b`
    /// * `pairs`: `({})|({})`
    /// * `branch_lengths`: `:\d+(\.\d+)?`
    /// * `parents`: `\)(\d+)`
    /// # Panics
    /// This function will panic if the regular expressions are invalid.
    pub fn new() -> Self {
        let lnode_pattern = r"\(\b(\d+)\b";
        let rnode_pattern = r",\b(\d+)\b";
        let pnode_pattern = r"\)(\d+)";
        let lnode_generic_pattern = r"\(\b(\w+)\b";
        let rnode_generic_pattern = r",\b(\w+)\b";
        let bl_pattern = r":\d+(\.\d+)?";
        let pair_pattern = format!(r"({lnode_pattern})|({rnode_pattern})");
        Self {
            // Pattern of an integer label on the left of a pair
            left_node: regex::Regex::new(lnode_pattern).unwrap(),
            // Pattern of an integer label on the right of a pair
            right_node: regex::Regex::new(rnode_pattern).unwrap(),
            // Pattern of a generic label on the left of a pair
            left_node_generic: regex::Regex::new(lnode_generic_pattern).unwrap(),
            // Pattern of a generic label on the right of a pair
            right_node_generic: regex::Regex::new(rnode_generic_pattern).unwrap(),
            // Pattern of a pair of integer labels
            pairs: regex::Regex::new(&pair_pattern).unwrap(),
            // Pattern of a branch length annotation
            branch_lengths: regex::Regex::new(bl_pattern).unwrap(),
            // Pattern of a parent label
            parents: regex::Regex::new(pnode_pattern).unwrap(),
        }
    }
}

impl Default for NewickPatterns {
    fn default() -> Self {
        Self::new()
    }
}
