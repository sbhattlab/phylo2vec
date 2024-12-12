use crate::tree_vec::types::Ancestry;

fn _get_cherries_recursive_inner(ancestry: &mut Ancestry, newick: &str, has_parents: bool) {
    let mut open_idx: usize = 0;

    for (i, ch) in newick.chars().enumerate() {
        if ch == '(' {
            open_idx = i + 1;
        } else if ch == ')' {
            let pairs: Vec<usize> = newick[open_idx..i]
                .split(',')
                .map(|x: &str| x.parse::<usize>().unwrap())
                .collect();
            let c1 = pairs[0];
            let c2 = pairs[1];
            let parent: usize;
            let new_newick: String;
            match has_parents {
                // For case that the newick string has parents
                true => {
                    parent = newick[i + 1..]
                        .split(',')
                        .next()
                        .unwrap_or("")
                        .split(')')
                        .next()
                        .unwrap_or("")
                        .parse::<usize>()
                        .unwrap();
                    new_newick = format!("{}{}", &newick[..open_idx - 1], &newick[i + 1..]);
                }
                // For case that the newick string does not have parents
                false => {
                    parent = std::cmp::max(c1, c2);
                    new_newick = newick.replace(
                        &newick[open_idx - 1..i + 1],
                        &std::cmp::min(c1, c2).to_string(),
                    );
                }
            }

            ancestry.push([c1, c2, parent]);

            return _get_cherries_recursive_inner(ancestry, &new_newick, has_parents);
        }
    }

    // Extract the children
}

pub fn get_cherries(newick: &str) -> Ancestry {
    let mut ancestry: Ancestry = Vec::new();
    _get_cherries_recursive_inner(&mut ancestry, &newick[..newick.len() - 1], true);
    ancestry
}

pub fn get_cherries_no_parents(newick: &str) -> Ancestry {
    let mut ancestry: Ancestry = Vec::new();
    _get_cherries_recursive_inner(&mut ancestry, &newick[..newick.len() - 1], false);
    ancestry
}

// The recursive function that builds the Newick string
fn _build_newick_recursive_inner(p: usize, ancestry: &Ancestry) -> String {
    let leaf_max = ancestry.len();

    // Extract the children (c1, c2) and ignore the parent from the ancestry tuple
    let [c1, c2, _] = ancestry[p - leaf_max - 1];

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
