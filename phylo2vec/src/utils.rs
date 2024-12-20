use rand::Rng;

/// Sample a vector with `n_leaves - 1` elements.
///
/// If ordering is True, sample an ordered tree, by default ordering is False
/// ordering=True: v_i in {0, 1, ..., i} for i in (0, n_leaves-1)
/// ordering=False: v_i in {0, 1, ..., 2*i} for i in (0, n_leaves-1)
///
/// # Examples
///
/// ```
/// use phylo2vec::utils::sample;
/// let v = sample(10, false);
/// let v2 = sample(5, true);
/// ```
pub fn sample(n_leaves: usize, ordering: bool) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::with_capacity(n_leaves - 1);
    let mut rng = rand::thread_rng();

    match ordering {
        true => {
            for i in 0..(n_leaves - 1) {
                v.push(rng.gen_range(0..(i + 1)));
            }
        }
        false => {
            for i in 0..(n_leaves - 1) {
                v.push(rng.gen_range(0..(2 * i + 1)));
            }
        }
    }

    v
}

/// Input validation of a Phylo2Vec vector
///
/// The input is checked to satisfy the Phylo2Vec constraints
///
/// # Panics
///
/// Panics if any element of the input vector is out of bounds
///
/// # Examples
///
/// ```
/// use phylo2vec::utils::check_v;
/// check_v(&vec![0, 0, 1]);
/// ```
pub fn check_v(v: &Vec<usize>) -> () {
    for i in 0..v.len() {
        _check_max(i, v[i]);
    }
}

/// Validate the maximum value of a Phylo2Vec vector element
///
/// # Panics
///
/// Panics if the value is out of bounds (max = 2 * idx)
fn _check_max(idx: usize, value: usize) -> () {
    let absolute_max = 2 * idx;
    assert!(
        value <= absolute_max,
        "Validation failed: v[{}] = {} is out of bounds (max = {})",
        idx,
        value,
        absolute_max
    );
}

/// Check if a Phylo2Vec vector is unordered
///
/// # Panics
///
/// Panics if any element of the input vector is out of bounds
///
/// # Returns
///
/// Returns true if the vector is unordered, false otherwise
///
/// # Examples
///
/// ```
/// use phylo2vec::utils::is_unordered;
///
/// let unordered = is_unordered(&vec![0, 0, 0, 3, 2, 9, 4, 1, 12]);
///
/// assert_eq!(unordered, true);
///
/// let unordered = is_unordered(&vec![0, 0, 0, 1, 3, 3, 1, 4, 4]);
///
/// assert_eq!(unordered, false);
/// ```
pub fn is_unordered(v: &Vec<usize>) -> bool {
    for i in 0..v.len() {
        _check_max(i, v[i]);
        if v[i] > i + 1 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(50, true, 1)]
    #[case(50, false, 2)]
    fn test_sample(#[case] n_leaves: usize, #[case] ordering: bool, #[case] scale: usize) {
        let v = sample(n_leaves, ordering);
        assert_eq!(v.len(), n_leaves - 1);
        check_v(&v);
        for i in 0..(n_leaves - 1) {
            assert!(v[i] <= scale * i);
        }
    }

    #[rstest]
    #[case(vec![0, 0, 1])]
    #[case(vec![0, 0, 2, 1, 8])]
    #[should_panic]
    #[case(vec![0, 0, 9, 1])]
    fn test_check_v(#[case] v: Vec<usize>) {
        check_v(&v);
    }

    #[rstest]
    #[case(vec![0, 0, 0, 1, 3, 3, 1, 4, 4], false)]
    #[case(vec![0, 0, 0, 3, 2, 9, 4, 1, 12], true)]
    #[should_panic]
    #[case(vec![0, 0, 1, 10, 2, 9, 4, 1, 12], true)]
    fn test_is_unordered(#[case] v: Vec<usize>, #[case] expected: bool) {
        assert_eq!(is_unordered(&v), expected);
    }
}
