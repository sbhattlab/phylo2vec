/// Base functions for `phylo2vec` vectors: sampling, validation, and checking properties.
use rand::Rng;

/// Sample a vector with `n_leaves` elements.
///
/// If ordering is True, sample an ordered tree, by default ordering is False.
///
/// ordering=True: `v_i` in {0, 1, ..., i} for i in (0, n_leaves-1)
///
/// ordering=False: `v_i` in {0, 1, ..., 2*i} for i in (0, n_leaves-1)
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::base::sample_vector;
/// let v = sample_vector(10, false);
/// let v2 = sample_vector(5, true);
/// ```
pub fn sample_vector(n_leaves: usize, ordered: bool) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::with_capacity(n_leaves);
    let mut rng = rand::rng();

    if ordered {
        for i in 0..(n_leaves - 1) {
            v.push(rng.random_range(0..=i));
        }
    } else {
        for i in 0..(n_leaves - 1) {
            v.push(rng.random_range(0..=2 * i));
        }
    }

    v
}

/// Input validation of a `phylo2vec` vector
///
/// The input is checked to satisfy the `phylo2vec` constraints
///
/// # Panics
///
/// Panics if any element of the input vector is out of bounds
///
/// # Examples
///
/// ```
/// use phylo2vec::vector::base::check_v;
/// check_v(&vec![0, 0, 1]);
/// ```
pub fn check_v(v: &[usize]) {
    for (i, vi) in v.iter().enumerate() {
        check_max(i, *vi);
    }
}

/// Validate the maximum value of a `phylo2vec` vector element
///
/// # Panics
/// Panics if the value is out of bounds (max = 2 * idx)
fn check_max(idx: usize, value: usize) {
    let absolute_max = 2 * idx;
    assert!(
        value <= absolute_max,
        "{}",
        format!("Validation failed: v[{idx}] = {value} is out of bounds (max = {absolute_max})"),
    );
}

/// Check if a `phylo2vec` vector is unordered
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
/// use phylo2vec::vector::base::is_unordered;
///
/// let unordered = is_unordered(&vec![0, 0, 0, 3, 2, 9, 4, 1, 12]);
///
/// assert_eq!(unordered, true);
///
/// let unordered = is_unordered(&vec![0, 0, 0, 1, 3, 3, 1, 4, 4]);
///
/// assert_eq!(unordered, false);
/// ```
pub fn is_unordered(v: &[usize]) -> bool {
    for (i, vi) in v.iter().enumerate() {
        check_max(i, *vi);
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
    fn test_sample_vector(#[case] n_leaves: usize, #[case] ordering: bool, #[case] scale: usize) {
        let v = sample_vector(n_leaves, ordering);
        assert_eq!(v.len(), n_leaves - 1);
        check_v(&v);
        for (i, vi) in v.iter().enumerate() {
            assert!(*vi <= scale * i);
        }
    }

    #[rstest]
    #[case(vec![0, 0, 1])]
    #[case(vec![0, 0, 2, 1, 8])]
    #[case(vec![0, 2, 4, 6])]
    #[should_panic]
    #[case(vec![0, 0, 9, 1])]
    #[should_panic]
    #[case(vec![1, 0, 0, 0])]
    #[should_panic]
    #[case(vec![0, 3, 5, 7])]
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
