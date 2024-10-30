use rand::Rng;

pub enum SampleOrdering {
    Ordered,
    NotOrdered
}

impl Default for SampleOrdering {
    fn default() -> Self {
        SampleOrdering::NotOrdered
    }
}

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
/// use phylo2vec::utils::SampleOrdering;
/// let v = sample(10, SampleOrdering::NotOrdered);
/// let v2 = sample(5, SampleOrdering::Ordered);
/// ```
pub fn sample(n_leaves: usize, ordering: SampleOrdering) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::with_capacity(n_leaves - 1);
    let mut rng = rand::thread_rng();

    match ordering {
        SampleOrdering::Ordered => {
            for i in 0..(n_leaves - 1) {
                v.push(rng.gen_range(0..(i + 1)));
            }
        }
        SampleOrdering::NotOrdered => {
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
pub fn check_v(v: &Vec<usize>) {
    let k = v.len();
    let v_max: Vec<usize> = (0..k).map(|i| i * 2).collect();

    for i in 0..k {
        assert!(v[i] <= v_max[i], "Validation failed: v[{}] = {} is out of bounds", i, v[i]);
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use super::*;

    #[rstest]
    #[case(20, SampleOrdering::Ordered, 1)]
    #[case(20, SampleOrdering::NotOrdered, 2)]
    fn test_sample(#[case] n_leaves: usize, #[case] ordering: SampleOrdering, #[case] scale: usize) {
        let v = sample(n_leaves, ordering);
        assert_eq!(v.len(), n_leaves - 1);
        check_v(&v);
        for i in 0..(n_leaves - 1) {
            assert!(v[i] <= scale * i);
        };
    }

    #[rstest]
    #[case(vec![0, 0, 1])]
    #[case(vec![0, 0, 2, 1, 8])]
    #[should_panic]
    #[case(vec![0, 0, 9, 1])]
    fn test_check_v(#[case] v: Vec<usize>) {
        check_v(&v);
    }
}
