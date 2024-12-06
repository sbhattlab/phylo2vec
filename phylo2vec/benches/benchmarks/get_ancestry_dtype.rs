use std::ops::Range;
use std::time::Duration;

use criterion::{criterion_group, BenchmarkId, Criterion};
use phylo2vec::tree_vec::ops;
use phylo2vec::utils::{is_unordered, sample};

pub type AncestryTuple = Vec<(usize, usize, usize)>;
pub type AncestryVec = Vec<[usize; 3]>;
pub type AncestryNDArray = ndarray::Array2<usize>;

const RANGE: Range<u32> = 8..18;

fn compare_get_ancestry_datatypes(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_ancestry_datatypes");
    // compare the three functions with three different data types
    for j in RANGE {
        let i = 2_i32.checked_pow(j).unwrap() as usize;
        let v = sample(i, true);
        group.bench_with_input(BenchmarkId::new("tuple", i), &v, |b, v| {
            b.iter(|| {
                get_ancestry_tuple(v);
            });
        });
        group.bench_with_input(BenchmarkId::new("vector", i), &v, |b, v| {
            b.iter(|| {
                get_ancestry_vec(v);
            });
        });
        group.bench_with_input(BenchmarkId::new("ndarray", i), &v, |b, v| {
            b.iter(|| {
                get_ancestry_ndarray(v);
            });
        });
    }
    group.finish();
}

pub fn get_ancestry_tuple(v: &Vec<usize>) -> AncestryTuple {
    let pairs: ops::vector::PairsVec;

    // Determine the implementation to use
    // based on whether this is an ordered
    // or unordered tree vector
    match is_unordered(&v) {
        true => {
            pairs = ops::get_pairs_avl(&v);
        }
        false => {
            pairs = ops::get_pairs(&v);
        }
    }
    let num_of_leaves = v.len();
    // Initialize Ancestry with capacity `k`
    let mut ancestry: AncestryTuple = Vec::with_capacity(num_of_leaves);
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
        ancestry.push((parent_of_child1, parent_of_child2, next_parent as usize));

        // Update the parents of current children
        parents[c1] = next_parent;
        parents[c2] = next_parent;
    }

    ancestry
}

pub fn get_ancestry_vec(v: &Vec<usize>) -> AncestryVec {
    let pairs: ops::vector::PairsVec;

    // Determine the implementation to use
    // based on whether this is an ordered
    // or unordered tree vector
    match is_unordered(&v) {
        true => {
            pairs = ops::get_pairs_avl(&v);
        }
        false => {
            pairs = ops::get_pairs(&v);
        }
    }
    let num_of_leaves = v.len();
    // Initialize Ancestry with capacity `k`
    let mut ancestry: AncestryVec = Vec::with_capacity(num_of_leaves);
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

pub fn get_ancestry_ndarray(v: &Vec<usize>) -> AncestryNDArray {
    let pairs: ops::vector::PairsVec;

    // Determine the implementation to use
    // based on whether this is an ordered
    // or unordered tree vector
    match is_unordered(&v) {
        true => {
            pairs = ops::get_pairs_avl(&v);
        }
        false => {
            pairs = ops::get_pairs(&v);
        }
    }
    let num_of_leaves = v.len();
    // Initialize Ancestry with capacity `k`
    let mut ancestry: AncestryNDArray = ndarray::Array2::zeros((num_of_leaves, 3));
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
        ancestry[[i, 0]] = parent_of_child1;
        ancestry[[i, 1]] = parent_of_child2;
        ancestry[[i, 2]] = next_parent as usize;

        // Update the parents of current children
        parents[c1] = next_parent;
        parents[c2] = next_parent;
    }

    ancestry
}

criterion_group! {
    name = get_ancestry_datatypes;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_millis(1000)).warm_up_time(Duration::from_millis(1000));
    targets = compare_get_ancestry_datatypes
}
