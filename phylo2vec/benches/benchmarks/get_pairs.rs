use std::ops::Range;
use std::time::Duration;

use criterion::{criterion_group, BenchmarkId, Criterion};
use phylo2vec::to_newick;
use phylo2vec::utils::sample;

const GET_PAIRS: &str = "get_pairs";
const GET_PAIRS_AVL: &str = "get_pairs_avl";
const RANGE: Range<u32> = 8..18;

/// Function to benchmark the get_pairs and get_pairs_avl functions
fn run_get_pairs(func: &str, n_leaves: usize, ordering: bool) {
    let v = sample(n_leaves, ordering);
    let _ = match func {
        GET_PAIRS => to_newick::_get_pairs(&v),
        GET_PAIRS_AVL => to_newick::_get_pairs_avl(&v),
        _ => panic!("Invalid function name"),
    };
}

/// Benchmark the get_pairs and get_pairs_avl functions
/// with unordered sample input
fn compare_unordered_get_pairs_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_pairs_unordered");
    let ordered = false;
    for j in RANGE {
        let i = 2_i32.checked_pow(j).unwrap() as usize;
        group.bench_with_input(BenchmarkId::new(GET_PAIRS, i), &i, |b, i| {
            b.iter(|| run_get_pairs(GET_PAIRS, *i, ordered));
        });
        group.bench_with_input(BenchmarkId::new(GET_PAIRS_AVL, i), &i, |b, i| {
            b.iter(|| run_get_pairs(GET_PAIRS_AVL, *i, ordered));
        });
    }
    group.finish();
}

/// Benchmark the get_pairs and get_pairs_avl functions
/// with ordered sample input
fn compare_ordered_get_pairs_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_pairs_ordered");
    let ordered = true;
    for j in RANGE {
        let i = 2_i32.checked_pow(j).unwrap() as usize;
        group.bench_with_input(BenchmarkId::new(GET_PAIRS, i), &i, |b, i| {
            b.iter(|| run_get_pairs(GET_PAIRS, *i, ordered));
        });
        group.bench_with_input(BenchmarkId::new(GET_PAIRS_AVL, i), &i, |b, i| {
            b.iter(|| run_get_pairs(GET_PAIRS_AVL, *i, ordered));
        });
    }
    group.finish();
}

criterion_group! {
    name = get_pairs;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_millis(1000)).warm_up_time(Duration::from_millis(1000));
    targets = compare_unordered_get_pairs_group, compare_ordered_get_pairs_group
}
