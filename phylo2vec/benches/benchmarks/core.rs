use criterion::{criterion_group, BenchmarkId, Criterion};
use phylo2vec::matrix::base::sample_matrix;
use phylo2vec::matrix::convert::{
    from_newick as from_newick_from_matrix, to_newick as to_newick_from_matrix,
};
use phylo2vec::vector::base::sample_vector;
use phylo2vec::vector::convert::{
    from_newick as from_newick_from_vector, to_newick as to_newick_from_vector,
};
use std::ops::RangeInclusive;
use std::time::Duration;

/// A range of multipliers used to generate benchmark sample sizes.
/// Each value in this range is multiplied by 10,000 to produce sample sizes
/// ranging from 10,000 to 100,000.
const SAMPLE_SIZES: RangeInclusive<usize> = 1..=10;

/// Benchmark to_newick with both ordered and unordered inputs
fn bench_to_newick(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_newick");
    // Set logarithmic scale for plot
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for i in SAMPLE_SIZES {
        // Scale the sample size by 10000 to adjust the range for benchmarking purposes
        let sample_size = 10000 * i;

        // Benchmark ordered case
        group.bench_with_input(
            BenchmarkId::new("vector_ordered", sample_size),
            &sample_size,
            |b, &size| {
                let v = sample_vector(size, true);

                // Benchmark only the to_newick operation
                b.iter(|| to_newick_from_vector(&v));
            },
        );

        // Benchmark unordered case
        group.bench_with_input(
            BenchmarkId::new("vector_unordered", sample_size),
            &sample_size,
            |b, &size| {
                let v = sample_vector(size, false);

                // Benchmark only the to_newick operation
                b.iter(|| to_newick_from_vector(&v));
            },
        );

        // Benchmark unordered case
        group.bench_with_input(
            BenchmarkId::new("matrix_unordered", sample_size),
            &sample_size,
            |b, &size| {
                let m = sample_matrix(size, false);

                // Benchmark only the to_newick operation
                b.iter(|| to_newick_from_matrix(&m.view()));
            },
        );
    }
    group.finish();
}

/// Benchmark to_vector
fn bench_from_newick(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_newick");
    // Set logarithmic scale for plot
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );
    for i in SAMPLE_SIZES {
        let sample_size = 10000 * i;
        group.bench_with_input(
            BenchmarkId::new("vector", sample_size),
            &sample_size,
            |b, &size| {
                // Generate the Newick string once outside the benchmark loop
                let v = sample_vector(size, true);
                let newick = to_newick_from_vector(&v);

                // Benchmark only the to_vector operation
                b.iter(|| from_newick_from_vector(&newick));
            },
        );
    }

    for i in SAMPLE_SIZES {
        let sample_size = 10000 * i;
        group.bench_with_input(
            BenchmarkId::new("matrix", sample_size),
            &sample_size,
            |b, &size| {
                // Generate the matrix once outside the benchmark loop
                let m = sample_matrix(size, false);
                let newick = to_newick_from_matrix(&m.view());

                // Benchmark only the to_vector operation
                b.iter(|| from_newick_from_matrix(&newick));
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = core;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(1000));
    targets = bench_to_newick, bench_from_newick
}
