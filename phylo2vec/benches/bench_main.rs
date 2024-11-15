use criterion::criterion_main;

mod benchmarks;

criterion_main!(benchmarks::get_pairs::get_pairs);
