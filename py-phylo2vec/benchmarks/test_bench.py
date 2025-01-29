from phylo2vec import _phylo2vec_core as p2v
import pytest

@pytest.mark.parametrize("sample_size", [2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18])
def test_to_newick_ordered(benchmark, sample_size):
    benchmark(p2v.to_newick, p2v.sample(sample_size, True))

@pytest.mark.parametrize("sample_size", [2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18])
def test_to_newick_unordered(benchmark, sample_size):
    benchmark(p2v.to_newick, p2v.sample(sample_size, False))

@pytest.mark.parametrize("sample_size", [2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14])
def test_to_vector(benchmark, sample_size):
    benchmark(p2v.to_vector, p2v.to_newick(p2v.sample(sample_size, True)))
