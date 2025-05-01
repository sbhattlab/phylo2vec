"""Benchmark of Python bindings for the Rust core of phylo2vec."""

import pytest

from phylo2vec import _phylo2vec_core as core

BIG_RANGE = range(10000, 110000, 10000)


@pytest.mark.parametrize("sample_size", BIG_RANGE)
def test_to_newick_ordered(benchmark, sample_size):
    """Benchmark the conversion of an ordered vector to Newick format.

    Parameters
    ----------
    benchmark : _pytest.fixtures.BenchmarkFixture
        Benchmark fixture from pytest-benchmark
    sample_size : int
        Number of leaves in the vector
    """
    benchmark(core.to_newick_from_vector, core.sample_vector(sample_size, True))


@pytest.mark.parametrize("sample_size", BIG_RANGE)
def test_to_newick_unordered(benchmark, sample_size):
    """Benchmark the conversion of an unordered vector to Newick format.

    Parameters
    ----------
    benchmark : _pytest.fixtures.BenchmarkFixture
        Benchmark fixture from pytest-benchmark
    sample_size : int
        Number of leaves in the vector
    """
    benchmark(core.to_newick_from_vector, core.sample_vector(sample_size, False))


@pytest.mark.parametrize("sample_size", BIG_RANGE)
def test_to_vector(benchmark, sample_size):
    """Benchmark the conversion of a Newick string (without branch lengths) to vector format

    Parameters
    ----------
    benchmark : _pytest.fixtures.BenchmarkFixture
        Benchmark fixture from pytest-benchmark
    sample_size : int
        Number of leaves in the vector
    """
    benchmark(
        core.to_vector,
        core.to_newick_from_vector(core.sample_vector(sample_size, True)),
    )
