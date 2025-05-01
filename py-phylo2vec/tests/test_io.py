"""Tests for I/O functions."""

import numpy as np
import pytest


from phylo2vec.io.reader import load, load_newick
from phylo2vec.io.writer import save, save_newick
from phylo2vec.utils.vector import sample_vector
from .config import MAX_N_LEAVES, MIN_N_LEAVES


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES + 1])
def test_save_and_load(tmp_path, n_leaves):
    """Test the read and write functions for Phylo2Vec objects

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    v = sample_vector(n_leaves)

    csv_path = tmp_path / "test.csv"
    save(v, csv_path)
    v2 = load(csv_path)
    assert np.array_equal(v, v2)

    random_path = tmp_path / "test.random"
    np.savetxt(random_path, v, delimiter=",")
    with pytest.raises(AssertionError):
        save(v, random_path)
        _ = load(random_path)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES + 1])
def test_save_and_load_newick(tmp_path, n_leaves):
    """Test the read and write functions for Phylo2Vec objects

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    v = sample_vector(n_leaves)

    newick_path = tmp_path / "test.newick"
    save_newick(v, newick_path)
    v2 = load_newick(newick_path)
    assert np.array_equal(v, v2)

    random_path = tmp_path / "test.random"
    np.savetxt(random_path, v, delimiter=",")
    with pytest.raises(AssertionError):
        save_newick(v, random_path)
        _ = load_newick(random_path)
