"""Tests for I/O functions."""

import numpy as np
import pytest

from ete4 import Tree

from phylo2vec.base.newick import from_newick, to_newick
from phylo2vec.io.reader import load, load_newick
from phylo2vec.io.writer import save, save_newick
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.newick import create_label_mapping
from phylo2vec.utils.vector import sample_vector
from .config import MAX_N_LEAVES, MIN_N_LEAVES


def _test_save_and_load(tmp_path, n_leaves, sample_fn):
    """Test the read and write functions for Phylo2Vec objects

    Parameters
    ----------
    tmp_path : Path
        Temporary path for saving files
    n_leaves : int
        Number of leaves
    sample_fn : function
        Function to sample a vector or matrix
    """
    vector_or_matrix = sample_fn(n_leaves)

    csv_path = tmp_path / "test.csv"
    save(vector_or_matrix, csv_path)
    loaded_vector_or_matrix = load(csv_path)
    assert np.array_equal(vector_or_matrix, loaded_vector_or_matrix)

    # Unallowed file extension should trigger an AssertionError
    random_path = tmp_path / "test.random"
    np.savetxt(random_path, vector_or_matrix, delimiter=",")
    with pytest.raises(ValueError):
        save(vector_or_matrix, random_path)
        _ = load(random_path)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_and_load_vector(tmp_path, n_leaves):
    """Test the read and write functions for Phylo2Vec objects

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    _test_save_and_load(tmp_path, n_leaves, sample_vector)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_and_load_matrix(tmp_path, n_leaves):
    """Test the read and write functions for Phylo2Vec objects

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    _test_save_and_load(tmp_path, n_leaves, sample_matrix)


def _test_and_save_load_newick(tmp_path, n_leaves, sample_fn):
    vector_or_matrix = sample_fn(n_leaves)

    newick_path = tmp_path / "test.newick"
    save_newick(vector_or_matrix, newick_path)
    m2 = load_newick(newick_path)
    assert np.array_equal(vector_or_matrix, m2)

    # Unallowed file extension should trigger an AssertionError
    random_path = tmp_path / "test.random"
    np.savetxt(random_path, vector_or_matrix, delimiter=",")
    with pytest.raises(ValueError):
        save_newick(vector_or_matrix, random_path)
        _ = load_newick(random_path)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_and_load_newick_vector(tmp_path, n_leaves):
    """Test the read and write functions for Phylo2Vec objects

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    _test_and_save_load_newick(tmp_path, n_leaves, sample_vector)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_and_load_newick_matrix(tmp_path, n_leaves):
    """Test the read and write functions for Phylo2Vec objects

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    _test_and_save_load_newick(tmp_path, n_leaves, sample_matrix)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_newick_with_labels(tmp_path, n_leaves):
    """Test saving a Newick file with labels."""

    tr = Tree()
    tr.populate(n_leaves)
    nw_str = tr.write(parser=9)

    # Create an str-to-str mapping and create an integer Newick
    # key: node, value: label
    nw_int, label_mapping = create_label_mapping(nw_str)

    v = from_newick(nw_int)

    newick_path = tmp_path / "test.newick"

    save_newick(v, newick_path, label_mapping)

    with open(newick_path, "r", encoding="utf-8") as f:
        nw_str2 = f.read().strip()

    assert nw_str == nw_str2


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_load_newick_vector_from_str(n_leaves):
    v = sample_vector(n_leaves)
    nw_str = to_newick(v)
    v2 = load_newick(nw_str)
    assert np.array_equal(v, v2)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_load_newick_matrix_from_str(n_leaves):
    m = sample_matrix(n_leaves)
    nw_str = to_newick(m)
    m2 = load_newick(nw_str)
    assert np.array_equal(m, m2)


class TestSaveEdgeCases:
    def test_save_empty(self):
        v = np.array(0)
        with pytest.raises(ValueError):
            save(v, "test.csv")

    def test_save_ndim3(self):
        # array with 3 dimensions
        t = np.zeros((MIN_N_LEAVES, 3, 1))
        # Check that we raise a ValueError
        with pytest.raises(ValueError):
            save(t, "test.csv")
