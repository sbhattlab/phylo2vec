"""Tests for I/O functions."""

import numpy as np
import pytest

from ete3 import Tree

from phylo2vec.base.newick import from_newick
from phylo2vec.io.reader import load, load_newick
from phylo2vec.io.writer import save, save_newick
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.newick import create_label_mapping
from phylo2vec.utils.vector import sample_vector
from .config import MAX_N_LEAVES, MIN_N_LEAVES


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_and_load_vector(tmp_path, n_leaves):
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

    # Unallowed file extension should trigger an AssertionError
    random_path = tmp_path / "test.random"
    np.savetxt(random_path, v, delimiter=",")
    with pytest.raises(AssertionError):
        save(v, random_path)
        _ = load(random_path)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_and_load_matrix(tmp_path, n_leaves):
    """Test the read and write functions for Phylo2Vec objects

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    m = sample_matrix(n_leaves)

    csv_path = tmp_path / "test.csv"
    save(m, csv_path)
    m2 = load(csv_path)
    assert np.allclose(m, m2)

    # Unallowed file extension should trigger an AssertionError
    random_path = tmp_path / "test.random"
    np.savetxt(random_path, m, delimiter=",")
    with pytest.raises(AssertionError):
        save(m, random_path)
        _ = load(random_path)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_and_load_newick_vector(tmp_path, n_leaves):
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

    # Unallowed file extension should trigger an AssertionError
    random_path = tmp_path / "test.random"
    np.savetxt(random_path, v, delimiter=",")
    with pytest.raises(AssertionError):
        save_newick(v, random_path)
        _ = load_newick(random_path)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_and_load_newick_matrix(tmp_path, n_leaves):
    """Test the read and write functions for Phylo2Vec objects

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    m = sample_matrix(n_leaves)

    newick_path = tmp_path / "test.newick"
    save_newick(m, newick_path)
    m2 = load_newick(newick_path)
    assert np.array_equal(m, m2)

    # Unallowed file extension should trigger an AssertionError
    random_path = tmp_path / "test.random"
    np.savetxt(random_path, m, delimiter=",")
    with pytest.raises(AssertionError):
        save_newick(m, random_path)
        _ = load_newick(random_path)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES])
def test_save_newick_with_labels(tmp_path, n_leaves):
    """Test saving a Newick file with labels."""

    tr = Tree()
    tr.populate(n_leaves)
    nw_str = tr.write(format=9)

    # Create an str-to-str mapping and create an integer Newick
    # key: node, value: label
    nw_int, label_mapping = create_label_mapping(nw_str)

    v = from_newick(nw_int)

    newick_path = tmp_path / "test.newick"

    save_newick(v, newick_path, label_mapping)

    with open(newick_path, "r", encoding="utf-8") as f:
        nw_str2 = f.read().strip()

    assert nw_str == nw_str2
