"""Tests for utility functions."""

import secrets

import numpy as np
import pytest
from ete3 import Tree

from phylo2vec.base import to_newick
from phylo2vec.utils import (
    add_leaf,
    apply_label_mapping,
    check_v,
    create_label_mapping,
    find_num_leaves,
    get_common_ancestor,
    read_vector_csv,
    read_newick_file,
    read_newick_file_labeled,
    remove_leaf,
    sample_vector,
    write_vector_csv,
    write_newick_file,
    write_newick_file_labeled,
)

from .config import MAX_N_LEAVES, MIN_N_LEAVES, N_REPEATS


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES + 1])
def test_read_write_newick(tmp_path, n_leaves):
    """Test the read and write functions for Newick trees

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    v = sample_vector(n_leaves)
    newick = to_newick(v)
    write_newick_file(newick, tmp_path / "test.newick")
    newick2 = read_newick_file(tmp_path / "test.newick")
    assert newick == newick2
    write_newick_file(newick, tmp_path / "test.random")
    with pytest.raises(Exception):
        _ = read_newick_file(tmp_path / "test.random")


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES + 1])
def test_read_write_csv(tmp_path, n_leaves):
    """Test the read and write functions for CSV files

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    v = sample_vector(n_leaves)
    write_vector_csv(v, tmp_path / "test.csv")
    v2 = read_vector_csv(tmp_path / "test.csv")
    assert np.all(v == v2)
    write_vector_csv(v, tmp_path / "test.random")
    with pytest.raises(Exception):
        _ = read_vector_csv(tmp_path / "test.random")


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES + 1])
def test_read_write_newick_labeled(tmp_path, n_leaves):
    """Test the read and write functions for labeled Newick trees

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    t = Tree()
    t.populate(n_leaves)
    # Random Newick string
    newick_labeled = t.write(format=9)
    newick_int, label_mapping = create_label_mapping(newick_labeled)
    write_newick_file_labeled(newick_int, label_mapping, tmp_path / "test_labeled.newick")
    newick_read, label_mapping_read = read_newick_file_labeled(
        tmp_path / "test_labeled.newick"
    )
    assert newick_labeled == apply_label_mapping(newick_read, label_mapping_read)
    assert label_mapping == label_mapping_read


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_sample(n_leaves):
    """Test the sample function

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        check_v(v)  # Asserts that v is valid


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_find_num_leaves(n_leaves):
    """Test find_num_leaves

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        newick = to_newick(v)
        # Assert that find_num_leaves returns the true number of leaves in the the tree
        assert find_num_leaves(newick) == n_leaves


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_create_and_apply_label_mapping(n_leaves):
    """Test create_label_mapping and apply_label_mapping

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        # Random string Newick
        t = Tree()
        t.populate(n_leaves)
        nw_str = t.write(format=9)

        # Create an int-to-str label mapping and create an integer Newick
        nw_int, label_mapping = create_label_mapping(nw_str)

        # Apply the mapping to retrieve the string Newick
        new_nw_str = apply_label_mapping(nw_int, label_mapping)

        # We should have nw_str == new_nw_str
        assert nw_str == new_nw_str


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_remove_and_add(n_leaves):
    """Test removing and adding a node to a vector

    Sample a v, remove a random leaf and add it back
    Test if we recover the same v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)

        leaf = secrets.randbelow(n_leaves)

        v_sub, sis = remove_leaf(v, leaf)

        # To deal with increment when removing a leaf
        if sis >= leaf:
            sis -= 1

        v_add = add_leaf(v_sub, leaf, sis)

        assert np.array_equal(v, v_add)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_get_common_ancestor(n_leaves):
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)

        node1, node2 = np.random.choice(np.arange(2 * (n_leaves - 1)), 2, replace=False)

        p2v_common_ancestor = get_common_ancestor(v, node1, node2)

        nw = to_newick(v)

        tr = Tree(nw, format=8)

        ete3_common_ancestor = int(tr.get_common_ancestor(f"{node1}", f"{node2}").name)

        assert p2v_common_ancestor == ete3_common_ancestor


if __name__ == "__main__":
    pytest.main()
