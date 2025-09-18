"""Tests for utility functions."""

import secrets

import numpy as np
import pytest

from ete4 import Tree

from phylo2vec.base.newick import to_newick
from phylo2vec.utils.matrix import check_matrix, sample_matrix
from phylo2vec.utils.vector import (
    add_leaf,
    check_vector,
    get_common_ancestor,
    remove_leaf,
    sample_vector,
)
from phylo2vec.utils.newick import (
    apply_label_mapping,
    create_label_mapping,
    find_num_leaves,
)

from .config import MAX_N_LEAVES, MIN_N_LEAVES, N_REPEATS


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_sample_vector(n_leaves):
    """Test the sample_vector function

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        check_vector(v)  # Asserts that v is valid


class TestSampleVectorEdgeCases:
    def test_sample_vector_negative(self):
        """Test the sample_vector function with negative n_leaves"""
        with pytest.raises(ValueError):
            sample_vector(-1)

    def test_sample_vector_zero(self):
        """Test the sample_vector function with zero leaves"""
        with pytest.raises(ValueError):
            sample_vector(0)

    def test_sample_vector_two(self):
        """Test the sample_vector function with two leaves"""
        v = sample_vector(2)
        check_vector(v)  # Asserts that v is valid


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_sample_matrix(n_leaves):
    """Test the sample_vector function

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)
        check_matrix(m)  # Asserts that m is valid


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
        nw_str = t.write(parser=9)

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
    """Test get_common_ancestor against ete

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)

        node1, node2 = np.random.choice(np.arange(2 * (n_leaves - 1)), 2, replace=False)

        p2v_common_ancestor = get_common_ancestor(v, node1, node2)

        nw = to_newick(v)

        tr = Tree(nw, parser=8)

        ete_common_ancestor = int(tr.common_ancestor(f"{node1}", f"{node2}").name)

        assert p2v_common_ancestor == ete_common_ancestor


if __name__ == "__main__":
    pytest.main()
