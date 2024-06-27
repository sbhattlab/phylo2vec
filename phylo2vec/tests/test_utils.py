"""Tests for utility functions."""

import random

import numpy as np
import pytest

from ete3 import Tree

from phylo2vec.tests.config import MIN_N_LEAVES, MAX_N_LEAVES, N_REPEATS
from phylo2vec.base import to_newick
from phylo2vec.utils import (
    add_leaf,
    apply_label_mapping,
    create_label_mapping,
    check_v,
    remove_leaf,
    sample,
    find_num_leaves,
)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_sample(n_leaves):
    """Test the sample function

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample(n_leaves)
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
        v = sample(n_leaves)
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
        v = sample(n_leaves)

        leaf = random.randint(0, n_leaves - 1)

        v_sub, sis = remove_leaf(v, leaf)

        # To deal with increment when removing a leaf
        if sis >= leaf:
            sis -= 1

        v_add = add_leaf(v_sub, leaf, sis)

        assert np.array_equal(v, v_add)


if __name__ == "__main__":
    pytest.main()
