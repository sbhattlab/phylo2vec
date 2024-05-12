"""Tests for utility functions."""

import random

import numpy as np
import pytest

from phylo2vec.tests.config import MIN_N_LEAVES, MAX_N_LEAVES, N_REPEATS
from phylo2vec.base import to_newick
from phylo2vec.utils import (
    add_leaf,
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
        assert find_num_leaves(newick) == n_leaves


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
