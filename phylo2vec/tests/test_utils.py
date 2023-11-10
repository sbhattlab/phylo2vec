"""Tests for utility functions."""
import pytest

from phylo2vec.tests.config import MIN_N_LEAVES, MAX_N_LEAVES, N_REPEATS
from phylo2vec.base.to_newick import to_newick
from phylo2vec.utils import (
    check_v,
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


if __name__ == "__main__":
    pytest.main()
