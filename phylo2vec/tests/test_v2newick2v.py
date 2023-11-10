"""Test conversion from v to newick and back to v"""
import numpy as np
import pytest

from ete3 import Tree

from phylo2vec.tests.config import MIN_N_LEAVES, MAX_N_LEAVES, N_REPEATS
from phylo2vec.base import to_newick, to_vector, to_vector_no_parents
from phylo2vec.base.to_vector import (
    _find_cherries,
    _order_cherries_no_parents,
    _reduce,
    _reduce_no_parents,
)
from phylo2vec.utils import sample


MIN_N_LEAVES = 5
MAX_N_LEAVES = 200
N_REPEATS = 10


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_v2newick2v(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample(n_leaves)
        newick = to_newick(v)
        v2 = to_vector(newick)
        assert np.all(v == v2)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_cherries_no_parents(n_leaves):
    """Test that the functions of to_vector_no_parents and to_vector do the same thing

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample(n_leaves)
        newick = to_newick(v)
        newick_no_parents = Tree(newick).write(format=9)
        cherries = _find_cherries(_reduce(newick))
        cherries_no_parents = _order_cherries_no_parents(
            _reduce_no_parents(newick_no_parents)
        )

        assert np.array_equal(cherries, cherries_no_parents)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_cherry_permutations(n_leaves):
    """Simple way to check that permutation of leaf nodes does not change v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    def permutate_cherries(newick):
        for i, char in enumerate(newick):
            if char == "(":
                open_idx = i + 1
            elif char == ")" and open_idx != -1:
                child1, child2 = newick[open_idx:i].split(",", 2)

                # Switch the cherries
                newick = newick.replace(f"({child1},{child2})", f"({child2},{child1})")

                open_idx = -1
        return newick

    for _ in range(N_REPEATS):
        nw = to_newick(sample(n_leaves))
        nw_perm = permutate_cherries(nw)
        assert np.array_equal(to_vector(nw), to_vector(nw_perm))


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_ladderize(n_leaves):
    """Simple way to check that isomorphic Newick strings have the same v

    ete3's ladderize should create an isomorphism the original Newick string

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        nw = to_newick(sample(n_leaves))
        tr = Tree(nw)

        # Ladderize the tree and output a new Newick string
        tr.ladderize()

        nw_ladderized = tr.write(format=9)

        assert np.array_equal(to_vector(nw), to_vector_no_parents(nw_ladderized))


if __name__ == "__main__":
    pytest.main()
