"""Test conversion from v to newick and back to v"""

import numpy as np
import pytest

from ete3 import Tree

from phylo2vec.base.newick import from_newick, to_newick
from phylo2vec.utils.vector import sample_vector
from .config import MIN_N_LEAVES, MAX_N_LEAVES, N_REPEATS


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_v2newick2v(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        newick = to_newick(v)
        v2 = from_newick(newick)
        assert np.all(v == v2)


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
        nw = to_newick(sample_vector(n_leaves))
        nw_perm = permutate_cherries(nw)
        assert np.array_equal(from_newick(nw), from_newick(nw_perm))


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
        nw = to_newick(sample_vector(n_leaves))
        tr = Tree(nw)

        # Ladderize the tree and output a new Newick string
        tr.ladderize()

        nw_ladderized = tr.write(format=9)

        assert np.array_equal(from_newick(nw), from_newick(nw_ladderized))


if __name__ == "__main__":
    pytest.main()
