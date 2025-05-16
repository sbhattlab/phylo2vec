"""Tests for metrics."""

import numpy as np
import pytest

from ete3 import Tree

from phylo2vec.base.newick import to_newick
from phylo2vec.metrics import cophenetic_distances
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.vector import sample_vector
from .config import MIN_N_LEAVES, MAX_N_LEAVES, N_REPEATS

MAX_N_LEAVES_COPH = MAX_N_LEAVES // 4


def _cophenetic(n_leaves, sample_fn):
    """Helper function to test cophenetic distances using a sample function."""

    def coph_ete3(tr, n_leaves):
        dmat = np.zeros((n_leaves, n_leaves))

        for i in range(n_leaves):
            for j in range(i):
                dist = tr.get_distance(f"{i}", f"{j}", topology_only=False)
                dmat[i, j] = dist
                dmat[j, i] = dist

        return dmat

    for _ in range(N_REPEATS):
        vector_or_matrix = sample_fn(n_leaves)

        # tree with all branch lengths = 1
        tr = Tree(to_newick(vector_or_matrix))

        # Our distance matrix
        dmat_p2v = cophenetic_distances(vector_or_matrix)

        # ete3 distance matrix
        dmat_ete3 = coph_ete3(tr, n_leaves)

        assert np.allclose(dmat_p2v, dmat_ete3)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_COPH))
def test_cophenetic_vector(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _cophenetic(n_leaves, sample_vector)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_COPH))
def test_cophenetic_matrix(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _cophenetic(n_leaves, sample_matrix)


if __name__ == "__main__":
    pytest.main()
