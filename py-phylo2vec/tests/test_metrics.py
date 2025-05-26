"""Tests for metrics."""

import numpy as np
import pytest

from ete3 import Tree

from phylo2vec.base.newick import to_newick
from phylo2vec.metrics import cophenetic_distances, pairwise_distances
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.vector import sample_vector
from .config import MIN_N_LEAVES, N_REPEATS

# Function is currently a bit slow for large trees,
# so we limit the number of leaves to 50
MAX_N_LEAVES_COPH = 50


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


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_COPH + 1))
def test_cophenetic_vector(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _cophenetic(n_leaves, sample_vector)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_COPH + 1))
def test_cophenetic_matrix(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _cophenetic(n_leaves, sample_matrix)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES_COPH + 1])
def test_pairwise_distances_cophenetic(n_leaves):
    """Test the `pairwise_distances` function with cophenetic distances.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    # Test with a vector
    vector = sample_vector(n_leaves)
    dmat_vector = pairwise_distances(vector, metric="cophenetic")
    dmat_vector2 = cophenetic_distances(vector)
    assert np.array_equal(dmat_vector, dmat_vector2)

    # Test with a matrix
    matrix = sample_matrix(n_leaves)
    dmat_matrix = pairwise_distances(matrix, metric="cophenetic")
    dmat_matrix2 = cophenetic_distances(matrix)
    assert np.array_equal(dmat_matrix, dmat_matrix2)


if __name__ == "__main__":
    pytest.main()
