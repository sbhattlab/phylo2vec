"""Tests for metrics."""

import numpy as np
import pytest

from ete3 import Tree

from phylo2vec.base.newick import to_newick
from phylo2vec.metrics import cophenetic_distances
from phylo2vec.utils.vector import sample_vector
from .config import MIN_N_LEAVES, N_REPEATS


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, 51))
def test_cophenetic(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    def coph_ete3(tr, n_leaves):
        dmat = np.zeros((n_leaves, n_leaves))

        for i in range(n_leaves):
            for j in range(i):
                dmat[i, j] = tr.get_distance(f"{i}", f"{j}", topology_only=False)

        return dmat + dmat.T

    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)

        # tree with all branch lengths = 1
        tr = Tree(to_newick(v))

        # Our distance matrix
        dmat_p2v = cophenetic_distances(v)

        # ete3 distance matrix
        dmat_ete3 = coph_ete3(tr, n_leaves)

        assert np.array_equal(dmat_p2v, dmat_ete3)

        # Test for unrooted trees
        dmat_p2v_unr = cophenetic_distances(v, unrooted=True)

        tr.unroot()
        dmat_ete3_unr = coph_ete3(tr, n_leaves)

        assert np.array_equal(dmat_p2v_unr, dmat_ete3_unr)


if __name__ == "__main__":
    pytest.main()
