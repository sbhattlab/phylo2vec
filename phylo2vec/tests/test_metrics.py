"""Tests for metrics."""

import numpy as np
import pytest
import rpy2.robjects as ro

from ete3 import Tree
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from phylo2vec.tests.config import MIN_N_LEAVES, MAX_N_LEAVES
from phylo2vec.base import to_newick
from phylo2vec.metrics import cophenetic_distances
from phylo2vec.utils import sample


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_cophenetic(n_leaves):
    """Test that v to newick to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(2):
        v = sample(n_leaves)

        # newick with all branch lengths = 1
        newick = Tree(to_newick(v)).write()

        # Our distance matrix
        D_p2v = cophenetic_distances(v)

        with localconverter(ro.default_converter + numpy2ri.converter):
            importr("ape")

            ro.globalenv["newick"] = newick
            ro.globalenv["n"] = len(v)

            D_ape = ro.r(
                """
                x <- read.tree(text = newick)

                D_ape <- cophenetic.phylo(x)[paste0(0:n),paste0(0:n)]
                """
            )

        assert np.array_equal(D_p2v, D_ape)


if __name__ == "__main__":
    pytest.main()
