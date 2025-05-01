"""
Methods to convert a Phylo2Vec vector to a Newick-format string.
"""

import numpy as np

from phylo2vec import _phylo2vec_core


def _get_ancestry(v: np.ndarray) -> np.ndarray:
    """
    Get the "ancestry" of v (see "Returns" paragraph)

    v[i] = which BRANCH we do the pairing from

    The initial situation looks like this:
                      R
                      |
                      | --> branch 2
                    // \\
      branch 0 <-- //   \\  --> branch 1
                   0     1

    For v[1], we have 3 possible branches too choose from.
    v[1] = 0 or 1 indicates that we branch out from branch 0 or 1, respectively.
    The new branch yields leaf 2 (like in ordered trees)

    v[1] = 2 is somewhat similar: we create a new branch from R that yields leaf 2

    Parameters
    ----------
    v : numpy.array
        Phylo2Vec vector

    Returns
    -------
    ancestry : numpy.array
        Ancestry matrix
        1st column: child 1
        2nd column: child 2
        3rd column: parent node
    """
    ancestry_list = _phylo2vec_core.get_ancestry(v)
    return np.asarray(ancestry_list)


def to_newick(v):
    """Recover a rooted tree (in Newick format) from a Phylo2Vec v

    Parameters
    ----------
    v : numpy.array
        Phylo2Vec vector

    Returns
    -------
    newick : str
        Newick tree
    """
    return _phylo2vec_core.to_newick_from_vector(v)
