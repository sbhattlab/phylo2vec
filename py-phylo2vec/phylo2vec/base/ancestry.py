"""
Methods to convert Phylo2Vec vectors to an ancestry object and vice versa.
"""

import numpy as np

import phylo2vec._phylo2vec_core as core


def from_ancestry(ancestry: np.ndarray) -> np.ndarray:
    """Convert an "ancestry matrix" to a vector

    Parameters
    ----------
    ancestry : np.ndarray
        Ancestry matrix
        1st column: child 1
        2nd column: child 2
        3rd column: parent node

    Returns
    -------
    numpy.ndarray
        Phylo2Vec vector
    """
    v_list = core.from_ancestry(ancestry)
    return np.asarray(v_list)


def to_ancestry(v: np.ndarray) -> np.ndarray:
    """Convert a Phylo2Vec vector to an ancestry matrix

    v[i] indicates which branch we do the pairing from.

    The initial situation looks like this:

    0 and 1 are leaves, which form a cherry.
    2 is the parent node of 0 and 1.
    R is an "extra root" that is connected to 2.
    In terms of edges:
    0 -- 2 (branch 0)
    1 -- 2 (branch 1)
    2 -- R (branch 2)

    For v[1], we have 3 possible branches to choose from.
    v[1] = 0 or 1 indicates that we branch out from branch 0 or branch 1, respectively.
    The new branch yields leaf 2 (like in ordered trees).

    v[1] = 2 is somewhat similar: we create a new branch from R that yields leaf 2.

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector

    Returns
    -------
    ancestry : numpy.ndarray
        Ancestry matrix
        1st column: child 1
        2nd column: child 2
        3rd column: parent node
    """

    ancestry_list = core.get_ancestry(v)
    return np.asarray(ancestry_list)
