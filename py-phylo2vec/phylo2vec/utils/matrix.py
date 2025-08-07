"""Phylo2Vec matrix manipulation functions."""

import numpy as np

from phylo2vec import _phylo2vec_core as core


def check_matrix(m: np.ndarray) -> None:
    """Input validation of a Phylo2Vec matrix

    The input is checked to satisfy the Phylo2Vec constraints

    Parameters
    ----------
    m : numpy.ndarray
        Phylo2Vec matrix
    """
    core.check_m(m)


def sample_matrix(n_leaves: int, ordered: bool = False) -> np.ndarray:
    """Sample a random tree with branch lengths via Phylo2Vec, in matrix form.

    Parameters
    ----------
    n_leaves : int
        Number of leaves (>= 2)
    ordered : bool, optional
        If True, sample an ordered tree, by default False

    Returns
    -------
    numpy.ndarray
        Phylo2Vec matrix
        Dimensions (n_leaves, 3)
        1st column: Phylo2Vec vector
        2nd and 3rd columns: branch lengths of cherry [i] in the ancestry matrix
    """

    return core.sample_matrix(n_leaves, ordered)
