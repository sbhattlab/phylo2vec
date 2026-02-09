"""Methods to convert a Phylo2Vec vector to Newick format and vice versa."""

import numpy as np

import phylo2vec._phylo2vec_core as core


def from_newick(newick: str) -> np.ndarray:
    """Convert a Newick string to a Phylo2Vec vector or matrix

    Parameters
    ----------
    newick : str
        Newick string for a tree

    Returns
    -------
    numpy.ndarray
        Phylo2Vec matrix if branch lengths are present, otherwise a vector
    """
    if core.has_branch_lengths(newick):
        arr = core.to_matrix(newick)
    else:
        arr = core.to_vector(newick)

    return np.asarray(arr)


def to_newick(vector_or_matrix: np.ndarray) -> str:
    """Convert a Phylo2Vec vector or matrix to Newick format

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)

    Returns
    -------
    newick : str
        Newick tree
    """
    return core.to_newick(vector_or_matrix)
