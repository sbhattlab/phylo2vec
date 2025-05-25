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
    if vector_or_matrix.ndim == 2:
        newick = core.to_newick_from_matrix(vector_or_matrix)
    elif vector_or_matrix.ndim == 1:
        newick = core.to_newick_from_vector(vector_or_matrix)
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )

    return newick
