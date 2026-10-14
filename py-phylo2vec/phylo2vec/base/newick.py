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

    Examples
    --------
    Convert a topology-only Newick string to a vector:

    >>> import phylo2vec as p2v
    >>> p2v.from_newick("((0,1),(2,3));")
    array([0, 2, 2])

    With branch lengths, returns a matrix:

    >>> m = p2v.from_newick("(0:0.1,1:0.2)2;")
    >>> m.shape
    (1, 3)
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

    Examples
    --------
    >>> import numpy as np
    >>> import phylo2vec as p2v
    >>> v = np.array([0, 2, 2])
    >>> p2v.to_newick(v)
    '((0,1)5,(2,3)4)6;'

    With a matrix (includes branch lengths):

    >>> m = np.array([[0.0, 0.5, 0.8], [1.0, 0.7, 0.6]])
    >>> p2v.to_newick(m)
    '(0:0.7,(1:0.5,2:0.8)3:0.6)4;'
    """
    return core.to_newick(vector_or_matrix)
