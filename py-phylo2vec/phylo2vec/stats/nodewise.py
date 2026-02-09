"""Nodewise metrics/statistics between nodes within phylogenetic trees."""

import numpy as np

from phylo2vec import _phylo2vec_core as core
from phylo2vec.utils.matrix import check_matrix
from phylo2vec.utils.vector import check_vector


def cophenetic_distances(tree, unrooted=False):
    """
    Compute the cophenetic distance matrix of a Phylo2Vec
    vector (topological) or matrix (from branch lengths).

    Parameters
    ----------
    tree : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)

    Returns
    -------
    numpy.ndarray
        Cophenetic distance matrix
    """
    return core.cophenetic_distances(tree, unrooted=unrooted)


NODEWISE_DISTANCES = {"cophenetic": cophenetic_distances}


def pairwise_distances(tree, metric="cophenetic"):
    """
    Compute a pairwise distance matrix
    for tree nodes from a Phylo2Vec vector.

    Currently, only the cophenetic distance is supported.

    Parameters
    ----------
    tree : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)
    metric : str, optional
        Pairwise distance metric, by default "cophenetic"

    Returns
    -------
    numpy.ndarray
        Distance matrix
    """
    func = NODEWISE_DISTANCES[metric]

    return func(tree)


def cov(tree):
    """
    Compute the covariance matrix of a Phylo2Vec vector or matrix.

    Adapted from `vcv.phylo` in <https://github.com/emmanuelparadis/ape>

    Parameters
    ----------
    tree : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)

    Returns
    -------
    vcv : numpy.ndarray
        Covariance matrix
    """
    return core.vcv(tree)


def precision(tree):
    """
    Compute the precision matrix of a Phylo2Vec vector or matrix.

    Adapted from: `inverseA.R` in <https://github.com/cran/MCMCglmm>

    Parameters
    ----------
    tree : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)

    Returns
    -------
    precision: numpy.ndarray
        Precision matrix
    """
    precursor = core.pre_precision(tree)

    # Schur complement of the precursor matrix
    n_leaves = tree.shape[0] + 1
    a = precursor[:n_leaves, :n_leaves]
    b = precursor[:n_leaves, n_leaves:]
    c = precursor[n_leaves:, n_leaves:]
    d = precursor[n_leaves:, :n_leaves]  # b.T

    return a - b @ np.linalg.solve(c, d)


# pylint: disable=redefined-builtin
def incidence(vector, format="coo"):
    """
    Compute the incidence matrix of a Phylo2Vec vector.

    Parameters
    ----------
    vector : numpy.ndarray
        Phylo2Vec vector (ndim == 1)
    format : str, optional
        Format of the incidence matrix, by default "coo"

    Returns
    -------
    numpy.ndarray or List[List[int], List[int], List[int]] or Dict[Tuple[int, int], int]
        Incidence matrix in the specified format
    """
    if vector.ndim != 1:
        raise ValueError("vector should be a 1D array")

    if format == "coo":
        return core.incidence_coo(vector)
    if format == "csr":
        return core.incidence_csr(vector)
    if format == "csc":
        return core.incidence_csc(vector)
    if format == "dense":
        return core.incidence_dense(vector)

    raise ValueError(f"Unknown format: {format}")
