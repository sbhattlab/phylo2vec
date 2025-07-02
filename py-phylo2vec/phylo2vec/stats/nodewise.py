"""Pairwise distance metrics/statistics between nodes within phylogenetic trees."""

import warnings
import numpy as np

from phylo2vec import _phylo2vec_core as core
from phylo2vec.utils.matrix import check_matrix
from phylo2vec.utils.vector import check_vector


def cophenetic_distances(vector_or_matrix, unrooted=False):
    """
    Compute the cophenetic distance matrix of a Phylo2Vec
    vector (topological) or matrix (from branch lengths).

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)

    Returns
    -------
    numpy.ndarray
        Cophenetic distance matrix
    """
    if unrooted:
        warnings.warn(
            (
                "Argument `unrooted` is ignored. It is deprecated and "
                "will be removed in future versions. When ensuring "
                "compatibility with `ape` and `ete` (mode='keep'), the "
                "argument becomes unnecessary. "
            ),
            FutureWarning,
        )
    if vector_or_matrix.ndim == 2:
        coph = core.cophenetic_distances_with_bls(vector_or_matrix)
    elif vector_or_matrix.ndim == 1:
        coph = core.cophenetic_distances(vector_or_matrix)
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )
    return coph


NODEWISE_DISTANCES = {"cophenetic": cophenetic_distances}


def pairwise_distances(vector_or_matrix, metric="cophenetic"):
    """
    Compute a pairwise distance matrix
    for tree nodes from a Phylo2Vec vector.

    Currently, only the cophenetic distance is supported.

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)
    metric : str, optional
        Pairwise distance metric, by default "cophenetic"

    Returns
    -------
    numpy.ndarray
        Distance matrix
    """
    if vector_or_matrix.ndim == 2:
        check_matrix(vector_or_matrix)
    elif vector_or_matrix.ndim == 1:
        check_vector(vector_or_matrix)
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )

    func = NODEWISE_DISTANCES[metric]

    return func(vector_or_matrix)


def cov(vector_or_matrix):
    """
    Compute the covariance matrix of a Phylo2Vec vector or matrix.

    Adapted from `vcv.phylo` in <https://github.com/emmanuelparadis/ape>

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)

    Returns
    -------
    vcv : numpy.ndarray
        Covariance matrix
    """
    if vector_or_matrix.ndim == 2:
        vcv = core.vcv_with_bls(vector_or_matrix)
    elif vector_or_matrix.ndim == 1:
        vcv = core.vcv(vector_or_matrix)
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )
    return vcv


def precision(vector_or_matrix):
    """
    Compute the precision matrix of a Phylo2Vec vector or matrix.

    Adapted from: `inverseA.R` in <https://github.com/cran/MCMCglmm>

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)

    Returns
    -------
    precicion: numpy.ndarray
        Precision matrix
    """
    if vector_or_matrix.ndim == 2:
        precursor = core.pre_precision_with_bls(vector_or_matrix)
    elif vector_or_matrix.ndim == 1:
        precursor = core.pre_precision(vector_or_matrix)
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )

    # Schur complement of the precursor matrix
    n_leaves = vector_or_matrix.shape[0] + 1
    a = precursor[:n_leaves, :n_leaves].astype(np.float64)
    b = precursor[:n_leaves, n_leaves:].astype(np.float64)
    c = precursor[n_leaves:, n_leaves:].astype(np.float64)
    d = precursor[n_leaves:, :n_leaves].astype(np.float64)  # b.T

    return a - b @ np.linalg.solve(c, d)
