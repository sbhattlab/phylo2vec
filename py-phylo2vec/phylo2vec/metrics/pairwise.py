"""Pairwise distance metrics for nodes within phylogenetic trees."""

import warnings
import numpy as np

from phylo2vec import _phylo2vec_core as core
from phylo2vec.utils.vector import check_vector


def cophenetic_distances(vector_or_matrix, unrooted=False):
    """
    Compute the (topological) cophenetic distance matrix
    for tree nodes from a Phylo2Vec vector.

    Parameters
    ----------
    vector_or_matrix : numpy.array
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
    return np.asarray(coph)


PAIRWISE_DISTANCES = {"cophenetic": cophenetic_distances}


def pairwise_distances(v, metric="cophenetic"):
    """
    Compute a pairwise distance matrix
    for tree nodes from a Phylo2Vec vector.

    Currently, only the cophenetic distance is supported.

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector
    metric : str, optional
        Pairwise distance metric, by default "cophenetic"

    Returns
    -------
    numpy.ndarray
        Distance matrix
    """
    check_vector(v)

    func = PAIRWISE_DISTANCES[metric]

    return func(v)
