"""Pairwise distance metrics for nodes within phylogenetic trees."""

from phylo2vec import _phylo2vec_core as core
from phylo2vec.utils.vector import check_vector


def cophenetic_distances(v, unrooted=False):
    """
    Compute the (topological) cophenetic distance matrix
    for tree nodes from a Phylo2Vec vector.

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector
    unrooted : bool, optional
        Whether to consider the tree as unrooted or not, by default False

    Returns
    -------
    numpy.ndarray
        Cophenetic distance matrix
    """
    return core.cophenetic_distances(v, unrooted)


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
