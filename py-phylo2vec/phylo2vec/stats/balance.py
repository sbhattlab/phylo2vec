"""Tree balance indices for characterizing tree topology shape."""

import numpy as np

from phylo2vec import _phylo2vec_core as core


def sackin(vector_or_matrix: np.ndarray) -> int:
    """Compute the Sackin index of a phylogenetic tree.

    The Sackin index is the sum of depths of all leaves. Higher values
    indicate more imbalanced trees.

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        Phylo2Vec vector (ndim == 1) or matrix (ndim == 2).
        Only topology is used; branch lengths are ignored.

    Returns
    -------
    int
        Sackin index.

    References
    ----------
    Sackin MJ (1972). "Good" and "Bad" phenograms.
    Systematic Biology, 21(2), 225-226.
    """
    if vector_or_matrix.ndim == 2:
        v = vector_or_matrix[:, 0].astype(int)
    elif vector_or_matrix.ndim == 1:
        v = vector_or_matrix
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )
    return core.sackin(v)


def b2(vector_or_matrix: np.ndarray) -> float:
    """Compute the B2 index of a phylogenetic tree.

    For a binary rooted tree, B2 = sum(d_i * 2^{-d_i}) where d_i is the
    depth of leaf i. This has a probabilistic interpretation: the entropy
    of reaching each leaf via equiprobable branching from the root.

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        Phylo2Vec vector (ndim == 1) or matrix (ndim == 2).
        Only topology is used; branch lengths are ignored.

    Returns
    -------
    float
        B2 index.

    References
    ----------
    Shao KT, Sokal RR (1990). Tree balance.
    Systematic Zoology, 39(3), 266-276.
    """
    if vector_or_matrix.ndim == 2:
        v = vector_or_matrix[:, 0].astype(int)
    elif vector_or_matrix.ndim == 1:
        v = vector_or_matrix
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )
    return core.b2(v)


def leaf_depth_variance(vector_or_matrix: np.ndarray) -> float:
    """Compute the variance of leaf depths of a phylogenetic tree.

    Perfectly balanced trees have a variance of 0 (all leaves at the same
    depth). Ladder (caterpillar) trees have the highest variance for a
    given number of leaves.

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        Phylo2Vec vector (ndim == 1) or matrix (ndim == 2).
        Only topology is used; branch lengths are ignored.

    Returns
    -------
    float
        Variance of leaf depths.
    """
    if vector_or_matrix.ndim == 2:
        v = vector_or_matrix[:, 0].astype(int)
    elif vector_or_matrix.ndim == 1:
        v = vector_or_matrix
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )
    return core.leaf_depth_variance(v)
