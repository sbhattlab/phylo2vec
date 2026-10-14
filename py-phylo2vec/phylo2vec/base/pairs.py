"""
Methods to convert Phylo2Vec vectors to a list of pairs and vice versa.
"""

import numpy as np
import phylo2vec._phylo2vec_core as core

from typing import List, Tuple


def from_pairs(pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Convert a list of pairs to a Phylo2Vec vector

    Each pair is represented as a tuple (B, L)
    indicating that leaf L descends from branch B.

    Parameters
    ----------
    pairs : List[Tuple[int, int]]
        List of (branch, leaf) pairs

    Returns
    -------
    v : numpy.ndarray
        Phylo2Vec vector

    Examples
    --------
    >>> import phylo2vec as p2v
    >>> pairs = [(1, 4), (0, 3), (0, 2), (0, 1)]
    >>> p2v.from_pairs(pairs)
    array([0, 0, 0, 1])
    """
    v = core.from_pairs(pairs)
    return np.asarray(v)


def to_pairs(v: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a Phylo2Vec vector to a list of pairs

    Each pair is represented as a tuple (B, L)
    indicating that leaf L descends from branch B.

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector

    Returns
    -------
    pairs : List[Tuple[int, int]]
        List of (branch, leaf) pairs

    Examples
    --------
    >>> import numpy as np
    >>> import phylo2vec as p2v
    >>> v = np.array([0, 0, 0, 1])
    >>> p2v.to_pairs(v)
    [(1, 4), (0, 3), (0, 2), (0, 1)]
    """
    pairs = core.get_pairs(v)
    return pairs
