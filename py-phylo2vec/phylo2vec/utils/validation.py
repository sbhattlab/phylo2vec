"""Utilities for Phylo2Vec vector validation."""

import numpy as np

from phylo2vec import _phylo2vec_core


def check_v(v: np.ndarray) -> None:
    """Input validation of a Phylo2Vec vector

    The input is checked to satisfy the Phylo2Vec constraints

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector
    """
    _phylo2vec_core.check_v(v.tolist())
