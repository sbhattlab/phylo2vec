"""Utilities for Phylo2Vec vector validation."""
import numpy as np


def check_v(v):
    """Input validation of a Phylo2Vec vector

    The input is checked to satisfy the Phylo2Vec constraints

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector
    """
    k = len(v)

    v_max = 2 * np.arange(k)

    assert np.all((0 <= v) & (v <= v_max)), print(v, v >= 0, v <= v_max)
