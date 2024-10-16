"""Random utility functions: sampling and seeding."""

import os
import random

import numba as nb
import numpy as np


@nb.njit
def sample(n_leaves, ordered=False):
    """Sample a random tree via Phylo2Vec

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    ordered : bool, optional
        If True, sample an ordered tree, by default False

        True:
        v_i in {0, 1, ..., i} for i in (0, n_leaves-1)

        False:
        v_i in {0, 1, ..., 2*i} for i in (0, n_leaves-1)

    Returns
    -------
    numpy.ndarray
        Phylo2Vec vector
    """

    if ordered:
        v_list = [np.random.randint(0, i + 1) for i in range(n_leaves - 1)]
    else:
        v_list = [np.random.randint(0, 2 * i + 1) for i in range(n_leaves - 1)]
    return np.array(v_list, dtype=np.uint16)


def seed_everything(seed):
    """Seed random, the Python hash seed, numpy

    Parameters
    ----------
    seed : int
        Random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
