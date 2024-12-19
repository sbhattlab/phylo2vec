"""Random utility functions: sampling and seeding."""

import os
import random

import numpy as np

from phylo2vec import _phylo2vec_core

def sample(n_leaves: int, ordered: bool = False) -> np.ndarray:
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

    v_list = _phylo2vec_core.sample(n_leaves, ordered)
    return np.asarray(v_list)


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
