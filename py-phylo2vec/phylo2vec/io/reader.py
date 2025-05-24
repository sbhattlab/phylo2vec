"""Methods to process files or strings into Phylo2Vec vectors or matrices."""

import os

import numpy as np

from phylo2vec.base.newick import from_newick
from phylo2vec.utils.matrix import check_matrix
from phylo2vec.utils.vector import check_vector
from ._validation import check_path


def load(filepath: str, delimiter: str = ",") -> np.ndarray:
    """Read a text/csv file into a Phylo2Vec vector or matrix.

    Parameters
    ----------
    filepath : str or file-like object
        File path to read
    delimiter : str, optional
        Character used to separate values, by default ","

    Returns
    -------
    arr : numpy.ndarray
        A vector (ndim == 1) or matrix (ndim == 2)
        which satisfies Phylo2Vec constraints
    """
    check_path(filepath, "array")
    # np.genfromtxt with dtype = None will infer the type
    # Using should solve the edge case:
    # load a matrix with 2 leaves ([[0, bl1, bl2]] = float)
    # vs. a vector with 4 leaves ([0, v1, v2] = int)
    arr = np.genfromtxt(filepath, delimiter=delimiter, dtype=None)

    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        check_vector(arr)
    elif arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
        check_matrix(arr)
    else:
        raise ValueError(
            "Input file should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )

    return arr


def load_newick(filepath_or_buffer: str) -> np.ndarray:
    """Read a Newick string/file into a Phylo2Vec vector or matrix.

    Parameters
    ----------
    filepath_or_buffer : str or file-like object
        File path or string containing a Newick-formatted tree

    Returns
    -------
    numpy.ndarray
        A vector (ndim == 1) or matrix (ndim == 2)
        which satisfies Phylo2Vec constraints
    """
    if os.path.isfile(filepath_or_buffer):
        check_path(filepath_or_buffer, "newick")

        with open(filepath_or_buffer, "r", encoding="utf-8") as f:
            newick = f.read().strip()
    else:
        newick = filepath_or_buffer

    return from_newick(newick)
