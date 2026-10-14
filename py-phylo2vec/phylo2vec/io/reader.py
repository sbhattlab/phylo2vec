"""Methods to process files or strings into Phylo2Vec vectors or matrices."""

import os

from io import StringIO, TextIOBase
from typing import Union

import numpy as np

from phylo2vec.base.newick import from_newick
from phylo2vec.utils.matrix import check_matrix
from phylo2vec.utils.vector import check_vector
from ._validation import check_path


def read(buffer: str, delimiter: str = ",") -> np.ndarray:
    """Parse a string buffer into a Phylo2Vec vector or matrix.

    Parameters
    ----------
    buffer : str
        String containing comma-separated values.
    delimiter : str, optional
        Character used to separate values, by default ","

    Returns
    -------
    numpy.ndarray
        A vector (ndim == 1) or matrix (ndim == 2)

    Examples
    --------
    >>> from phylo2vec.io.reader import read
    >>> v = read("0,1,2")
    >>> v
    array([0, 1, 2])
    """
    return load(StringIO(buffer), delimiter=delimiter)


def load(
    filepath_or_buffer: Union[str, TextIOBase], delimiter: str = ","
) -> np.ndarray:
    """Read a text/csv file into a Phylo2Vec vector or matrix.

    Parameters
    ----------
    filepath : str, path object or file-like object
        File path to read or StringIO object containing the data.
    delimiter : str, optional
        Character used to separate values, by default ","

    Returns
    -------
    arr : numpy.ndarray
        A vector (ndim == 1) or matrix (ndim == 2)
        which satisfies Phylo2Vec constraints

    Examples
    --------
    >>> import phylo2vec as p2v
    >>> v = p2v.load("tree.csv")  # doctest: +SKIP
    """
    if not hasattr(filepath_or_buffer, "read"):
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError(f"File not found: {filepath_or_buffer}")
        check_path(filepath_or_buffer, "array")
    # np.genfromtxt with dtype = None will infer the type
    # Using should solve the edge case:
    # load a matrix with 2 leaves ([[0, bl1, bl2]] = float)
    # vs. a vector with 4 leaves ([0, v1, v2] = int)
    arr = np.genfromtxt(filepath_or_buffer, delimiter=delimiter, dtype=None)

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

    Examples
    --------
    From a Newick string:

    >>> import phylo2vec as p2v
    >>> v = p2v.load_newick("((0,1),(2,3));")
    >>> v
    array([0, 2, 2])

    From a file:

    >>> v = p2v.load_newick("tree.nwk")  # doctest: +SKIP
    """
    if os.path.isfile(filepath_or_buffer):
        check_path(filepath_or_buffer, "newick")

        with open(filepath_or_buffer, "r", encoding="utf-8") as f:
            newick = f.read().strip()
    else:
        newick = filepath_or_buffer

    return from_newick(newick)
