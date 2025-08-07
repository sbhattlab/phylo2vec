"""Methods to write Phylo2Vec vectors/matrices."""

from io import StringIO
from typing import Dict, Optional

import numpy as np

from phylo2vec.base.newick import to_newick
from phylo2vec.utils.newick import apply_label_mapping
from ._validation import check_path


def write(vector_or_matrix: np.ndarray, delimiter: str = ",") -> str:
    if vector_or_matrix.ndim == 2:
        buffer = StringIO()
        np.savetxt(buffer, vector_or_matrix, delimiter=delimiter)
        return buffer.getvalue()

    if vector_or_matrix.ndim == 1:
        return np.array2string(vector_or_matrix, separator=delimiter)[1:-1]

    raise ValueError(
        "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
    )


def save(vector_or_matrix: np.ndarray, filepath: str, delimiter: str = ",") -> None:
    """Save a Phylo2Vec vector or matrix to a file.

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        A vector (ndim == 1) or matrix (ndim == 2)
        which satisfies Phylo2Vec constraints
    filepath : str
        Path to the output file
    delimiter : str, optional
        Delimiter to use for saving the file, by default ","
    """
    check_path(filepath, "array")
    if vector_or_matrix.ndim == 2:
        fmt = "%.18e"
    elif vector_or_matrix.ndim == 1:
        fmt = "%d"
    else:
        raise ValueError(
            "vector_or_matrix should either be a vector (ndim == 1) or matrix (ndim == 2)"
        )

    np.savetxt(filepath, vector_or_matrix, fmt=fmt, delimiter=delimiter)


def save_newick(
    vector_or_matrix: np.ndarray,
    filepath: str,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Save a Phylo2Vec vector or matrix to Newick format into a file.

    Parameters
    ----------
    vector_or_matrix : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2)
    filepath : str
        Path to the output file
    labels : Dict[str, str], optional
        A mapping of integer labels (as string) to taxon names, by default None
    """
    check_path(filepath, "newick")
    newick = to_newick(vector_or_matrix)

    if labels:
        newick = apply_label_mapping(newick, labels)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(newick)
