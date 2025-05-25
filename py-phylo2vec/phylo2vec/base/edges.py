"""
Methods to convert Phylo2Vec vectors to a list of edges and vice versa.
"""

from typing import List, Tuple

import numpy as np

import phylo2vec._phylo2vec_core as core


def from_edges(edges: List[Tuple[int, int]]) -> np.ndarray:
    """Convert a list of edges to a Phylo2Vec vector

    Each edge is represented as a list of two nodes (child, parent)

    Parameters
    ----------
    edges : List[Tuple[int, int]]
        List of (child, parent) edges

    Returns
    -------
    v : numpy.ndarray
        Phylo2Vec vector
    """
    v = core.from_edges(edges)
    return np.asarray(v)


def to_edges(v: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a Phylo2Vec vector to an edge list

    Each edge is represented as a list of two nodes (child, parent)

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector

    Returns
    -------
    edges : List[Tuple[int, int]]
        List of (child, parent) edges
    """
    edge_list = core.get_edges(v)
    return edge_list
