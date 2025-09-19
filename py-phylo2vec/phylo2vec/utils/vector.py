"""Phylo2Vec vector manipulation functions."""

import random
import warnings

from typing import List, Tuple

import numpy as np

from ete4 import Tree

from phylo2vec import _phylo2vec_core as core
from phylo2vec.base.newick import from_newick, to_newick


def check_vector(v: np.ndarray) -> None:
    """Input validation of a Phylo2Vec vector

    The input is checked to satisfy the Phylo2Vec constraints

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector
    """
    core.check_v(v.tolist())


def sample_vector(n_leaves: int, ordered: bool = False) -> np.ndarray:
    """Sample a random tree via Phylo2Vec, in vector form.

    Parameters
    ----------
    n_leaves : int
        Number of leaves (>= 2)
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

    v_list = core.sample_vector(n_leaves, ordered)
    return np.asarray(v_list)


def remove_leaf(v, leaf) -> Tuple[np.ndarray, int]:
    """Remove a leaf from a Phylo2Vec v

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector
    leaf : int
        A leaf node to remove

    Returns
    -------
    v_sub : numpy.ndarray
        Phylo2Vec vector without `leaf`
    sister : int
        Sister node of leaf
    """
    v_sub, sister = core.remove_leaf(v, leaf)

    return np.asarray(v_sub), sister


def add_leaf(v, leaf, pos) -> np.ndarray:
    """Add a leaf to a Phylo2Vec vector v

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector
    leaf : int >= 0
        A leaf node to add
    pos : int >= 0
        A branch from where the leaf will be added

    Returns
    -------
    v_add : numpy.ndarray
        Phylo2Vec vector including the new leaf
    """
    v_add = core.add_leaf(v, leaf, pos)
    return np.asarray(v_add)


def queue_shuffle(v, shuffle_cherries=False) -> Tuple[np.ndarray, List[int]]:
    """
    Produce an ordered version (i.e., birth-death process version)
    of a Phylo2Vec vector using the Queue Shuffle algorithm.

    Queue Shuffle ensures that the output tree is ordered,
    while also ensuring a smooth path through the space of orderings

    For more details, see https://doi.org/10.1093/gbe/evad213
    Illustration of the algorithm:
                    ////-3
                ////6|
        ////7|      \\\\-2
        |     |
    -8|      \\\\-1
        |
        |      ////-4
        \\\\5|
                \\\\-0

    The ancestry array of this tree is:
    [8, 7, 5]
    [7, 6, 1]
    [6, 3, 2]
    [5, 4, 0]

    Unrolled, it becomes:
    8 7 5 6 1 3 2 4 0

    We encode the nodes as it:
    Start by encoding the first two non-root nodes as 0, 1
    For the next pairs:
        * The left member takes the label was the previous parent node
        * The right member increments the previous right member by 1

    Ex:
    8 7 5 6 1 3 2 4 0
        0 1 0 2

    then

    8 7 5 6 1 3 2 4 0
        0 1 0 2 1 3

    then

    8 7 5 6 1 3 2 4 0
        0 1 0 2 1 3 0 4

    The code for the leaf nodes (0, 1, 2, 3, 4) is their new label

    Note that the full algorithm also features a queue of internal nodes
    which could switch the processing order of rows in the ancestry array.

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector
    shuffle_cherries : bool, optional
        If True, shuffle at random the order of the cherries in the ancestry matrix
        (i.e., the first two columns of the ancestry matrix).

    Returns
    -------
    v_new : numpy.ndarray
        Reordered Phylo2Vec vector
    vec_mapping : List[int]
        Mapping of the original vector to the new vector
        index: leaf
        value: new leaf index in the reordered vector
    """
    v_new, vec_mapping = core.queue_shuffle(v, shuffle_cherries)

    return np.asarray(v_new), vec_mapping


def reorder_v(reorder_method, v_old, label_mapping_old, shuffle_cols=False):
    """Shuffle v by reordering leaf labels

    Current pipeline: get ancestry matrix --> reorder --> re-build vector

    Note: reordering functions are not up to date.
    They will be integrated in the Rust core in the future

    Parameters
    ----------
    reorder_fun : function
        Function used to reorder the ancestry matrix
    v_old : numpy.ndarray or list
        Current Phylo2vec vector
    label_mapping_old : dict[int, str]
        Current mapping of node label (integer) to taxa
        Note: Will be deprecated in a future release; see `queue_shuffle` instead.
    shuffle_cols : bool, optional
        If True, shuffle at random the order of the cherries in the ancestry matrix
        (i.e., the first two columns of the ancestry matrix).


    Returns
    -------
    v_new : numpy.ndarray or list
        New Phylo2vec vector
    label_mapping_new : dict[int, str]
        New integer-taxon dictionary
    """

    warnings.warn(
        (
            "`reorder_v` is deprecated and will be removed "
            "in a future version. Use `queue_shuffle` instead."
        ),
        FutureWarning,
    )

    # Reorder M
    if reorder_method == "birth_death":
        v_new, vec_mapping = queue_shuffle(v_old, shuffle_cherries=shuffle_cols)
        # For compatibility with the old label mapping (for _hc)
        label_mapping_new = {
            i: label_mapping_old[idx] for i, idx in enumerate(vec_mapping)
        }
    elif reorder_method == "bfs":
        raise ValueError(
            (
                "`bfs` method is no longer supported as erroneous. "
                "Use `birth_death` instead."
            )
        )
    else:
        raise ValueError(f"Unknown value for `reorder_method`: {reorder_method}")

    return v_new, label_mapping_new


def get_common_ancestor(v, node1, node2):
    """Get the first recent common ancestor between two nodes in a Phylo2Vec tree

    `node1` and `node2` can be leaf nodes (0 to n_leaves)
    or internal nodes (n_leaves to 2*(n_leaves-1)).

    Similar to `get_common_ancestor` in ETE, but for Phylo2Vec vectors.

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector
    node1 : int
        A node in the tree
    node2 : int
        A node in the tree

    Returns
    -------
    mrca : int
        Most recent common ancestor node between node1 and node2
    """
    if not (node1 >= 0 and node2 >= 0):
        raise ValueError("Nodes must be greater than or equal to 0")
    return core.get_common_ancestor(v, node1, node2)


def reroot(v, node) -> np.ndarray:
    """Reroot a tree (via its Phylo2Vec vector v) at a given node

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec representation of a tree
    node : int
        A node to reroot the tree at

        Must be a valid node in the tree, i.e., in the range [0, 2 * n_leaves - 1]

    Returns
    -------
    numpy.ndarray
        rerooted vector
    """
    ete_tree = Tree(to_newick(v), parser=8)

    ete_tree.set_outgroup(f"{node}")
    ete_tree.set_outgroup(f"{node}")

    newick = ete_tree.write(parser=9)

    v_new = from_newick(newick)

    check_vector(v_new)

    return from_newick(newick)


def reroot_at_random(v) -> np.ndarray:
    """Reroot a tree (via its Phylo2Vec vector v) at a random node

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec representation of a tree

    Returns
    -------
    numpy.ndarray
        rerooted vector
    """

    return reroot(v, random.randint(0, 2 * len(v) - 1))
