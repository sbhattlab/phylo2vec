"""Phylo2Vec vector manipulation functions."""

import random

import numba as nb
import numpy as np

from ete3 import Tree

from phylo2vec.base.to_newick import _get_ancestry, to_newick
from phylo2vec.base.to_vector import (
    _build_vector,
    _find_cherries,
    to_vector,
)
from phylo2vec.utils.validation import check_v
from phylo2vec import _phylo2vec_core as core


def remove_leaf(v, leaf):
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
    return core.remove_leaf(v, leaf)


def add_leaf(v, leaf, pos):
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
    return core.add_leaf(v, leaf, pos)


def reorder_v(reorder_method, v_old, label_mapping_old):
    """Shuffle v by reordering leaf labels

    Current pipeline: get ancestry matrix --> reorder --> re-build vector

    Parameters
    ----------
    reorder_fun : function
        Function used to reorder the ancestry matrix
    v_old : numpy.ndarray or list
        Current Phylo2vec vector
    label_mapping_old : dict[int, str]
        Current mapping of node label (integer) to taxa

    Returns
    -------
    v_new : numpy.ndarray or list
        New Phylo2vec vector
    label_mapping_new : nb.types.Dict
        New integer-taxon dictionary
    """
    # TODO: make this function inplace?
    # Get ancestry
    ancestry_old = _get_ancestry(v_old)

    # Reorder M
    if reorder_method == "birth_death":
        reorder_fun = _reorder_birth_death
    elif reorder_method == "bfs":
        reorder_fun = _reorder_bfs
    else:
        raise ValueError("`method` must be 'birth_death' or 'bfs'")

    # Pass the dict to Numba
    label_mapping_old_ = nb.typed.Dict.empty(
        key_type=nb.types.uint16, value_type=nb.types.unicode_type
    )

    for k, v in label_mapping_old.items():
        label_mapping_old_[k] = v

    ancestry_new, label_mapping_new = reorder_fun(
        np.flip(ancestry_old, axis=0), label_mapping_old_
    )

    # Re-build v
    v_new = _build_vector(_find_cherries(ancestry_new))

    return v_new, label_mapping_new


@nb.njit
def _reorder_birth_death(
    ancestry_old, label_mapping_old, reorder_internal=True, shuffle_cols=False
):
    """Reorder v as a birth-death process (i.e., an "ordered" vector)

    Parameters
    ----------
    ancestry_old : numpy.ndarray
        Ancestry matrix
        1st column: child 1 parent node
        2nd column: child 2
        3rd column: parent node
    label_mapping_old : dict[int, str]
        Mapping of leaf labels (integer) to taxa
    reorder_internal : bool, optional
        If True, reorder internal labels, by default True
    shuffle_cols : bool, optional
        If True, shuffle children columns in the ancestry, by default False

    Returns
    -------
    ancestry_new : numpy.ndarray
        Reordered ancestry matrix
    label_mapping_new :
        Reordered mapping of leaf labels (integer) to taxa
    """
    # Copy old M
    ancestry_new = ancestry_old.copy()

    # Internal nodes to visit (2*len(M_old) = root label)
    to_visit = [2 * len(ancestry_old)]

    # Number of visits
    visits = 1

    # Internal labels
    internal_labels = list(range(len(ancestry_old) + 1, 2 * len(ancestry_old)))

    # Leaf "code"
    node_code = []

    # List of all visited nodes
    visited = []

    # List of visited internal nodes
    visited_internals = []

    # Taxa dict to be updated
    label_mapping_new = nb.typed.Dict.empty(
        key_type=nb.types.uint16, value_type=nb.types.unicode_type
    )

    while len(to_visit) > 0:
        row = 2 * len(ancestry_old) - to_visit.pop(0)

        if node_code:
            next_pair = [node_code[visited.index(visited_internals.pop(0))], visits]
        else:
            next_pair = [0, 1]

        if shuffle_cols:
            col_order = 2 * random.randint(0, 1) - 1
            ancestry_old[row, :2] = ancestry_old[row, :2][::col_order]
            next_pair = next_pair[::col_order]

        for i, child in enumerate(ancestry_old[row, :2]):
            if child < len(ancestry_old) + 1:
                label_mapping_new[next_pair[i]] = label_mapping_old[child]

                ancestry_new[row, i] = next_pair[i]

            # Not a leaf node --> add it to the visit list
            else:
                visited_internals.append(child)
                if reorder_internal:
                    # Basically, flip the nodes
                    # Ex: relabel 7 in M_old as 9 in M_new
                    # Then relabel 9 in M_old as 7 in M_new
                    internal_node = internal_labels.pop()
                    ancestry_new[row, i] = internal_node
                    ancestry_new[2 * len(ancestry_new) - ancestry_old[row, i], 2] = (
                        ancestry_new[row, i]
                    )

                to_visit.append(child)

        visited.extend(ancestry_old[row, :2])

        node_code.extend(next_pair)
        visits += 1

    # Re-sort M such that the root node R is the first row, then internal nodes R-1, R-2, ...
    ancestry_new = ancestry_new[ancestry_new[:, 2].argsort()[::-1]]

    return ancestry_new, label_mapping_new


@nb.njit(cache=True)
def _reorder_bfs(ancestry_old, label_mapping_old):
    # Copy old M
    ancestry_new = ancestry_old.copy()

    # Internal nodes to visit (2*len(M_old) = root label)
    to_visit = [2 * len(ancestry_old)]

    # Leaf order
    order = []

    # Taxa dict to be updated
    label_mapping_new = nb.typed.Dict.empty(
        key_type=nb.types.uint16, value_type=nb.types.unicode_type
    )

    while len(to_visit) > 0:
        # Current row of M
        row = 2 * len(ancestry_old) - to_visit.pop(0)

        for i, child in enumerate(ancestry_old[row, :-1]):
            # Leaf node
            if child < len(ancestry_old) + 1:
                order.append(child)

                # Update taxa dict
                label_mapping_new[len(order) - 1] = label_mapping_old[child]

                # Update M_new
                ancestry_new[row, i] = len(order) - 1

            # Not a leaf node --> add it to the visit list
            else:
                to_visit.append(child)

    return ancestry_new, label_mapping_new


def reroot_at_random(v):
    """Reroot a tree (via its Phylo2Vec vector v) at a random node

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec representation of a tree

    Returns
    -------
    numpy.ndarray
        rerooted v
    """
    ete3_tree = Tree(to_newick(v), format=8)

    ete3_tree.set_outgroup(f"{random.randint(0, 2 * len(v) - 1)}")

    newick = ete3_tree.write(format=9)

    v_new = to_vector(newick)

    check_v(v_new)

    return to_vector(newick)


# faster than np.nonzero
@nb.njit(cache=True)
def _find_indices_of_first_leaf(ancestry, leaf):
    for r in range(ancestry.shape[0]):
        for c in range(ancestry.shape[1]):
            if ancestry[r, c] == leaf:
                return r, c


def get_ancestry_paths(v):
    """
    Get the ancestry paths for each node in the Phylo2Vec vector.

    Parameters
    ----------
    v : numpy.ndarray
        Phylo2Vec vector

    Returns
    -------
    ancestry_paths : list of list of int
        Ancestry paths for each node
    """
    ancestry = _get_ancestry(v)
    parent_vec = np.zeros(2 * len(v), dtype=np.uint64)

    for i in range(len(ancestry)):
        parent_vec[ancestry[i, :2]] = ancestry[i, 2]

    ancestry_paths = []
    for i in range(2 * len(v)):
        path = [i]
        while (2 * len(v)) not in path:
            path.insert(0, parent_vec[int(path[0])])
        ancestry_paths.append(path)

    return ancestry_paths


def get_common_ancestor(v, node1, node2):
    """Get the first recent common ancestor between two nodes in a Phylo2Vec tree

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
    paths = get_ancestry_paths(v)
    path1 = paths[node1]
    path2 = paths[node2]
    common_path = np.intersect1d(path1, path2)
    mrca = common_path[0]
    return mrca
