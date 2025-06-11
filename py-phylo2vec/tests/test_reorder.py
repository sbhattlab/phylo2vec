"""Temporary test of reordering functions against legacy algorithms (v0.x)"""

import random
import string

from itertools import product

import numpy as np
import pytest

from ete3 import Tree

from phylo2vec.base.ancestry import from_ancestry, to_ancestry
from phylo2vec.base.newick import to_newick
from phylo2vec.base.pairs import from_pairs
from phylo2vec.utils.newick import apply_label_mapping
from phylo2vec.utils.vector import queue_shuffle, sample_vector
from .config import MIN_N_LEAVES, MAX_N_LEAVES, N_REPEATS


def legacy_queue_shuffle(
    ancestry_old, label_mapping_old, reorder_internal=False, shuffle_cols=False
):
    """
    Legacy version of the queue shuffle algorithm (called _reorder_birth_death in v0.x)

    Reorders v as a birth-death process (i.e., an "ordered" vector)

    Removed numba dependency as speed is not an issue for these tests

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
    label_mapping_new : dict[int, str]
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
    label_mapping_new = {}

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

        for i, child in enumerate(ancestry_old[row, 1:]):
            if child < len(ancestry_old) + 1:
                label_mapping_new[next_pair[i]] = label_mapping_old[child]

                ancestry_new[row, i + 1] = next_pair[i]

            # Not a leaf node --> add it to the visit list
            else:
                visited_internals.append(child)
                if reorder_internal:
                    # Basically, flip the nodes
                    # Ex: relabel 7 in M_old as 9 in ancestry_new
                    # Then relabel 9 in M_old as 7 in ancestry_new
                    internal_node = internal_labels.pop()
                    ancestry_new[row, i + 1] = internal_node
                    ancestry_new[
                        2 * len(ancestry_new) - ancestry_old[row, i + 1], 0
                    ] = ancestry_new[row, i + 1]

                to_visit.append(child)

        visited.extend(ancestry_old[row, 1:])
        node_code.extend(next_pair)
        visits += 1

    # Re-sort M such that the root node R is the first row, then internal nodes R-1, R-2, ...
    ancestry_new = ancestry_new[ancestry_new[:, 0].argsort()[::-1]]

    return np.flip(ancestry_new), label_mapping_new, node_code


# def to_pairs(ancestry):
#     num_cherries = ancestry.shape[0]
#     ancestry = ancestry[np.argsort(ancestry[:, 2])]
#     num_nodes = 2 * num_cherries + 2
#     MAX = np.iinfo(np.int32).max
#     min_desc = np.ones(num_nodes, dtype=np.int32) * MAX
#     pairs = []

#     for i, cherry in enumerate(ancestry):
#         c1, c2, p = cherry

#         min_desc1 = min_desc[c1] if min_desc[c1] != MAX else c1
#         min_desc2 = min_desc[c2] if min_desc[c2] != MAX else c2

#         desc_min, _ = sorted([min_desc1, min_desc2])
#         min_desc[p] = desc_min
#         pairs.append((int(min_desc1), int(min_desc2)))
#     return pairs


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES))
def test_queue_shuffle(n_leaves):
    """Test the legacy queue shuffle algorithm

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        # Sample a vector and convert it to an ancestry matrix
        v = sample_vector(n_leaves)

        taxa = product(string.ascii_lowercase, repeat=2)
        dict_mapping_old = {i: "".join(next(taxa)) for i in range(n_leaves)}

        ancestry_old = to_ancestry(v)
        # the output ancestry is difficult to use without adding more dependencies
        # as it doesn't respect the ordering of internal nodes in an ancestry matrix
        # best solution it to use the `node code`, which was a precursor of the pair format
        _, dict_mapping_py, node_code = legacy_queue_shuffle(
            np.flip(ancestry_old), dict_mapping_old
        )
        pairs_new = [
            (node_code[i], node_code[i + 1]) for i in range(0, len(node_code), 2)
        ][::-1]

        # Convert back to vector and check if it matches the original vector

        v_new_legacy = from_pairs(pairs_new)

        v_new_rust, vec_mapping_rust = queue_shuffle(v, shuffle_cherries=False)

        assert np.array_equal(v_new_legacy, v_new_rust)

        # Convert the rust list to a dict
        dict_mapping_rust = {
            i: dict_mapping_old[idx] for i, idx in enumerate(vec_mapping_rust)
        }

        assert dict_mapping_py == dict_mapping_rust
