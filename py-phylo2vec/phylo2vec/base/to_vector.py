"""
Methods to convert a Newick to a Phylo2Vec vector.

Two main methods:
    - to_vector for a Newick with parent labels
    - to_vector_no_parents for a Newick without parent labels
"""

import numba as nb
import numpy as np


def _reduce(newick):
    ancestry = []

    def do_reduce(ancestry, newick):
        for i, char in enumerate(newick):
            if char == "(":
                open_idx = i + 1
            elif char == ")":
                child1, child2 = newick[open_idx:i].split(",", 2)
                parent = newick[i + 1 :].split(",", 1)[0].split(")", 1)[0]

                ancestry.append(
                    [
                        int(child1),
                        int(child2),
                        int(parent),
                    ]
                )
                newick = newick[: open_idx - 1] + newick[i + 1 :]

                return do_reduce(ancestry, newick)

    do_reduce(ancestry, newick[:-1])

    return np.array(ancestry, dtype=np.int16)


def _reduce_no_parents(newick):
    ancestry = []

    def do_reduce(ancestry, newick):
        for i, char in enumerate(newick):
            if char == "(":
                open_idx = i + 1
            elif char == ")":
                child1, child2 = newick[open_idx:i].split(",", 2)

                child1 = int(child1)
                child2 = int(child2)

                ancestry.append([child1, child2, max(child1, child2)])

                newick = newick.replace(
                    newick[open_idx - 1 : i + 1], f"{min(child1, child2)}"
                )

                return do_reduce(ancestry, newick)

    do_reduce(ancestry, newick[:-1])

    return np.array(ancestry, dtype=np.int16)


@nb.njit(cache=True)
def _find_cherries(ancestry):
    ancestry_sorted = ancestry[np.argsort(ancestry[:, -1]), :]

    small_children = nb.typed.Dict.empty(
        key_type=nb.types.int16, value_type=nb.types.int16
    )

    for i, row in enumerate(ancestry_sorted):
        c1, c2, p = row

        parent_c1, parent_c2 = small_children.get(c1, c1), small_children.get(c2, c2)

        small_children[p] = min(parent_c1, parent_c2)

        ancestry_sorted[i, :] = [parent_c1, parent_c2, max(parent_c1, parent_c2)]
    return ancestry_sorted


@nb.njit(cache=True)
def _order_cherries_no_parents(cherries):
    old_cherries = cherries.copy()

    n_cherries = cherries.shape[0]

    idxs = np.zeros((n_cherries,), dtype=np.int32)

    for i in range(n_cherries):
        d = np.zeros((2 * n_cherries,), dtype=np.uint8)
        max_leaf = -1

        for j, ch in enumerate(old_cherries):
            c1, c2, _ = ch

            if idxs[j] == 1:
                continue

            if not (d[c1] == 1 or d[c2] == 1):
                if c1 <= n_cherries and c2 > n_cherries:
                    if c1 > max_leaf:
                        max_leaf = c1
                        idx = j
                elif c2 <= n_cherries and c1 > n_cherries:
                    if c2 > max_leaf:
                        max_leaf = c2
                        idx = j
                elif c1 <= n_cherries and c2 <= n_cherries:
                    c_max = max(c1, c2)
                    if c_max > max_leaf:
                        max_leaf = c_max
                        idx = j
                else:
                    idx = j

            d[c1] = 1
            d[c2] = 1

        idxs[idx] = 1  # idx

        cherries[i] = old_cherries[idx]

    return cherries


@nb.njit(cache=True)
def _build_vector(cherries):
    v_res = np.zeros((cherries.shape[0],), dtype=np.uint16)
    for i in range(cherries.shape[0] - 1, -1, -1):
        c1, c2, _ = cherries[i]

        c_max = max(c1, c2)

        subset = cherries[cherries[:, -1] <= c_max][:, :-1]

        idx = np.where(subset == c_max)[0][0]

        if idx == 0:
            v_res[c_max - 1] = min(c1, c2)
        else:
            v_res[c_max - 1] = c_max - 1 + idx

    return v_res


def to_vector(newick):
    """Convert a Newick string with parent labels to a vector

    Parameters
    ----------
    newick : str
        Newick string for a tree

    Returns
    -------
    v : numpy.ndarray
        Phylo2Vec vector
    """
    ancestry = _reduce(newick)

    cherries = _find_cherries(ancestry)

    v = _build_vector(cherries)

    return v


def to_vector_no_parents(newick_no_parents):
    """Convert a Newick string without parent labels to a vector

    Parameters
    ----------
    newick_no_parents : str
        Newick string for a tree

    Returns
    -------
    v : numpy.ndarray
        Phylo2Vec vector
    """
    ancestry = _reduce_no_parents(newick_no_parents)

    cherries = _order_cherries_no_parents(ancestry)

    v = _build_vector(cherries)

    return v
