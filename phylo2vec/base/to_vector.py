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

    return np.array(ancestry)


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

    return np.array(ancestry)


@nb.njit(cache=True)
def _find_cherries(ancestry):
    ancestry_sorted = ancestry[np.argsort(ancestry[:, -1]), :]

    small_children = nb.typed.Dict.empty(
        key_type=nb.types.int64, value_type=nb.types.int64
    )

    for i, row in enumerate(ancestry_sorted):
        c1, c2, p = row

        parent_c1, parent_c2 = small_children.get(c1, c1), small_children.get(c2, c2)

        small_children[p] = min(parent_c1, parent_c2)

        ancestry_sorted[i, :] = [parent_c1, parent_c2, max(parent_c1, parent_c2)]
    return ancestry_sorted


@nb.njit(cache=True)
def _order_cherries_no_parents(cherries):
    cherries_ = np.zeros((len(cherries), 4), dtype=np.int16) - 1
    cherries_[:, :-1] = cherries
    cherries_copy = cherries[:, :-1]

    n_leaves = cherries.shape[0]

    for i in range(n_leaves):
        next_internal = len(cherries) + i + 1

        max_leaf = -1

        d = set()

        for j, ch in enumerate(cherries_copy):
            c1, c2 = ch

            if c1 == -1 and c2 == -1:
                continue

            if not (c1 in d or c2 in d):
                leaf_tmp = ch[ch <= n_leaves]

                if leaf_tmp.shape[0] > 0:
                    max_leaf_tmp = leaf_tmp.max()
                    if max_leaf_tmp > max_leaf:
                        max_leaf = max_leaf_tmp
                        idx = j
                else:
                    idx = j
            d.add(c1)
            d.add(c2)

        cherries_[idx, -1] = next_internal

        cherries_copy[idx] = -1

    cherries_ = cherries_[cherries_[:, -1].argsort()]

    return cherries_[:, :-1]


@nb.njit(cache=True)
def _build_vector(cherries):
    v_res = np.zeros((cherries.shape[0],), dtype=np.int16)
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
    """Convert a newick string with parent labels to a vector

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
    """Convert a newick string without parent labels to a vector

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
