"""
Methods to convert a Phylo2Vec vector to a Newick-format string.
"""

import numba as nb
import numpy as np

# Numba tuple type
value_type = nb.types.UniTuple(nb.types.int64, 2)


@nb.njit(cache=True)
def _get_pairs(v):
    pairs = []

    for i in range(len(v) - 1, -1, -1):
        next_leaf = i + 1
        if v[i] <= i:
            # If v[i] <= i, it's an easy BD
            # We now that the next pair to add now is (v[i], next_leaf)
            # (as the branch leading to v[i] gives birth to the next_leaf)
            # Why pairs.insert(0)? Let's take an example with [0, 0]
            # We initially have (0, 1), but 0 gives birth to 2 afterwards
            # So the "shallowest" pair is (0, 2)
            pairs.append((v[i], next_leaf))

    for j in range(1, len(v)):
        next_leaf = j + 1
        if v[j] == 2 * j:
            # 2*j = extra root ==> pairing = (0, next_leaf)
            pairs.append((0, next_leaf))
        elif v[j] > j:
            # If v[i] > i, it's not the branch leading v[i] that gives birth but an internal branch
            # Remark 1: it will not be the "shallowest" pair, so we do not insert it at position 0
            # len(pairs) = number of pairings we did so far
            # So what v[i] - len(pairs) gives us is the depth of the next pairing
            # And pairs[v[i] - len(pairs) - 1][0] is a node that we processed beforehand
            # which is deeper than the branch v[i]
            index = v[j] - 2 * j
            pairs.insert(index, (pairs[index - 1][0], next_leaf))

    return pairs


@nb.njit(cache=True)
def _get_ancestry(v):
    """
    Get the "ancestry" of v (see "Returns" paragraph)

    v[i] = which BRANCH we do the pairing from

    The initial situation looks like this:
                      R
                      |
                      | --> branch 2
                    // \\
      branch 0 <-- //   \\  --> branch 1
                   0     1

    For v[1], we have 3 possible branches too choose from.
    v[1] = 0 or 1 indicates that we branch out from branch 0 or 1, respectively.
    The new branch yields leaf 2 (like in ordered trees)

    v[1] = 2 is somewhat similar: we create a new branch from R that yields leaf 2

    Parameters
    ----------
    v : numpy.array
        Phylo2Vec vector

    Returns
    -------
    ancestry : numpy.array
        Ancestry matrix
        1st column: child 1
        2nd column: child 2
        3rd column: parent node
    """
    pairs = _get_pairs(v)

    # We have our pairs, we can now build our ancestry
    # Matrix with 3 columns: child1, child2, parent
    ancestry = np.zeros((len(pairs), 3), dtype=np.int16)

    # Dictionary to keep track of the following relationship: child->highest parent
    parents = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)

    # Dictionary to keep track of siblings (i.e., sister nodes)
    # siblings = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)

    # Leaves are number 0, 1, ..., n_leaves - 1, so the next parent is n_leaves
    next_parent = len(v) + 1

    for i, pair in enumerate(pairs):
        child1, child2 = pair

        parent_child1 = parents.get(child1, child1)
        parent_child2 = parents.get(child2, child2)

        ancestry[i, :] = [parent_child1, parent_child2, next_parent]

        # Change the parents of the current children
        parents[child1] = next_parent
        parents[child2] = next_parent

        next_parent += 1

    return ancestry


@nb.njit
def _build_newick(ancestry):
    """Build a Newick string from an "ancestry" array

    The input should always be 3-dimensional with the following format:
    1st column: child 1
    2nd column: child 2
    3rd column: parent node

    The matrix is processed such that we iteratively write a Newick string
    to describe the tree.

    Parameters
    ----------
    ancestry : numpy.ndarray
        "Ancestry" array of size (n_leaves - 1, 3)

    Returns
    -------
    newick : str
        Newick string
    """

    # TODO: drop the ancestry matrix form?
    ancestry_dict = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=value_type)

    for c1, c2, p in ancestry:
        ancestry_dict[p] = (c1, c2)

    newick = f"{_build_newick_inner(ancestry[-1][-1], ancestry_dict)};"

    return newick


@nb.njit
def _build_newick_inner(node, ancestry_dict):
    if node in ancestry_dict:
        c1, c2 = ancestry_dict.pop(node)
        return "".join(
            (
                "(",
                _build_newick_inner(c1, ancestry_dict),
                _build_newick_inner(c2, ancestry_dict),
                ")",
                f"{node}",
            )
        )

        #    ( "("
        #     + _build_newick_inner(c1, ancestry_dict)
        #     + ","
        #     + _build_newick_inner(c2, ancestry_dict)
        #     + ")"
        #     + f"{node}"
        # )
    else:
        return f"{node}"


@nb.njit
def to_newick(v):
    """Recover a rooted tree (in Newick format) from a Phylo2Vec v

    Parameters
    ----------
    v : numpy.array
        Phylo2Vec vector

    Returns
    -------
    newick : str
        Newick tree
    """
    ancestry = _get_ancestry(v)

    newick = _build_newick(ancestry)

    return newick
