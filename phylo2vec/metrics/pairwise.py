import numba as nb
import numpy as np

from phylo2vec.base.to_newick import _get_ancestry
from phylo2vec.utils.validation import check_v


@nb.njit(cache=True)
def cophenetic_distances(v):
    # Should be very similar to dist_nodes in ape

    # Ancestry
    A = _get_ancestry(v)

    n_leaves = len(v) + 1

    # Distance matrix
    D = np.zeros((2 * n_leaves - 1, 2 * n_leaves - 1), dtype=np.uint32)

    # Keep track of visited nodes
    all_visited = []

    for i in range(n_leaves - 1):
        c1, c2, p = A[n_leaves - i - 2, :]

        for visited in all_visited[:-1]:
            dist_from_visited = D[p, visited] + 1
            # c1 to visited
            D[c1, visited] = dist_from_visited
            D[visited, c1] = dist_from_visited
            # c2 to visited
            D[c2, visited] = dist_from_visited
            D[visited, c2] = dist_from_visited

        # c1 to c2: path length = 2
        D[c1, c2] = 2
        D[c2, c1] = 2
        # c1 to parent: path length = 1
        D[c1, p] = 1
        D[p, c1] = 1
        # c2 to parent: path length = 1
        D[c2, p] = 1
        D[p, c2] = 1

        all_visited.extend([c1, c2, p])

    return D[:n_leaves, :n_leaves]


PAIRWISE_DISTANCES = {"cophenetic": cophenetic_distances}


def pairwise_distances(v, metric="cophenetic"):
    check_v(v)

    func = PAIRWISE_DISTANCES[metric]

    return func(v)
