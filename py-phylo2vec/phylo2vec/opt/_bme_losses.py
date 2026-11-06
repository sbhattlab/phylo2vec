"""Balanced minimum evolution (BME): tree length and branch lengths."""

import numpy as np

from phylo2vec import _phylo2vec_core as core


def bme_objective(tree, dm):
    """Balanced minimum evolution (BME) length of a tree.

    Pauplin's tree length ``L = sum_{i < j} 2^(1 - tau_ij) * dm[i, j]``,
    where ``tau_ij`` is the number of edges between leaves ``i`` and ``j``
    in the unrooted tree (Pauplin, 2000). Lower is better.

    Parameters
    ----------
    tree : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2).
        Only the topology is used: the branch lengths of a matrix are ignored.
    dm : array_like
        Symmetric ``n x n`` distance matrix, indexed by leaf label.

    Returns
    -------
    float
        BME length of the tree.

    Raises
    ------
    ValueError
        If ``dm`` is not ``n x n``, with ``n`` the number of leaves of ``tree``.

    Examples
    --------
    >>> import numpy as np
    >>> from phylo2vec.opt import bme_objective
    >>> dm = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
    >>> bme_objective(np.array([0, 0]), dm)  # every pair is 2 edges apart
    2.5
    """
    return core.bme_objective(tree, np.asarray(dm, dtype=np.float64))


def bme_branch_lengths(tree, dm):
    """Balanced minimum evolution (BME) branch lengths for the topology of a tree.

    FastME/Desper & Gascuel (2002) BME lengths of the edges of the unrooted
    tree, given ``dm``. They sum to :func:`bme_objective`.

    Parameters
    ----------
    tree : numpy.ndarray
        Phylo2Vec vector (ndim == 1)/matrix (ndim == 2).
        Only the topology is used: the branch lengths of a matrix are ignored.
    dm : array_like
        Symmetric ``n x n`` distance matrix, indexed by leaf label.

    Returns
    -------
    numpy.ndarray
        Branch lengths of shape ``(n - 1, 2)``, in the layout of the branch
        lengths of a Phylo2Vec matrix: row ``i`` holds the lengths of the edges
        from internal node ``n + i`` to its two children. The unrooted tree has
        one edge through the root, whose length is split equally between the
        root's two child edges.

    Raises
    ------
    ValueError
        If ``dm`` is not ``n x n``, with ``n`` the number of leaves of ``tree``.

    Notes
    -----
    The lengths are not clamped and can be negative, a sign that the topology
    fits ``dm`` poorly. A Phylo2Vec matrix needs non-negative branch lengths,
    so clamp them before building one:
    ``np.column_stack([v, np.clip(bls, 0, None)])``.

    Examples
    --------
    On distances measured along a tree, the BME branch lengths are exact:

    >>> import numpy as np
    >>> from phylo2vec.opt import bme_branch_lengths
    >>> from phylo2vec.stats import cophenetic_distances
    >>> m = np.array([[0.0, 0.1, 0.2], [2.0, 0.3, 0.4], [2.0, 0.5, 0.6]])
    >>> bls = bme_branch_lengths(m, cophenetic_distances(m))
    >>> np.allclose(bls[:2], m[:2, 1:])
    True
    >>> np.allclose(bls[2], 0.5 * (0.5 + 0.6))  # root edge, split equally
    True
    """
    return core.bme_branch_lengths(tree, np.asarray(dm, dtype=np.float64))
