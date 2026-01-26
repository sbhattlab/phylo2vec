"""
Distance metrics between phylogenetic trees.
"""

import numpy as np

from phylo2vec import _phylo2vec_core as core


def robinson_foulds(
    tree1: np.ndarray,
    tree2: np.ndarray,
    normalize: bool = False,
) -> float:
    """
    Compute the Robinson-Foulds distance between two trees.

    RF distance counts the number of bipartitions (splits) that differ
    between two tree topologies. Lower values indicate more similar trees.

    Parameters
    ----------
    tree1 : np.ndarray
        First tree as Phylo2Vec vector (1D) or matrix (2D).
        Only topology is used; branch lengths are ignored.
    tree2 : np.ndarray
        Second tree as Phylo2Vec vector (1D) or matrix (2D).
        Only topology is used; branch lengths are ignored.
    normalize : bool, default=False
        If True, return normalized distance in range [0.0, 1.0].

    Returns
    -------
    float
        RF distance. Integer value if normalize=False, float in [0,1] otherwise.

    Raises
    ------
    AssertionError
        If trees have different numbers of leaves.

    Examples
    --------
    >>> import numpy as np
    >>> from phylo2vec.stats import robinson_foulds
    >>> v1 = np.array([0, 1, 2, 3], dtype=np.int16)
    >>> v2 = np.array([0, 0, 1, 2], dtype=np.int16)
    >>> robinson_foulds(v1, v1)  # Identical trees
    0.0
    >>> robinson_foulds(v1, v2)  # Different trees
    2.0

    See Also
    --------
    ete3.Tree.robinson_foulds : Reference implementation in ete3
    ape::dist.topo : Reference implementation in R's ape package
    """
    # Extract topology (column 0) if matrix input
    v1 = tree1[:, 0].astype(int).tolist() if tree1.ndim == 2 else tree1.tolist()
    v2 = tree2[:, 0].astype(int).tolist() if tree2.ndim == 2 else tree2.tolist()

    return core.robinson_foulds(v1, v2, normalize)
