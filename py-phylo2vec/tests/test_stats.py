"""Tests for metrics."""

import numpy as np
import pytest

from ete4 import Tree
from scipy import sparse

from phylo2vec.base.newick import to_newick
from phylo2vec.stats.balance import b2, leaf_depth_variance, sackin
from phylo2vec.stats.nodewise import (
    cophenetic_distances,
    cov,
    incidence,
    pairwise_distances,
    precision,
)
from phylo2vec.stats.treewise import robinson_foulds
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.vector import sample_vector

from .config import MIN_N_LEAVES, N_REPEATS


MAX_N_LEAVES_STATS = 50


def _test_cophenetic(n_leaves, sample_fn, topological, unrooted=False):
    """Helper function to test cophenetic distances using a sample function."""

    def coph_ete(tr, n_leaves):
        dmat = np.zeros((n_leaves, n_leaves))

        for i in range(n_leaves):
            for j in range(i):
                dist = tr.get_distance(f"{i}", f"{j}", topological=topological)
                dmat[i, j] = dist
                dmat[j, i] = dist

        return dmat

    for _ in range(N_REPEATS):
        vector_or_matrix = sample_fn(n_leaves)

        # Our distance matrix
        dmat_p2v = cophenetic_distances(vector_or_matrix, unrooted=unrooted)

        # tree with all branch lengths = 1
        tr = Tree(to_newick(vector_or_matrix), parser=1)

        if unrooted:
            # Unroot by deleting the max node
            # ete systematically unroots by removing the first non-leaf node
            # Our approach is to remove the node with the highest index
            # (see rust implementation)
            names = [int(tr.children[0].name), int(tr.children[1].name)]
            if names[0] > names[1]:
                tr.children[0].delete()
            else:
                tr.children[1].delete()

        # ete distance matrix
        dmat_ete = coph_ete(tr, n_leaves)

        assert np.allclose(dmat_p2v, dmat_ete)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cophenetic_vector(n_leaves):
    """
    Test that the cophenetic distance matrix matches the ete implementation
    for rooted trees without branch lengths.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _test_cophenetic(n_leaves, sample_vector, topological=True)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cophenetic_vector_unrooted(n_leaves):
    """
    Test that the cophenetic distance matrix matches the ete implementation
    for unrooted trees without branch lengths.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _test_cophenetic(n_leaves, sample_vector, topological=True, unrooted=True)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cophenetic_matrix(n_leaves):
    """
    Test that the cophenetic distance matrix matches the ete implementation
    for rooted trees with branch lengths.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _test_cophenetic(n_leaves, sample_matrix, topological=False)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cophenetic_matrix_unrooted(n_leaves):
    """
    Test that the cophenetic distance matrix matches the ete implementation
    for unrooted trees with branch lengths.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    _test_cophenetic(n_leaves, sample_matrix, topological=False, unrooted=True)


@pytest.mark.parametrize("n_leaves", [MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1])
def test_pairwise_distances_cophenetic(n_leaves):
    """Test the `pairwise_distances` function with cophenetic distances.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    # Test with a vector
    vector = sample_vector(n_leaves)
    dmat_vector = pairwise_distances(vector, metric="cophenetic")
    dmat_vector2 = cophenetic_distances(vector)
    assert np.array_equal(dmat_vector, dmat_vector2)

    # Test with a matrix
    matrix = sample_matrix(n_leaves)
    dmat_matrix = pairwise_distances(matrix, metric="cophenetic")
    dmat_matrix2 = cophenetic_distances(matrix)
    assert np.array_equal(dmat_matrix, dmat_matrix2)


def _test_cov_and_precision(n_leaves, sample_fn):
    for _ in range(N_REPEATS):
        vector_or_matrix = sample_fn(n_leaves)
        cov_matrix = cov(vector_or_matrix)
        precision_matrix = precision(vector_or_matrix)
        identity = np.eye(n_leaves, dtype=cov_matrix.dtype)
        # Note: this test might fail for larger n_leaves due to numerical precision issues
        assert np.allclose(cov_matrix @ precision_matrix, identity)
        assert np.allclose(precision_matrix @ cov_matrix, identity)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cov_vector(n_leaves):
    """Test covariance matrix for vector input.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    _test_cov_and_precision(n_leaves, sample_vector)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_cov_matrix(n_leaves):
    """Test covariance matrix for matrix input.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    _test_cov_and_precision(n_leaves, sample_matrix)


@pytest.mark.parametrize(
    "v, d",
    [
        (np.array([0]), np.array([[1, 0], [0, 1], [-1, -1]])),
        (
            np.array([0, 1]),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, -1, -1, 1],
                    [-1, 0, 0, -1],
                ]
            ),
        ),
        (
            np.array([0, 1, 2]),
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, -1, -1, 1, 0],
                    [0, -1, 0, 0, -1, 1],
                    [-1, 0, 0, 0, 0, -1],
                ]
            ),
        ),
    ],
)
def test_incidence(v, d):
    """Test incidence matrix for different formats.

    Parameters
    ----------
    v : np.ndarray
        Input vector.
    d : np.ndarray
        Expected incidence matrix.
    """
    format_fns = {
        "coo": sparse.coo_matrix,
        "csr": sparse.csr_matrix,
        "csc": sparse.csc_matrix,
        "dense": lambda x: x,
    }
    for f, func in format_fns.items():
        inc = incidence(v, format=f)
        if f == "dense":
            assert np.array_equal(inc, d)
        elif f == "coo":
            data, rows, cols = inc
            assert np.array_equal(func((data, (rows, cols))).toarray(), d)
        else:
            assert np.array_equal(func(inc).toarray(), d)


# Robinson-Foulds tests


def _test_robinson_foulds_identical(n_leaves, sample_fn):
    """Helper function to test RF distance for identical trees."""
    vector_or_matrix = sample_fn(n_leaves)
    assert robinson_foulds(vector_or_matrix, vector_or_matrix) == 0.0


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_robinson_foulds_identical_vectors(n_leaves):
    """Test that identical trees have RF distance 0."""
    _test_robinson_foulds_identical(n_leaves, sample_vector)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_robinson_foulds_identical_matrices(n_leaves):
    """Test that identical matrices have RF distance 0."""
    _test_robinson_foulds_identical(n_leaves, sample_matrix)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_robinson_foulds_symmetric(n_leaves):
    """Test that RF distance is symmetric."""
    v1 = sample_vector(n_leaves)
    v2 = sample_vector(n_leaves)
    assert robinson_foulds(v1, v2) == robinson_foulds(v2, v1)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_robinson_foulds_normalized_bounds(n_leaves):
    """Test that normalized RF distance is in [0, 1]."""
    v1 = sample_vector(n_leaves)
    v2 = sample_vector(n_leaves)
    rf_norm = robinson_foulds(v1, v2, normalize=True)
    assert 0.0 <= rf_norm <= 1.0


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_robinson_foulds_matches_ete4(n_leaves):
    """Test that our RF distance matches ete4's implementation."""
    for _ in range(N_REPEATS):
        v1 = sample_vector(n_leaves)
        v2 = sample_vector(n_leaves)

        # Our implementation
        rf_ours = robinson_foulds(v1, v2, normalize=False)

        # Check that our RF matches ete4's RF
        t1 = Tree(to_newick(v1), parser=1)
        t2 = Tree(to_newick(v2), parser=1)
        t1.unroot()
        t2.unroot()
        result = t1.robinson_foulds(t2, unrooted_trees=True)
        rf_ete4 = result[0]

        assert rf_ours == rf_ete4, f"Mismatch: ours={rf_ours}, ete4={rf_ete4}"

        # Check that our normalized RF matches ete4's normalized RF
        max_rf = result[1]
        rf_ete4_norm = rf_ete4 / max_rf

        rf_ours_norm = robinson_foulds(v1, v2, normalize=True)

        assert np.isclose(
            rf_ours_norm, rf_ete4_norm
        ), f"Mismatch: ours={rf_ours_norm}, ete4={rf_ete4_norm}"


@pytest.mark.parametrize(
    "v, expected",
    [
        # Small trees
        (np.array([0]), 2),
        (np.array([0, 2, 2, 3]), 12),  # n=5
        # Ladder trees: S = n(n + 1) / 2 - 1
        (np.array([0, 0]), 5),
        (np.array([0, 0, 0]), 9),
        (np.array([0, 0, 0, 0]), 14),  # n=5 -> 5*6/2 - 1 = 14
        (np.zeros(49, dtype=int), 1274),  # n=50 -> 50*51/2 - 1 = 1274
        (np.zeros(99, dtype=int), 5049),  # n=100 -> 100*101/2 - 1 = 5049
        # Balanced trees: S = n * log2(n)
        (np.array([0, 2, 2]), 8),
        (np.array([0, 2, 2, 6, 4, 6, 6]), 24),  # n=8 -> 8*3 = 24
        (np.array([0, 2, 2, 6, 4, 6, 6, 14, 8, 10, 10, 14, 12, 14, 14]), 64),  # n=16 -> 16*4
    ],
)
def test_sackin(v, expected):
    assert sackin(v) == expected


@pytest.mark.parametrize(
    "v, expected",
    [
        # Small trees
        (np.array([0]), 0.0),  # n=2
        (np.array([0, 2, 2, 3]), 0.24),  # n=5
        # Balanced trees: Var = 0
        (np.array([0, 2, 2]), 0.0),  # n=4
        (np.array([0, 2, 2, 6, 4, 6, 6]), 0.0),  # n=8
        (np.array([0, 2, 2, 6, 4, 6, 6, 14, 8, 10, 10, 14, 12, 14, 14]), 0.0),  # n=16
        # Ladder trees: Var = (n-1)(n-2)(n^2+3n-6) / (12n^2)
        (np.array([0, 0]), 0.2222222),  # n=3
        (np.array([0, 0, 0]), 0.6875),  # n=4
        (np.array([0, 0, 0, 0]), 1.36),  # n=5
        (np.zeros(49, dtype=int), 207.2896),  # n=50
    ],
)
def test_leaf_depth_variance(v, expected):
    assert np.isclose(leaf_depth_variance(v), expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "v, expected",
    [
        # Small trees
        (np.array([0]), 1.0),
        (np.array([0, 2, 2, 3]), 2.25),
        # Ladder trees: B2 = 2 - 2^(2 - n)
        (np.array([0, 0]), 1.5),  # n=3 -> 2 - 2^(-1)
        (np.array([0, 0, 0]), 1.75),  # n=4 -> 2 - 2^(-2)
        (np.array([0, 0, 0, 0]), 1.875),  # n=5 -> 2 - 2^(-3)
        (np.array([0, 0, 0, 0, 0]), 1.9375),  # n=6 -> 2 - 2^(-4)
        (np.zeros(49, dtype=int), 2.0),  # n=50 -> ~2.0
        (np.zeros(99, dtype=int), 2.0),  # n=100 -> ~2.0
        # Balanced trees: B2 = log2(n)
        (np.array([0, 2, 2]), 2.0),  # n=4 -> log2(4) = 2
        (np.array([0, 2, 2, 6, 4, 6, 6]), 3.0),  # n=8 -> log2(8) = 3
        (np.array([0, 2, 2, 6, 4, 6, 6, 14, 8, 10, 10, 14, 12, 14, 14]), 4.0),  # n=16
    ],
)
def test_b2(v, expected):
    assert np.isclose(b2(v), expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_balance_indices_matrix(n_leaves):
    """Test that balance indices give the same result for vector and matrix input."""
    v = sample_vector(n_leaves)
    m = sample_matrix(n_leaves)
    m[:, 0] = v  # same topology, different branch lengths

    assert sackin(m) == sackin(v)
    assert leaf_depth_variance(m) == leaf_depth_variance(v)
    assert b2(m) == b2(v)
