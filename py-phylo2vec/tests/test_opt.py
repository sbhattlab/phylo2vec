"""Tests for phylo2vec.opt"""

import subprocess
import sys

import numpy as np
import pytest

from phylo2vec.opt import bme_branch_lengths, bme_objective
from phylo2vec.stats import cophenetic_distances
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.vector import sample_vector

from .config import MIN_N_LEAVES, N_REPEATS

MAX_N_LEAVES_BME = 50

# Hide jax, so that the [gradme] extras look missing even if they are installed
HIDE_JAX = "import sys; sys.modules['jax'] = None; "

# 3-leaf star [0, 0]: every pair is 2 edges apart
THREE_LEAF_DM = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.5], [2.0, 2.5, 0.0]])

# 4-leaf tree [0, 2, 2], i.e. ((0,1),(2,3)): pairs within a cherry are 2 edges apart,
# the others 3. Its ancestry is [[2, 3, 4], [0, 1, 5], [5, 4, 6]].
FOUR_LEAF_DM = np.array(
    [
        [0.0, 0.3, 0.8, 0.9],
        [0.3, 0.0, 0.7, 0.6],
        [0.8, 0.7, 0.0, 0.4],
        [0.9, 0.6, 0.4, 0.0],
    ]
)


def run_without_gradme_extras(code):
    """Run code in a fresh interpreter, leaving the test session's imports as is."""
    return subprocess.run(
        [sys.executable, "-c", HIDE_JAX + code],
        capture_output=True,
        text=True,
        check=False,
    )


def random_distances(n_leaves, rng):
    """Random symmetric distance matrix with a zero diagonal."""
    upper = np.triu(rng.uniform(0.1, 1.0, size=(n_leaves, n_leaves)), k=1)
    return upper + upper.T


def brute_force_bme(v, dm):
    """Pauplin's formula with unrooted topological distances.

    Only valid from 3 leaves: cophenetic_distances counts 2 edges between the
    leaves of a 2-leaf tree, where BME counts 1.
    """
    tau = cophenetic_distances(v, unrooted=True)
    i, j = np.triu_indices(len(dm), k=1)
    return np.sum(2.0 ** (1.0 - tau[i, j]) * dm[i, j])


# Optional dependencies


def test_opt_imports_without_gradme_extras():
    """Without the [gradme] extras, phylo2vec.opt imports and leaves GradME out."""
    result = run_without_gradme_extras(
        "import phylo2vec.opt as opt; print(opt.list_methods(), opt.__all__)"
    )
    assert result.returncode == 0, result.stderr
    expected = (
        "['HillClimbing'] "
        "['HillClimbing', 'bme_branch_lengths', 'bme_objective', 'list_methods']"
    )
    assert result.stdout.strip() == expected


def test_gradme_without_extras_says_what_to_install():
    """Asking for GradME without the [gradme] extras names the extras to install."""
    result = run_without_gradme_extras("from phylo2vec.opt import GradME")
    assert result.returncode != 0
    assert "ImportError: GradME needs the optional dependencies" in result.stderr
    assert "pip install phylo2vec[gradme]" in result.stderr


def test_unknown_name_without_gradme_extras():
    """Other missing names still raise Python's usual AttributeError."""
    result = run_without_gradme_extras("import phylo2vec.opt as opt; opt.missing")
    assert result.returncode != 0
    expected = "AttributeError: module 'phylo2vec.opt' has no attribute 'missing'"
    assert expected in result.stderr


# BME: distance matrix check


@pytest.mark.parametrize("func", [bme_objective, bme_branch_lengths])
@pytest.mark.parametrize("shape", [(3, 3), (5, 5), (4, 3), (3, 4)])
def test_bme_wrong_distance_shape(func, shape):
    """A distance matrix of the wrong shape raises a ValueError, not a Rust panic."""
    with pytest.raises(ValueError, match="Distance matrix must be 4 x 4"):
        func(np.array([0, 2, 2]), np.zeros(shape))


# bme_objective


def test_bme_objective_null_trees():
    """1 leaf: no pairs. 2 leaves: one edge between them, so L = D01."""
    assert bme_objective([], [[0.0]]) == 0.0
    assert bme_objective([0], [[0.0, 0.7], [0.7, 0.0]]) == pytest.approx(0.7)


def test_bme_objective_three_leaves():
    """Star: every pair is 2 edges apart, so L = (D01 + D02 + D12) / 2."""
    dm = THREE_LEAF_DM
    expected = 0.5 * (dm[0, 1] + dm[0, 2] + dm[1, 2])
    assert bme_objective(np.array([0, 0]), dm) == pytest.approx(expected)


def test_bme_objective_four_leaves():
    """((0,1),(2,3)): cherries weigh 1/2, the four cross pairs 1/4."""
    dm = FOUR_LEAF_DM
    expected = 0.5 * (dm[0, 1] + dm[2, 3]) + 0.25 * (
        dm[0, 2] + dm[0, 3] + dm[1, 2] + dm[1, 3]
    )
    assert bme_objective(np.array([0, 2, 2]), dm) == pytest.approx(expected)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_BME + 1))
def test_bme_objective_matches_brute_force(n_leaves):
    """Random trees and distances: same value as Pauplin's formula."""
    rng = np.random.default_rng(n_leaves)
    for _ in range(N_REPEATS):
        for ordered in (True, False):
            v = sample_vector(n_leaves, ordered=ordered)
            dm = random_distances(n_leaves, rng)
            assert bme_objective(v, dm) == pytest.approx(brute_force_bme(v, dm))


def test_bme_objective_matrix_input():
    """A matrix gives the same value as its vector: its branch lengths are ignored."""
    rng = np.random.default_rng(0)
    m = sample_matrix(20)
    dm = random_distances(20, rng)
    v = m[:, 0].astype(np.int64)
    m_other_bls = m.copy()
    m_other_bls[:, 1:] *= 3.0
    assert bme_objective(m, dm) == bme_objective(v, dm)
    assert bme_objective(m_other_bls, dm) == bme_objective(v, dm)


def test_bme_objective_array_like_distances():
    """Lists and integer arrays are accepted as distance matrices."""
    dm = [[0, 1, 2], [1, 0, 2], [2, 2, 0]]
    assert bme_objective([0, 0], dm) == 2.5
    assert bme_objective([0, 0], np.array(dm)) == 2.5


# bme_branch_lengths


def test_bme_branch_lengths_null_trees():
    """1 leaf: no edges. 2 leaves: the one edge gets D01, split over the root row."""
    assert bme_branch_lengths([], [[0.0]]).shape == (0, 2)
    np.testing.assert_allclose(
        bme_branch_lengths([0], [[0.0, 0.7], [0.7, 0.0]]), [[0.35, 0.35]]
    )


def test_bme_branch_lengths_four_leaves():
    """((0,1),(2,3)): Desper & Gascuel's closed forms for every edge."""
    dm = FOUR_LEAF_DM

    def pendant(x, y, z, w):
        # Edge of leaf x, with sibling y and other leaves z, w
        to_others = 0.5 * (dm[x, z] + dm[x, w])
        sibling_to_others = 0.5 * (dm[y, z] + dm[y, w])
        return 0.5 * (dm[x, y] + to_others - sibling_to_others)

    internal = 0.25 * (dm[0, 2] + dm[0, 3] + dm[1, 2] + dm[1, 3]) - 0.5 * (
        dm[0, 1] + dm[2, 3]
    )
    # Rows follow the ancestry [[2, 3, 4], [0, 1, 5], [5, 4, 6]]
    expected = [
        [pendant(2, 3, 0, 1), pendant(3, 2, 0, 1)],
        [pendant(0, 1, 2, 3), pendant(1, 0, 2, 3)],
        [0.5 * internal, 0.5 * internal],
    ]
    np.testing.assert_allclose(bme_branch_lengths(np.array([0, 2, 2]), dm), expected)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_BME + 1))
def test_bme_branch_lengths_sum_to_bme_objective(n_leaves):
    """Random trees and distances: the lengths sum to the BME length."""
    rng = np.random.default_rng(n_leaves)
    for _ in range(N_REPEATS):
        for ordered in (True, False):
            v = sample_vector(n_leaves, ordered=ordered)
            dm = random_distances(n_leaves, rng)
            total = bme_branch_lengths(v, dm).sum()
            assert total == pytest.approx(bme_objective(v, dm))


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_BME + 1))
def test_bme_branch_lengths_recover_tree_metric(n_leaves):
    """On distances measured along a tree, BME recovers its branch lengths.

    Only the split of the root edge between the root's two children is a convention.
    """
    for _ in range(N_REPEATS):
        for ordered in (True, False):
            m = sample_matrix(n_leaves, ordered=ordered)
            bls = bme_branch_lengths(m, cophenetic_distances(m))
            np.testing.assert_allclose(bls[:-1], m[:-1, 1:], atol=1e-12)
            root_edge = m[-1, 1] + m[-1, 2]
            np.testing.assert_allclose(bls[-1], [0.5 * root_edge] * 2, atol=1e-12)


def test_bme_branch_lengths_negative():
    """Negative lengths are returned as is, not clamped.

    The distances fit ((0,2),(1,3)) better than ((0,1),(2,3)), so the internal edge
    of ((0,1),(2,3)) is 0.25 * 4 * 0.6 - 0.5 * (1.0 + 1.0) = -0.4.
    """
    dm = np.array(
        [
            [0.0, 1.0, 0.6, 0.6],
            [1.0, 0.0, 0.6, 0.6],
            [0.6, 0.6, 0.0, 1.0],
            [0.6, 0.6, 1.0, 0.0],
        ]
    )
    bls = bme_branch_lengths(np.array([0, 2, 2]), dm)
    np.testing.assert_allclose(bls[2], [-0.2, -0.2])


def test_bme_branch_lengths_matrix_input():
    """A matrix gives the same lengths as its vector: its branch lengths are ignored."""
    rng = np.random.default_rng(0)
    m = sample_matrix(20)
    dm = random_distances(20, rng)
    v = m[:, 0].astype(np.int64)
    bls = bme_branch_lengths(m, dm)
    assert bls.shape == (19, 2)
    assert bls.dtype == np.float64
    np.testing.assert_array_equal(bls, bme_branch_lengths(v, dm))
