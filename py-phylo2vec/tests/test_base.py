"""Test conversion from v to newick and back to v"""

import ete4
import numpy as np
import pytest

from ete4 import Tree

from phylo2vec.base.ancestry import from_ancestry, to_ancestry
from phylo2vec.base.edges import from_edges, to_edges
from phylo2vec.base.newick import from_newick, to_newick
from phylo2vec.base.pairs import from_pairs, to_pairs
from phylo2vec.utils.matrix import sample_matrix
from phylo2vec.utils.newick import remove_parent_labels
from phylo2vec.utils.vector import sample_vector
from .config import MIN_N_LEAVES, MAX_N_LEAVES, N_REPEATS


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_v2newick2v(n_leaves):
    """Test that v to newick to converted_v
    via `from_newick` leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        newick = to_newick(v)
        v2 = from_newick(newick)
        assert np.all(v == v2)


class TestToNewickEdgeCases:
    def test_to_newick_empty(self):
        # dim 0
        v0 = np.array(0)
        # Check that we raise a ValueError
        with pytest.raises(ValueError):
            to_newick(v0)

    def test_to_newick_ndim3(self):
        # array with 3 dimensions
        t = np.zeros((MIN_N_LEAVES, 3, 1))
        # Check that we raise a ValueError
        with pytest.raises(ValueError):
            to_newick(t)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_m2newick2m(n_leaves):
    """
    Test that m to newick to converted_m
    via `from_newick` leads to m == converted_m

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)
        newick = to_newick(m)
        m2 = from_newick(newick)
        assert np.allclose(m, m2, atol=1e-6)

        newick_no_parents = remove_parent_labels(newick)
        m3 = from_newick(newick_no_parents)
        assert np.allclose(m, m3, atol=1e-6)


def permutate_cherries(newick):
    for i, char in enumerate(newick):
        if char == "(":
            open_idx = i + 1
        elif char == ")" and open_idx != -1:
            child1, child2 = newick[open_idx:i].split(",", 2)

            # Switch the cherries
            newick = newick.replace(f"({child1},{child2})", f"({child2},{child1})")

            open_idx = -1
    return newick


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_newick_cherry_permutations(n_leaves):
    """Simple way to check that permutation of leaf nodes does not change v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        nw = to_newick(v)
        nw_perm = permutate_cherries(nw)
        v_perm = from_newick(nw_perm)
        assert np.array_equal(v, v_perm)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_newick_cherry_permutations_matrix(n_leaves):
    """Simple way to check that permutation of leaf nodes does not change v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """

    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)
        nw = to_newick(m)
        nw_perm = permutate_cherries(nw)
        m_perm = from_newick(nw_perm)
        assert np.allclose(m, m_perm)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_newick_ladderize_vector(n_leaves):
    """
    Simple way to check that isomorphic Newick strings
    without branch lengths have the same v

    ete's ladderize should create an isomorphism
    of the original Newick string

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        nw = to_newick(v)
        tr = Tree(nw)

        # Ladderize the tree and output a new Newick string
        tr.ladderize()
        nw_ladderized = tr.write(parser=9)

        v2 = from_newick(nw_ladderized)

        assert np.array_equal(v, v2)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_newick_ladderize_matrix(n_leaves):
    """
    Simple way to check that isomorphic Newick strings
    with branch lengths have the same matrix

    ete's ladderize should create an isomorphism
    of the original Newick string

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)
        nw = to_newick(m)
        tr = Tree(nw)

        # Ladderize the tree and output a new Newick string
        tr.ladderize()
        parser = ete4.parser.newick.make_parser(1, dist="%0.8g")
        nw_ladderized = tr.write(parser=parser)

        m2 = from_newick(nw_ladderized)

        assert np.allclose(m, m2)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_v2edges2v(n_leaves):
    """Test that v to edges to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        edges = to_edges(v)
        v2 = from_edges(edges)
        assert np.array_equal(v, v2)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_v2ancestry2v(n_leaves):
    """Test that v to ancestry to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        ancestry = to_ancestry(v)
        v2 = from_ancestry(ancestry)
        assert np.array_equal(v, v2)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_v2pairs2v(n_leaves):
    """Test that v to pairs to converted_v leads to v == converted_v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        pairs = to_pairs(v)
        v2 = from_pairs(pairs)
        assert np.array_equal(v, v2)
