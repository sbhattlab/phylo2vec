"""Tests for utility functions."""

import secrets

import numpy as np
import pytest

from ete4 import Tree

from phylo2vec.base.newick import to_newick
from phylo2vec.utils.matrix import check_matrix, sample_matrix
from phylo2vec.utils.vector import (
    add_leaf,
    check_vector,
    get_common_ancestor,
    get_node_depth,
    get_node_depths,
    remove_leaf,
    reroot,
    reroot_at_random,
    sample_vector,
)
from phylo2vec.utils.newick import (
    apply_label_mapping,
    create_label_mapping,
    find_num_leaves,
    remove_branch_lengths,
)

from .config import MAX_N_LEAVES, MIN_N_LEAVES, N_REPEATS


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_sample_vector(n_leaves):
    """Test the sample_vector function

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        check_vector(v)  # Asserts that v is valid


class TestSampleVectorEdgeCases:
    def test_sample_vector_negative(self):
        """Test the sample_vector function with negative n_leaves"""
        with pytest.raises(ValueError):
            sample_vector(-1)

    def test_sample_vector_zero(self):
        """Test the sample_vector function with zero leaves"""
        with pytest.raises(ValueError):
            sample_vector(0)

    def test_sample_vector_two(self):
        """Test the sample_vector function with two leaves"""
        v = sample_vector(2)
        check_vector(v)  # Asserts that v is valid


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_sample_matrix(n_leaves):
    """Test the sample_vector function

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)
        check_matrix(m)  # Asserts that m is valid


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_find_num_leaves(n_leaves):
    """Test find_num_leaves

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        newick = to_newick(v)
        # Assert that find_num_leaves returns the true number of leaves in the the tree
        assert find_num_leaves(newick) == n_leaves


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_create_and_apply_label_mapping(n_leaves):
    """Test create_label_mapping and apply_label_mapping

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        # Random string Newick
        t = Tree()
        t.populate(n_leaves)
        nw_str = t.write(parser=9)

        # Create an int-to-str label mapping and create an integer Newick
        nw_int, label_mapping = create_label_mapping(nw_str)

        # Apply the mapping to retrieve the string Newick
        new_nw_str = apply_label_mapping(nw_int, label_mapping)

        # We should have nw_str == new_nw_str
        assert nw_str == new_nw_str


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_remove_and_add(n_leaves):
    """Test removing and adding a node to a vector

    Sample a v, remove a random leaf and add it back
    Test if we recover the same v

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)

        leaf = secrets.randbelow(n_leaves)

        v_sub, sis = remove_leaf(v, leaf)

        # To deal with increment when removing a leaf
        if sis >= leaf:
            sis -= 1

        v_add = add_leaf(v_sub, leaf, sis)

        assert np.array_equal(v, v_add)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_get_common_ancestor(n_leaves):
    """Test get_common_ancestor against ete

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)

        node1, node2 = np.random.choice(np.arange(2 * (n_leaves - 1)), 2, replace=False)

        p2v_common_ancestor = get_common_ancestor(v, node1, node2)

        nw = to_newick(v)

        tr = Tree(nw, parser=8)

        ete_common_ancestor = int(tr.common_ancestor(f"{node1}", f"{node2}").name)

        assert p2v_common_ancestor == ete_common_ancestor


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_get_common_ancestor_matrix(n_leaves):
    """Test get_common_ancestor with matrix input

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)
        v = m[:, 0].astype(int)

        node1, node2 = np.random.choice(np.arange(2 * (n_leaves - 1)), 2, replace=False)

        # Matrix and vector should return the same result
        mrca_from_matrix = get_common_ancestor(m, node1, node2)
        mrca_from_vector = get_common_ancestor(v, node1, node2)

        assert mrca_from_matrix == mrca_from_vector


class TestGetCommonAncestorEdgeCases:

    def test_get_common_ancestor_negative_nodes(self):
        """Test get_common_ancestor with negative nodes"""
        v = sample_vector(5)
        # Negative node1
        with pytest.raises(ValueError):
            get_common_ancestor(v, -1, 2)

        # Negative node2
        with pytest.raises(ValueError):
            get_common_ancestor(v, 3, -1)

    def test_get_common_ancestor_node_out_of_bounds(self):
        """Test get_common_ancestor with nodes out of bounds"""
        v = sample_vector(7)
        n_leaves = len(v)
        max_node = 2 * n_leaves

        # node1 out of bounds
        with pytest.raises(ValueError):
            get_common_ancestor(v, max_node + 1, 4)

        # node2 out of bounds
        with pytest.raises(ValueError):
            get_common_ancestor(v, 0, max_node + 2)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_remove_branch_lengths(n_leaves):
    """Test removing branch lengths from a Newick string

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)

        nw_from_m = to_newick(m)

        nw_no_bl = remove_branch_lengths(nw_from_m)

        v = m[:, 0].astype(int)

        nw_from_v = to_newick(v)

        assert nw_no_bl == nw_from_v


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_reroot_vector(n_leaves):
    """
    Test reroot function on vectors

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    v = sample_vector(n_leaves)
    node = np.random.randint(0, 2 * len(v) - 1)

    v_rerooted = reroot(v, node)

    check_vector(v_rerooted)

    assert len(v) == len(v_rerooted)

    # Check that the unrooted trees are identical
    tr = Tree(to_newick(v), parser=8)
    tr.unroot()
    tr_rerooted = Tree(to_newick(v_rerooted), parser=8)
    tr_rerooted.unroot()

    rf, *_ = tr.robinson_foulds(tr_rerooted, unrooted_trees=True)
    assert rf == 0


class TestRerootEdgeCases:
    def test_reroot_negative_nodes(self):
        """Test reroot with negative nodes"""
        v = sample_vector(5)
        # Negative node
        with pytest.raises(ValueError):
            reroot(v, -1)

    def test_reroot_node_out_of_bounds(self):
        """Test reroot with nodes out of bounds"""
        v = sample_vector(7)
        n_leaves = len(v)
        max_node = 2 * n_leaves

        # Node out of bounds
        with pytest.raises(ValueError):
            reroot(v, max_node + 1)

    def test_reroot_at_current_root(self):
        """Test reroot at the current root node"""
        v = sample_vector(6)
        n_leaves = len(v)
        current_root = 2 * n_leaves

        # Reroot at current root
        with pytest.raises(ValueError):
            reroot(v, current_root)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_reroot_matrix(n_leaves):
    """
    Test reroot function on matrices

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    m = sample_matrix(n_leaves)
    node = np.random.randint(0, 2 * len(m) - 1)

    m_rerooted = reroot(m, node)

    check_matrix(m_rerooted)

    assert len(m) == len(m_rerooted)

    tr = Tree(to_newick(m), parser=1)
    tr.unroot()
    tr_rerooted = Tree(to_newick(m_rerooted), parser=1)
    tr_rerooted.unroot()

    # Check that the unrooted trees are identical
    rf, *_ = tr.robinson_foulds(tr_rerooted, unrooted_trees=True)
    assert rf == 0


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_reroot_vector_at_random(n_leaves):
    """
    Test reroot-at_random function on vectors

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    v = sample_vector(n_leaves)

    v_rerooted = reroot_at_random(v)

    check_vector(v_rerooted)


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES + 1))
def test_reroot_matrix_at_random(n_leaves):
    """
    Test reroot-at_random function on matrices

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    m = sample_matrix(n_leaves)

    m_rerooted = reroot_at_random(m)

    check_matrix(m_rerooted)


MAX_N_LEAVES_STATS = 50


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_get_node_depth_root_is_zero_vector(n_leaves):
    """Test that root node has depth 0 for vectors.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        root = 2 * len(v)  # Root node index
        assert get_node_depth(v, root) == 0.0


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_get_node_depth_root_is_zero_matrix(n_leaves):
    """Test that root node has depth 0 for matrices.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)
        root = 2 * len(m)  # Root node index
        assert get_node_depth(m, root) == 0.0


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_get_node_depth_topological_consistency(n_leaves):
    """Test that matrix depth with unit branch lengths matches vector depth.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)
        v = m[:, 0].astype(int)

        # Create matrix with unit branch lengths
        m_unit = m.copy()
        m_unit[:, 1] = 1.0
        m_unit[:, 2] = 1.0

        n_nodes = 2 * n_leaves - 1
        for node in range(n_nodes):
            depth_v = get_node_depth(v, node)
            depth_m = get_node_depth(m_unit, node)
            assert depth_v == depth_m, f"Mismatch at node {node}"


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_get_node_depths_root_is_zero_vector(n_leaves):
    """Test that root node has depth 0 in get_node_depths for vectors.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        root = 2 * len(v)
        depths = get_node_depths(v)
        assert depths[root] == 0.0


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_get_node_depths_root_is_zero_matrix(n_leaves):
    """Test that root node has depth 0 in get_node_depths for matrices.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        m = sample_matrix(n_leaves)
        root = 2 * len(m)
        depths = get_node_depths(m)
        assert depths[root] == 0.0


@pytest.mark.parametrize("n_leaves", range(MIN_N_LEAVES, MAX_N_LEAVES_STATS + 1))
def test_get_node_depths_returns_correct_length(n_leaves):
    """Test that get_node_depths returns correct length.

    Parameters
    ----------
    n_leaves : int
        Number of leaves
    """
    for _ in range(N_REPEATS):
        v = sample_vector(n_leaves)
        depths = get_node_depths(v)
        expected_len = 2 * n_leaves - 1
        assert len(depths) == expected_len


@pytest.mark.parametrize(
    "v, node, expected_depth",
    [
        # Tree: (0,(1,(2,3)4)5)6
        # Depth is distance from root to node
        (np.array([0, 1, 2]), 6, 0.0),  # Root has depth 0
        (np.array([0, 1, 2]), 5, 1.0),  # Node 5 is 1 edge from root
        (np.array([0, 1, 2]), 4, 2.0),  # Node 4 is 2 edges from root
        (np.array([0, 1, 2]), 0, 1.0),  # Leaf 0 is 1 edge from root
        (np.array([0, 1, 2]), 1, 2.0),  # Leaf 1 is 2 edges from root
        (np.array([0, 1, 2]), 2, 3.0),  # Leaf 2 is 3 edges from root
        (np.array([0, 1, 2]), 3, 3.0),  # Leaf 3 is 3 edges from root
        # Asymmetric tree: (1,(2,(0,3)4)5)6
        (np.array([0, 0, 0]), 6, 0.0),  # Root has depth 0
        (np.array([0, 0, 0]), 5, 1.0),  # Node 5 is 1 edge from root
        (np.array([0, 0, 0]), 4, 2.0),  # Node 4 is 2 edges from root
        (np.array([0, 0, 0]), 1, 1.0),  # Leaf 1 is 1 edge from root
        (np.array([0, 0, 0]), 2, 2.0),  # Leaf 2 is 2 edges from root
        (np.array([0, 0, 0]), 0, 3.0),  # Leaf 0 is 3 edges from root
        (np.array([0, 0, 0]), 3, 3.0),  # Leaf 3 is 3 edges from root
        # Balanced tree: ((0,1)4,(2,3)5)6
        (np.array([0, 0, 1]), 6, 0.0),  # Root has depth 0
        (np.array([0, 0, 1]), 4, 1.0),  # Node 4 is 1 edge from root
        (np.array([0, 0, 1]), 5, 1.0),  # Node 5 is 1 edge from root
        (np.array([0, 0, 1]), 0, 2.0),  # All leaves are 2 edges from root
        (np.array([0, 0, 1]), 1, 2.0),
        (np.array([0, 0, 1]), 2, 2.0),
        (np.array([0, 0, 1]), 3, 2.0),
    ],
)
def test_get_node_depth_specific_vectors(v, node, expected_depth):
    """Test get_node_depth with specific known vectors.

    Parameters
    ----------
    v : np.ndarray
        Input vector
    node : int
        Node to get depth for
    expected_depth : float
        Expected depth value (distance from root)
    """
    assert get_node_depth(v, node) == expected_depth


@pytest.mark.parametrize(
    "m, node, expected_depth",
    [
        # Tree: ((0:0.3,2:0.7)3:0.5,1:0.2)4  (v = [0, 0])
        # Depth is distance from root to node
        (np.array([[0.0, 0.3, 0.7], [0.0, 0.5, 0.2]]), 4, 0.0),  # Root has depth 0
        (np.array([[0.0, 0.3, 0.7], [0.0, 0.5, 0.2]]), 3, 0.5),  # Node 3 is 0.5 from root
        (np.array([[0.0, 0.3, 0.7], [0.0, 0.5, 0.2]]), 0, 0.8),  # Leaf 0 is 0.5+0.3 from root
        (np.array([[0.0, 0.3, 0.7], [0.0, 0.5, 0.2]]), 1, 0.2),  # Leaf 1 is 0.2 from root
        (np.array([[0.0, 0.3, 0.7], [0.0, 0.5, 0.2]]), 2, 1.2),  # Leaf 2 is 0.5+0.7 from root
        # Tree: (0:0.7,(1:0.5,2:0.8)3:0.6)4  (v = [0, 1])
        (np.array([[0.0, 0.5, 0.8], [1.0, 0.7, 0.6]]), 4, 0.0),  # Root has depth 0
        (np.array([[0.0, 0.5, 0.8], [1.0, 0.7, 0.6]]), 3, 0.6),  # Node 3 is 0.6 from root
        (np.array([[0.0, 0.5, 0.8], [1.0, 0.7, 0.6]]), 0, 0.7),  # Leaf 0 is 0.7 from root
        (np.array([[0.0, 0.5, 0.8], [1.0, 0.7, 0.6]]), 1, 1.1),  # Leaf 1 is 0.6+0.5 from root
        (np.array([[0.0, 0.5, 0.8], [1.0, 0.7, 0.6]]), 2, 1.4),  # Leaf 2 is 0.6+0.8 from root
    ],
)
def test_get_node_depth_specific_matrices(m, node, expected_depth):
    """Test get_node_depth with specific known matrices.

    Parameters
    ----------
    m : np.ndarray
        Input matrix
    node : int
        Node to get depth for
    expected_depth : float
        Expected depth value (distance from root)
    """
    assert abs(get_node_depth(m, node) - expected_depth) < 1e-10


class TestGetNodeDepthEdgeCases:
    """Edge case tests for get_node_depth."""

    def test_get_node_depth_negative_node(self):
        """Test that negative nodes raise an error."""
        v = sample_vector(5)
        with pytest.raises(ValueError):
            get_node_depth(v, -1)

    def test_get_node_depth_node_out_of_bounds(self):
        """Test that out of bounds nodes raise an error."""
        v = sample_vector(5)
        n_leaves = len(v) + 1
        max_node = 2 * n_leaves - 1
        with pytest.raises(ValueError):
            get_node_depth(v, max_node + 1)
