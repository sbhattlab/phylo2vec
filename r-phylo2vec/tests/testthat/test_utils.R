source("config.R")

test_that(desc = "sample_vector", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Check that the vector is valid
      expect_no_error(check_v(v))
    }
  }
})

test_that(desc = "sample_matrix", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # Check that the matrix is valid
      expect_no_error(check_m(m))
    }
  }
})

test_that(desc = "find_num_leaves", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Convert to Newick
      newick <- to_newick(v)

      # Find number of leaves
      n_leaves_found <- find_num_leaves(newick)

      expect_equal(n_leaves, n_leaves_found)
    }
  }
})

test_that(desc = "remove_and_add", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      leaf <- sample.int(n_leaves, 1) - 1

      v_and_sis <- remove_leaf(v, leaf)

      v_sub <- v_and_sis$v
      sis <- v_and_sis$branch

      if (sis >= leaf) {
        sis <- sis - 1
      }

      v_add <- add_leaf(v_sub, leaf, sis)

      expect_equal(v, v_add)
    }
  }
})

test_that(desc = "get_common_ancestor", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Generate two random leaf nodes
      nodes <- sample.int(n_leaves, 2)
      node1 <- nodes[1] - 1
      node2 <- nodes[2] - 1

      # Get the common ancestor using the Phylo2Vec function
      ca <- get_common_ancestor(v, node1, node2)

      # Get the common ancestor using ape
      tr <- ape::read.tree(text = to_newick(v))

      ca_ape <- ape::getMRCA(tr, c(as.character(node1), as.character(node2)))

      ca_ape_converted <- as.integer(tr$node.label[[ca_ape - n_leaves]])

      expect_equal(ca, ca_ape_converted)
    }
  }
})

test_create_and_apply_label_mapping <- function(...) {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      tr <- ape::rtree(n_leaves, rooted = TRUE, equiprob = TRUE, ...)

      nw_str <- ape::write.tree(tr)

      out <- create_label_mapping(nw_str)

      nw_str_new <- apply_label_mapping(out$newick, out$mapping)

      expect_equal(nw_str, nw_str_new)
    }
  }
}

test_that(desc = "create_and_apply_label_mapping_vector", {
  test_create_and_apply_label_mapping(br = NULL)
})

test_that(desc = "create_and_apply_label_mapping_matrix", {
  test_create_and_apply_label_mapping()
})

test_that(desc = "remove_branch_lengths", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      m <- sample_matrix(n_leaves, FALSE)

      nw_from_m <- to_newick(m)

      nw_no_bl <- remove_branch_lengths(nw_from_m)

      v <- as.integer(m[, 1])

      nw_from_v <- to_newick(v)

      expect_equal(nw_no_bl, nw_from_v)
    }
  }
})

test_that(desc = "has_branch_lengths", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    m <- sample_matrix(n_leaves, FALSE)

    nw_from_m <- to_newick(m)

    expect_true(has_branch_lengths(nw_from_m))

    v <- as.integer(m[, 1])

    nw_from_v <- to_newick(v)

    expect_false(has_branch_lengths(nw_from_v))
  }
})

test_that(desc = "get_node_depth_root_is_zero_vector", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    v <- sample_vector(n_leaves, FALSE)
    root <- 2 * length(v) # Root node index (0-indexed)
    expect_equal(get_node_depth(v, root), 0.0)
  }
})

test_that(desc = "get_node_depth_root_is_zero_matrix", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    m <- sample_matrix(n_leaves, FALSE)
    root <- 2 * nrow(m) # Root node index (0-indexed)
    expect_equal(get_node_depth(m, root), 0.0)
  }
})

test_that(desc = "get_node_depths_root_is_zero_vector", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    v <- sample_vector(n_leaves, FALSE)
    root <- 2 * length(v)
    depths <- get_node_depths(v)
    expect_equal(depths[root + 1], 0.0) # R is 1-indexed
  }
})

test_that(desc = "get_node_depths_root_is_zero_matrix", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    m <- sample_matrix(n_leaves, FALSE)
    root <- 2 * nrow(m)
    depths <- get_node_depths(m)
    expect_equal(depths[root + 1], 0.0) # R is 1-indexed
  }
})

test_that(desc = "get_node_depth_specific_vectors", {
  # Tree: (0,(1,(2,3)4)5)6;
  # Depth is distance from root to node
  v1 <- as.integer(c(0, 1, 2))
  expect_equal(get_node_depth(v1, 6L), 0.0) # Root has depth 0
  expect_equal(get_node_depth(v1, 5L), 1.0) # Node 5 is 1 edge from root
  expect_equal(get_node_depth(v1, 4L), 2.0) # Node 4 is 2 edges from root
  expect_equal(get_node_depth(v1, 0L), 1.0) # Leaf 0 is 1 edge from root
  expect_equal(get_node_depth(v1, 1L), 2.0) # Leaf 1 is 2 edges from root
  expect_equal(get_node_depth(v1, 2L), 3.0) # Leaf 2 is 3 edges from root
  expect_equal(get_node_depth(v1, 3L), 3.0) # Leaf 3 is 3 edges from root

  # Asymmetric tree: (1,(2,(0,3)4)5)6;
  v2 <- as.integer(c(0, 0, 0))
  expect_equal(get_node_depth(v2, 6L), 0.0) # Root has depth 0
  expect_equal(get_node_depth(v2, 5L), 1.0) # Node 5 is 1 edge from root
  expect_equal(get_node_depth(v2, 4L), 2.0) # Node 4 is 2 edges from root
  expect_equal(get_node_depth(v2, 1L), 1.0) # Leaf 1 is 1 edge from root
  expect_equal(get_node_depth(v2, 2L), 2.0) # Leaf 2 is 2 edges from root
  expect_equal(get_node_depth(v2, 0L), 3.0) # Leaf 0 is 3 edges from root
  expect_equal(get_node_depth(v2, 3L), 3.0) # Leaf 3 is 3 edges from root

  # Balanced tree: ((0,1)4,(2,3)5)6;
  v3 <- as.integer(c(0, 0, 1))
  expect_equal(get_node_depth(v3, 6L), 0.0) # Root has depth 0
  expect_equal(get_node_depth(v3, 4L), 1.0) # Node 4 is 1 edge from root
  expect_equal(get_node_depth(v3, 5L), 1.0) # Node 5 is 1 edge from root
  expect_equal(get_node_depth(v3, 0L), 2.0) # All leaves are 2 edges from root
  expect_equal(get_node_depth(v3, 1L), 2.0)
  expect_equal(get_node_depth(v3, 2L), 2.0)
  expect_equal(get_node_depth(v3, 3L), 2.0)
})

test_that(desc = "get_node_depth_specific_matrices", {
  # Tree: ((0:0.3,2:0.7)3:0.5,1:0.2)4; (v = [0, 0])
  # Depth is distance from root to node
  m1 <- matrix(
    c(0, 0.3, 0.7, 0, 0.5, 0.2),
    nrow = 2,
    byrow = TRUE
  )
  expect_equal(get_node_depth(m1, 4L), 0.0, tolerance = 1e-10) # Root has depth 0
  expect_equal(get_node_depth(m1, 3L), 0.5, tolerance = 1e-10) # Node 3 is 0.5 from root
  expect_equal(get_node_depth(m1, 0L), 0.8, tolerance = 1e-10) # Leaf 0 is 0.5+0.3 from root
  expect_equal(get_node_depth(m1, 1L), 0.2, tolerance = 1e-10) # Leaf 1 is 0.2 from root
  expect_equal(get_node_depth(m1, 2L), 1.2, tolerance = 1e-10) # Leaf 2 is 0.5+0.7 from root

  # Tree: (0:0.7,(1:0.5,2:0.8)3:0.6)4; (v = [0, 1])
  m2 <- matrix(
    c(0, 0.5, 0.8, 1, 0.7, 0.6),
    nrow = 2,
    byrow = TRUE
  )
  expect_equal(get_node_depth(m2, 4L), 0.0, tolerance = 1e-10) # Root has depth 0
  expect_equal(get_node_depth(m2, 3L), 0.6, tolerance = 1e-10) # Node 3 is 0.6 from root
  expect_equal(get_node_depth(m2, 0L), 0.7, tolerance = 1e-10) # Leaf 0 is 0.7 from root
  expect_equal(get_node_depth(m2, 1L), 1.1, tolerance = 1e-10) # Leaf 1 is 0.6+0.5 from root
  expect_equal(get_node_depth(m2, 2L), 1.4, tolerance = 1e-10) # Leaf 2 is 0.6+0.8 from root
})

test_that(desc = "get_node_depths_matches_ape_node_depth_edgelength", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix (with branch lengths)
      m <- sample_matrix(n_leaves, FALSE)

      # Get depths using phylo2vec
      p2v_depths <- get_node_depths(m)

      # Convert to Newick and read with ape
      nw <- to_newick(m)
      tr <- ape::read.tree(text = nw)

      # Get depths using ape's node.depth.edgelength
      ape_depths <- ape::node.depth.edgelength(tr)

      # Reorder phylo2vec depths to match ape's node ordering:
      # ape orders as [tips in newick order, internal nodes in newick order]
      # tip.label and node.label contain the phylo2vec indices as strings
      p2v_depths_reordered <- p2v_depths[
        as.integer(c(tr$tip.label, tr$node.label)) + 1
      ]

      expect_equal(p2v_depths_reordered, ape_depths, tolerance = 1e-10)
    }
  }
})
