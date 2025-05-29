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
      tr <- read.tree(text = to_newick(v))

      ca_ape <- getMRCA(tr, c(as.character(node1), as.character(node2)))

      ca_ape_converted <- as.integer(tr$node.label[[ca_ape - n_leaves]])

      expect_equal(ca, ca_ape_converted)
    }
  }
})
