source("config.R")

test_that(desc = "v2newick2v", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Convert to Newick and back
      newick <- to_newick(v)
      v2 <- to_vector(newick)
      expect_equal(v, v2)
    }
  }
})

test_that(desc = "m2newick2m", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # Convert to Newick and back
      newick <- to_newick(m)
      m2 <- to_matrix(newick)
      expect_equal(m, m2)
    }
  }
})

test_that(desc = "m2newick2m_no_parents", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # Convert to Newick and back without parent labels
      newick <- to_newick(m)
      newick_no_parents <- remove_parent_labels(newick)
      m3 <- from_newick(newick_no_parents)
      expect_equal(m, m3)
    }
  }
})

test_that(desc = "v2ancestry2v", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Convert to ancestry and back
      ancestry <- to_ancestry(v)
      v2 <- from_ancestry(ancestry)
      expect_equal(v, v2)
    }
  }
})

test_that(desc = "v2pairs2v", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Convert to pairs and back
      pairs <- to_pairs(v)
      v2 <- from_pairs(pairs)
      expect_equal(v, v2)
    }
  }
})


test_that(desc = "v2edges2v", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Convert to edges and back
      edges <- to_edges(v)
      v2 <- from_edges(edges)
      expect_equal(v, v2)
    }
  }
})

test_that(desc = "to_newick_empty", {
  expect_error(to_newick(c()))
})

test_that(desc = "to_newick_ndim3", {
  # Generate a random 3 x 4 x 5 array
  arr <- array(runif(60), dim = c(3, 4, 5))
  expect_error(to_newick(arr))
})
