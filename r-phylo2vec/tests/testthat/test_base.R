source("config.R")


test_that(desc = "v2newick2v", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Convert to Newick and back
      newick <- to_newick_from_vector(v)
      v2 <- to_vector(newick)
      expect_equal(v, v2)
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

test_that(desc = "v2edges2v", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Convert to ancestry and back
      edges <- to_edges(v)
      v2 <- from_edges(edges)
      expect_equal(v, v2)
    }
  }
})
