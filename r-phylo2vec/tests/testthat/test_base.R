source("config.R")

library(ape)

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

test_that(desc = "m2newick2m", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      m <- sample_matrix(n_leaves, FALSE)

      # Convert to Newick and back
      newick <- to_newick_from_matrix(m)
      m2 <- to_matrix(newick)
      expect_equal(m, m2)
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

      # Convert to ancestry and back
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

      # Convert to ancestry and back
      edges <- to_edges(v)
      v2 <- from_edges(edges)
      expect_equal(v, v2)
    }
  }
})
