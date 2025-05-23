source("config.R")

library(ape)

# 50 tests with N_REPEATS repeats
N_TESTS <- 50
MAX_N_LEAVES <- MIN_N_LEAVES + N_TESTS - 1

#' Adaptation of numpy's allclose function in R
#' @param a First vector
#' @param b Second vector
#' @param atol Absolute tolerance
#' @param rtol Relative tolerance
#' @return TRUE if the two vectors are close, FALSE otherwise
allclose <- function(a, b, atol = 1e-8, rtol = 1e-5) {
  all(abs(a - b) <= atol + rtol * abs(b))
}

test_that(desc = "cophenetic_vector", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)
      newick <- to_newick_from_vector(v)

      # Convert to Newick and back
      coph_p2v <- cophenetic_distances(v)
      tr <- read.tree(text = newick)
      tr <- compute.brlen(tr, 1)
      coph_ape <- cophenetic(tr)
      col_order <- order(as.numeric(colnames(coph_ape)))
      coph_ape <- coph_ape[col_order, col_order]

      expect_equal(coph_p2v, coph_ape)
    }
  }
})

test_that(desc = "cophenetic_matrix", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      m <- sample_matrix(n_leaves, FALSE)
      newick <- to_newick_from_matrix(m)

      # Convert to Newick and back
      coph_p2v <- cophenetic_distances(m)
      tr <- read.tree(text = newick)
      coph_ape <- cophenetic(tr)
      col_order <- order(as.numeric(colnames(coph_ape)))
      coph_ape <- coph_ape[col_order, col_order]

      expect_true(allclose(coph_p2v, coph_ape))
    }
  }
})
