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

      # Cophenetic distances from a vector
      coph_p2v <- cophenetic_distances(v)

      # Cophenetic distances from a tree using ape
      newick <- to_newick(v)
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
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # # Cophenetic distances from a matrix
      coph_p2v <- cophenetic_distances(m)

      # Cophenetic distances from a tree using ape
      newick <- to_newick(m)
      tr <- read.tree(text = newick)
      coph_ape <- cophenetic(tr)
      col_order <- order(as.numeric(colnames(coph_ape)))
      coph_ape <- coph_ape[col_order, col_order]

      expect_true(allclose(coph_p2v, coph_ape))
    }
  }
})

test_that(desc = "vcv_vector", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Variance-covariance matrix from a vector
      vcv_p2v <- vcovp(v)

      # Variance-covariance matrix from a tree using ape
      newick <- to_newick(v)
      tr <- read.tree(text = newick)
      # Set branch lengths to 1
      tr <- compute.brlen(tr, 1)
      vcv_ape <- vcv.phylo(tr)
      col_order <- order(as.numeric(colnames(vcv_ape)))
      vcv_ape <- vcv_ape[col_order, col_order]

      expect_true(allclose(vcv_p2v, vcv_ape))
    }
  }
})

test_that(desc = "vcv_matrix", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # Variance-covariance matrix from a matrix
      vcv_p2v <- vcovp(m)

      # Variance-covariance matrix from a tree using ape
      newick <- to_newick(m)
      tr <- read.tree(text = newick)
      vcv_ape <- vcv.phylo(tr)
      col_order <- order(as.numeric(colnames(vcv_ape)))
      vcv_ape <- vcv_ape[col_order, col_order]

      expect_true(allclose(vcv_p2v, vcv_ape))
    }
  }
})

test_that(desc = "vcv_and_precision", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # Test that the variance-covariance and precision matrices are inverses
      vcv_p2v <- vcovp(m)
      prec_p2v <- precision(m)
      identity <- diag(nrow(vcv_p2v))

      expect_true(allclose(vcv_p2v %*% prec_p2v, identity))
    }
  }
})

test_incidence_single <- function(v, d) {
  k <- length(v)
  expect_equal(incidence(v, "C"), incidence(v, "R"))
  expect_equal(incidence(v, "C"), incidence(v, "T"))
  d_from_c <- as.matrix(incidence(v, "C"))
  expect_equal(d_from_c, d)
  rownames(d_from_c) <- 0:(2 * k)
  colnames(d_from_c) <- 0:(2 * k - 1)
  expect_equal(incidence(v, "D"), d_from_c)
}

test_that(desc = "incidence_all_format", {
  v1 <- as.integer(c(0))
  d1 <- matrix(
    c(
      1, 0,
      0, 1,
      -1, -1
    ),
    nrow = 3, ncol = 2, byrow = TRUE
  )
  v2 <- as.integer(c(0, 1))
  d2 <- matrix(
    c(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, -1, -1, 1,
      -1, 0, 0, -1
    ),
    nrow = 5, ncol = 4, byrow = TRUE
  )
  v3 <- as.integer(c(0, 1, 2))
  d3 <- matrix(
    c(
      1, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0,
      0, 0, 0, 1, 0, 0,
      0, 0, -1, -1, 1, 0,
      0, -1, 0, 0, -1, 1,
      -1, 0, 0, 0, 0, -1
    ),
    nrow = 7, ncol = 6, byrow = TRUE
  )
  test_incidence_single(v1, d1)
  test_incidence_single(v2, d2)
  test_incidence_single(v3, d3)
})
