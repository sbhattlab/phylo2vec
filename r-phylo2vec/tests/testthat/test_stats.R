source("config.R")

MAX_N_LEAVES_STATS <- 50

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
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Cophenetic distances from a vector
      coph_p2v <- cophenetic_distances(v)

      # Cophenetic distances from a tree using ape
      newick <- to_newick(v)
      tr <- ape::read.tree(text = newick)
      tr <- ape::compute.brlen(tr, 1)
      coph_ape <- ape::cophenetic.phylo(tr)
      col_order <- order(as.numeric(colnames(coph_ape)))
      coph_ape <- coph_ape[col_order, col_order]

      expect_equal(coph_p2v, coph_ape)
    }
  }
})

test_that(desc = "cophenetic_matrix", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # # Cophenetic distances from a matrix
      coph_p2v <- cophenetic_distances(m)

      # Cophenetic distances from a tree using ape
      newick <- to_newick(m)
      tr <- ape::read.tree(text = newick)
      coph_ape <- ape::cophenetic.phylo(tr)
      col_order <- order(as.numeric(colnames(coph_ape)))
      coph_ape <- coph_ape[col_order, col_order]

      expect_true(allclose(coph_p2v, coph_ape))
    }
  }
})

test_that(desc = "vcv_vector", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Variance-covariance matrix from a vector
      vcv_p2v <- vcovp(v)

      # Variance-covariance matrix from a tree using ape
      newick <- to_newick(v)
      tr <- ape::read.tree(text = newick)
      # Set branch lengths to 1
      tr <- ape::compute.brlen(tr, 1)
      vcv_ape <- ape::vcv.phylo(tr)
      col_order <- order(as.numeric(colnames(vcv_ape)))
      vcv_ape <- vcv_ape[col_order, col_order]

      expect_true(allclose(vcv_p2v, vcv_ape))
    }
  }
})

test_that(desc = "vcv_matrix", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # Variance-covariance matrix from a matrix
      vcv_p2v <- vcovp(m)

      # Variance-covariance matrix from a tree using ape
      newick <- to_newick(m)
      tr <- ape::read.tree(text = newick)
      vcv_ape <- ape::vcv.phylo(tr)
      col_order <- order(as.numeric(colnames(vcv_ape)))
      vcv_ape <- vcv_ape[col_order, col_order]

      expect_true(allclose(vcv_p2v, vcv_ape))
    }
  }
})

test_that(desc = "vcv_and_precision_vector", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random vector
      v <- sample_vector(n_leaves, FALSE)

      # Test that the variance-covariance and precision matrices are inverses
      # vcov_from_vector = same as vcovp for vectors
      vcv_p2v <- vcov_from_vector(v)
      prec_p2v <- precision(v)
      identity <- diag(nrow(vcv_p2v))

      expect_true(allclose(vcv_p2v %*% prec_p2v, identity))
    }
  }
})

test_that(desc = "vcv_and_precision_matrix", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    for (j in seq_len(N_REPEATS)) {
      # Generate a random matrix
      m <- sample_matrix(n_leaves, FALSE)

      # Test that the variance-covariance and precision matrices are inverses
      # vcov_from_vector = same as vcovp for matrices
      vcv_p2v <- vcov_from_matrix(m)
      prec_p2v <- precision(m)
      identity <- diag(nrow(vcv_p2v))

      expect_true(allclose(vcv_p2v %*% prec_p2v, identity))
    }
  }
})

test_incidence_single <- function(v, d) {
  k <- length(v)
  # Check that format CSC (C) and CSR (R) are equivalent
  expect_equal(incidence(v, "C"), incidence(v, "R"))
  # Check that format CSC (C) and COO (T) are equivalent
  expect_equal(incidence(v, "C"), incidence(v, "T"))
  # Check that format DENSE (D) matches expected output
  d_from_c <- as.matrix(incidence(v, "C"))
  expect_equal(d_from_c, d)
  rownames(d_from_c) <- 0:(2 * k)
  colnames(d_from_c) <- 0:(2 * k - 1)
  expect_equal(incidence(v, "D"), d_from_c)
  # Check that unallowed formats trigger an error
  expect_error(incidence(v, "unknown"))
}

test_that(desc = "incidence_all_format", {
  v1 <- as.integer(c(0))
  # fmt: skip
  d1 <- matrix(
    c(
      1, 0,
      0, 1,
      -1, -1
    ),
    nrow = 3, ncol = 2, byrow = TRUE
  )
  v2 <- as.integer(c(0, 1))
  # fmt: skip
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
  # fmt: skip
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

# Robinson-Foulds distance tests

test_that(desc = "robinson_foulds_identical_vectors", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    v <- sample_vector(n_leaves, FALSE)
    expect_equal(robinson_foulds(v, v), 0)
  }
})

test_that(desc = "robinson_foulds_identical_matrices", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    m <- sample_matrix(n_leaves, FALSE)
    expect_equal(robinson_foulds(m, m), 0)
  }
})

test_that(desc = "robinson_foulds_symmetric", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    v1 <- sample_vector(n_leaves, FALSE)
    v2 <- sample_vector(n_leaves, FALSE)
    expect_equal(robinson_foulds(v1, v2), robinson_foulds(v2, v1))
  }
})

test_that(desc = "robinson_foulds_normalized_bounds", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    v1 <- sample_vector(n_leaves, FALSE)
    v2 <- sample_vector(n_leaves, FALSE)
    rf_norm <- robinson_foulds(v1, v2, normalize = TRUE)
    expect_true(rf_norm >= 0 && rf_norm <= 1)
  }
})

test_that(desc = "robinson_foulds_matches_treedist", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_STATS)) {
    for (j in seq_len(N_REPEATS)) {
      v1 <- sample_vector(n_leaves, FALSE)
      v2 <- sample_vector(n_leaves, FALSE)

      # Our implementation
      rf_ours <- robinson_foulds(v1, v2, normalize = FALSE)

      # TreeDist reference
      t1 <- ape::read.tree(text = to_newick(v1))
      t2 <- ape::read.tree(text = to_newick(v2))
      rf_treedist <- TreeDist::RobinsonFoulds(t1, t2)

      # Check that our RF matches treedist's RF
      expect_equal(rf_ours, rf_treedist)

      rf_ours_norm <- robinson_foulds(v1, v2, normalize = TRUE)
      rf_treedist_norm <- TreeDist::RobinsonFoulds(t1, t2, normalize = TRUE)

      # Check that our normalized RF matches treedist's normalized RF
      expect_equal(rf_ours_norm, rf_treedist_norm)
    }
  }
})
