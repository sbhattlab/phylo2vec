source("config.R")

MAX_N_LEAVES_BME <- 50

# 3-leaf star c(0, 0): every pair is 2 edges apart
THREE_LEAF_DM <- matrix(c(0, 1, 2, 1, 0, 2.5, 2, 2.5, 0), nrow = 3)

# 4-leaf tree c(0, 2, 2), i.e. ((0,1),(2,3)): pairs within a cherry are 2 edges
# apart, the others 3. Its ancestry is [[2, 3, 4], [0, 1, 5], [5, 4, 6]].
# Leaf i is row and column i + 1.
FOUR_LEAF_DM <- matrix(
  c(
    0.0, 0.3, 0.8, 0.9,
    0.3, 0.0, 0.7, 0.6,
    0.8, 0.7, 0.0, 0.4,
    0.9, 0.6, 0.4, 0.0
  ),
  nrow = 4
)

#' Random symmetric distance matrix with a zero diagonal
random_distances <- function(n_leaves) {
  dm <- matrix(0, n_leaves, n_leaves)
  dm[upper.tri(dm)] <- runif(n_leaves * (n_leaves - 1) / 2, 0.1, 1)
  dm + t(dm)
}

#' Brute force: Pauplin's formula with unrooted topological distances.
#' Only valid from 3 leaves: cophenetic_distances counts 2 edges between the
#' leaves of a 2-leaf tree, where BME counts 1.
brute_force_bme <- function(v, dm) {
  tau <- cophenetic_distances(v, unrooted = TRUE)
  upper <- upper.tri(dm)
  sum(2^(1 - tau[upper]) * dm[upper])
}

# BME: distance matrix check

test_that(desc = "bme_wrong_distance_shape", {
  for (shape in list(c(3, 3), c(5, 5), c(4, 3), c(3, 4))) {
    dm <- matrix(0, shape[1], shape[2])
    expect_error(bme_objective(c(0, 2, 2), dm), "Distance matrix must be 4 x 4")
    expect_error(bme_branch_lengths(c(0, 2, 2), dm), "Distance matrix must be 4 x 4")
  }
})

# bme_objective

test_that(desc = "bme_objective_null_trees", {
  # 1 leaf: no pairs. 2 leaves: one edge between them, so L = D01.
  expect_equal(bme_objective(integer(0), matrix(0, 1, 1)), 0)
  expect_equal(bme_objective(0, matrix(c(0, 0.7, 0.7, 0), nrow = 2)), 0.7)
})

test_that(desc = "bme_objective_three_leaves", {
  # Star: every pair is 2 edges apart, so L = (D01 + D02 + D12) / 2
  dm <- THREE_LEAF_DM
  expected <- 0.5 * (dm[1, 2] + dm[1, 3] + dm[2, 3])
  expect_equal(bme_objective(c(0, 0), dm), expected)
})

test_that(desc = "bme_objective_four_leaves", {
  # ((0,1),(2,3)): cherries weigh 1/2, the four cross pairs 1/4
  dm <- FOUR_LEAF_DM
  expected <- 0.5 * (dm[1, 2] + dm[3, 4]) +
    0.25 * (dm[1, 3] + dm[1, 4] + dm[2, 3] + dm[2, 4])
  expect_equal(bme_objective(c(0, 2, 2), dm), expected)
})

test_that(desc = "bme_objective_matches_brute_force", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_BME)) {
    for (j in seq_len(N_REPEATS)) {
      for (ordered in c(TRUE, FALSE)) {
        v <- sample_vector(n_leaves, ordered)
        dm <- random_distances(n_leaves)
        expect_equal(bme_objective(v, dm), brute_force_bme(v, dm))
      }
    }
  }
})

test_that(desc = "bme_objective_matrix_input", {
  # A matrix gives the same value as its vector: its branch lengths are ignored
  m <- sample_matrix(20, FALSE)
  dm <- random_distances(20)
  v <- as.integer(m[, 1])
  m_other_bls <- m
  m_other_bls[, 2:3] <- 3 * m[, 2:3]
  expect_identical(bme_objective(m, dm), bme_objective(v, dm))
  expect_identical(bme_objective(m_other_bls, dm), bme_objective(v, dm))
})

test_that(desc = "bme_objective_integer_distances", {
  # Integer distance matrices are accepted
  dm <- matrix(c(0L, 1L, 2L, 1L, 0L, 2L, 2L, 2L, 0L), nrow = 3)
  expect_equal(bme_objective(c(0, 0), dm), 2.5)
})

# bme_branch_lengths

test_that(desc = "bme_branch_lengths_null_trees", {
  # 1 leaf: no edges. 2 leaves: the one edge gets D01, split over the root row.
  expect_equal(dim(bme_branch_lengths(integer(0), matrix(0, 1, 1))), c(0, 2))
  bls <- bme_branch_lengths(0, matrix(c(0, 0.7, 0.7, 0), nrow = 2))
  expect_equal(bls, matrix(c(0.35, 0.35), nrow = 1))
})

test_that(desc = "bme_branch_lengths_four_leaves", {
  # ((0,1),(2,3)): Desper & Gascuel's closed forms for every edge
  dm <- FOUR_LEAF_DM
  # Edge of leaf x, with sibling y and other leaves z, w (0-indexed leaves)
  pendant <- function(x, y, z, w) {
    d <- function(a, b) dm[a + 1, b + 1]
    0.5 * (d(x, y) + 0.5 * (d(x, z) + d(x, w)) - 0.5 * (d(y, z) + d(y, w)))
  }
  internal <- 0.25 * (dm[1, 3] + dm[1, 4] + dm[2, 3] + dm[2, 4]) -
    0.5 * (dm[1, 2] + dm[3, 4])
  # Rows follow the ancestry [[2, 3, 4], [0, 1, 5], [5, 4, 6]]
  expected <- rbind(
    c(pendant(2, 3, 0, 1), pendant(3, 2, 0, 1)),
    c(pendant(0, 1, 2, 3), pendant(1, 0, 2, 3)),
    c(0.5 * internal, 0.5 * internal)
  )
  expect_equal(bme_branch_lengths(c(0, 2, 2), dm), expected)
})

test_that(desc = "bme_branch_lengths_sum_to_bme_objective", {
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_BME)) {
    for (j in seq_len(N_REPEATS)) {
      for (ordered in c(TRUE, FALSE)) {
        v <- sample_vector(n_leaves, ordered)
        dm <- random_distances(n_leaves)
        expect_equal(sum(bme_branch_lengths(v, dm)), bme_objective(v, dm))
      }
    }
  }
})

test_that(desc = "bme_branch_lengths_recover_tree_metric", {
  # On distances measured along a tree, BME recovers its branch lengths.
  # Only the split of the root edge between the root's two children is a
  # convention.
  for (n_leaves in seq(MIN_N_LEAVES, MAX_N_LEAVES_BME)) {
    for (j in seq_len(N_REPEATS)) {
      for (ordered in c(TRUE, FALSE)) {
        m <- sample_matrix(n_leaves, ordered)
        bls <- bme_branch_lengths(m, cophenetic_distances(m))
        k <- n_leaves - 1
        expect_equal(bls[-k, ], m[-k, 2:3])
        expect_equal(bls[k, ], rep(0.5 * sum(m[k, 2:3]), 2))
      }
    }
  }
})

test_that(desc = "bme_branch_lengths_negative", {
  # The distances fit ((0,2),(1,3)) better than ((0,1),(2,3)), so the internal
  # edge of ((0,1),(2,3)) is 0.25 * 4 * 0.6 - 0.5 * (1.0 + 1.0) = -0.4:
  # returned as is, not clamped
  dm <- matrix(
    c(
      0.0, 1.0, 0.6, 0.6,
      1.0, 0.0, 0.6, 0.6,
      0.6, 0.6, 0.0, 1.0,
      0.6, 0.6, 1.0, 0.0
    ),
    nrow = 4
  )
  bls <- bme_branch_lengths(c(0, 2, 2), dm)
  expect_equal(bls[3, ], c(-0.2, -0.2))
})

test_that(desc = "bme_branch_lengths_matrix_input", {
  # A matrix gives the same lengths as its vector: its branch lengths are ignored
  m <- sample_matrix(20, FALSE)
  dm <- random_distances(20)
  v <- as.integer(m[, 1])
  bls <- bme_branch_lengths(m, dm)
  expect_equal(dim(bls), c(19, 2))
  expect_identical(bls, bme_branch_lengths(v, dm))
})
