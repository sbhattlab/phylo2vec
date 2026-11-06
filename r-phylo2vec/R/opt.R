#' Balanced minimum evolution (BME) length of a tree
#'
#' Pauplin's tree length `L = sum_{i < j} 2^(1 - tau_ij) * dm[i, j]`,
#' where `tau_ij` is the number of edges between leaves i and j
#' in the unrooted tree (Pauplin, 2000). Lower is better.
#'
#' @param tree phylo2vec vector (ndim == 1)/matrix (ndim == 2).
#'   Only the topology is used: the branch lengths of a matrix are ignored.
#' @param dm Symmetric n x n distance matrix, indexed by leaf label
#'   (row and column i + 1 for leaf i).
#' @return BME length of the tree (numeric)
#' @examples
#' dm <- matrix(c(0, 1, 2, 1, 0, 2, 2, 2, 0), nrow = 3)
#' bme_objective(c(0, 0), dm) # every pair is 2 edges apart: 2.5
#' @export
bme_objective <- function(tree, dm) {
  v <- bme_topology(tree)
  .Call(wrap__bme_objective, v, bme_distance_matrix(dm, length(v) + 1))
}

#' Balanced minimum evolution (BME) branch lengths for the topology of a tree
#'
#' FastME/Desper & Gascuel (2002) BME lengths of the edges of the unrooted
#' tree, given `dm`. They sum to [bme_objective()].
#'
#' The lengths are not clamped and can be negative, a sign that the topology
#' fits `dm` poorly. A phylo2vec matrix needs non-negative branch lengths,
#' so clamp them before building one: `cbind(v, pmax(bls, 0))`.
#'
#' @inheritParams bme_objective
#' @return Matrix of branch lengths of shape (n - 1, 2), in the layout of the
#'   branch lengths of a phylo2vec matrix: row i holds the lengths of the edges
#'   from internal node n + i - 1 (0-indexed) to its two children.
#'   The unrooted tree has one edge through the root, whose length is split
#'   equally between the root's two child edges.
#' @examples
#' # On distances measured along a tree, the BME branch lengths are exact
#' m <- matrix(c(0, 2, 2, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6), nrow = 3)
#' bls <- bme_branch_lengths(m, cophenetic_distances(m))
#' all.equal(bls[1:2, ], m[1:2, 2:3])
#' bls[3, ] # root edge (0.5 + 0.6), split equally
#' @export
bme_branch_lengths <- function(tree, dm) {
  v <- bme_topology(tree)
  .Call(wrap__bme_branch_lengths, v, bme_distance_matrix(dm, length(v) + 1))
}

# Topology of a phylo2vec vector or matrix, as an integer vector
bme_topology <- function(tree) {
  if (is.matrix(tree)) as.integer(tree[, 1]) else as.integer(tree)
}

# Distance matrix as a double matrix, so that integer matrices are accepted.
# Checks its shape here: an error returned from Rust would only reach R as
# "User function panicked".
bme_distance_matrix <- function(dm, n_leaves) {
  dm <- as.matrix(dm)
  if (nrow(dm) != n_leaves || ncol(dm) != n_leaves) {
    stop(sprintf(
      "Distance matrix must be %d x %d for a tree with %d leaves, got %d x %d",
      n_leaves, n_leaves, n_leaves, nrow(dm), ncol(dm)
    ))
  }
  storage.mode(dm) <- "double"
  dm
}
